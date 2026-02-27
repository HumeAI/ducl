/*
 * pwalk2.c — High-performance filesystem walker with io_uring STATX batching
 *
 * Work queue + thread pool architecture for maximum metadata throughput
 * on network filesystems (WekaFS, Lustre, GPFS, etc.)
 *
 * Copyright (C) 2013-2016 John F Dey (original pwalk)
 * Copyright (C) 2025 pwalk2 rewrite
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <linux/stat.h>
#include <pthread.h>
#include <signal.h>
#include <fcntl.h>

#include "pwalk2.h"

static const char *whoami  = "pwalk2";
static const char *Version = "1.0.0 2025 io_uring rewrite";

/* Global state singleton */
struct pw2_state g_state;

/* Start time for progress reporting (shared with progress thread) */
static time_t g_start_time;

/* --- Work queue operations --- */

void
pw2_queue_push(struct pw2_work *item)
{
    item->next = NULL;
    pthread_mutex_lock(&g_state.queue.mutex);
    if (g_state.queue.tail)
        g_state.queue.tail->next = item;
    else
        g_state.queue.head = item;
    g_state.queue.tail = item;
    pthread_cond_signal(&g_state.queue.cond);
    pthread_mutex_unlock(&g_state.queue.mutex);
}

struct pw2_work *
pw2_queue_pop(void)
{
    struct pw2_work *item;

    pthread_mutex_lock(&g_state.queue.mutex);
    while (!g_state.queue.head && !atomic_load(&g_state.shutdown))
        pthread_cond_wait(&g_state.queue.cond, &g_state.queue.mutex);

    if (atomic_load(&g_state.shutdown) && !g_state.queue.head) {
        pthread_mutex_unlock(&g_state.queue.mutex);
        return NULL;
    }

    item = g_state.queue.head;
    g_state.queue.head = item->next;
    if (!g_state.queue.head)
        g_state.queue.tail = NULL;
    pthread_mutex_unlock(&g_state.queue.mutex);
    return item;
}

/* Non-blocking pop: returns NULL immediately if queue is empty */
struct pw2_work *
pw2_queue_try_pop(void)
{
    struct pw2_work *item;

    pthread_mutex_lock(&g_state.queue.mutex);
    if (!g_state.queue.head) {
        pthread_mutex_unlock(&g_state.queue.mutex);
        return NULL;
    }
    item = g_state.queue.head;
    g_state.queue.head = item->next;
    if (!g_state.queue.head)
        g_state.queue.tail = NULL;
    pthread_mutex_unlock(&g_state.queue.mutex);
    return item;
}

/* --- Progress reporting thread --- */

static void
print_final_summary(void)
{
    long long files = atomic_load(&g_state.total_files);
    long long bytes = atomic_load(&g_state.total_bytes);
    time_t elapsed = time(NULL) - g_start_time;
    long rate = elapsed > 0 ? (long)(files / elapsed) : 0;

    fprintf(stderr,
        "pwalk2 done: %lld files, %.2f GB, %ld sec, %ld files/sec\n",
        files, (double)bytes / (1024.0 * 1024.0 * 1024.0),
        (long)elapsed, rate);
    fflush(stderr);
    fflush(stdout);
}

/* No-op SIGUSR1 handler: installed WITHOUT SA_RESTART so that blocked
   syscalls (open, getdents64) return EINTR when the watchdog signals. */
static void
sigusr1_handler(int sig)
{
    (void)sig;
}

/* Watchdog: check all workers for stuck blocking ops. Returns count of
   workers that were signaled. */
static int
watchdog_check(void)
{
    time_t now = time(NULL);
    int signaled = 0;

    for (int i = 0; i < g_state.num_workers; i++) {
        struct pw2_worker *w = &g_state.workers[i];
        int op = atomic_load(&w->cur_op);
        if (op == PW2_OP_IDLE || op == PW2_OP_QUEUE_WAIT || op == PW2_OP_URING)
            continue;

        time_t since = atomic_load(&w->op_since);
        if (since == 0 || now - since < PW2_STUCK_TIMEOUT)
            continue;

        /* Worker is stuck in open() or getdents64() for >30s */
        atomic_store(&w->timed_out, 1);
        pthread_kill(w->thread, SIGUSR1);
        signaled++;
        fprintf(stderr,
            "pwalk2: watchdog: worker %d stuck in %s for %lds on %s\n",
            w->id, pw2_op_name(op), (long)(now - since), w->cur_path);
    }

    if (signaled)
        fflush(stderr);
    return signaled;
}

static void *
progress_thread(void *arg)
{
    (void)arg;
    long long prev_files = 0;
    int stall_count = 0;

    sleep(2);

    while (!atomic_load(&g_state.shutdown)) {
        long long files = atomic_load(&g_state.total_files);
        long long bytes = atomic_load(&g_state.total_bytes);
        time_t elapsed = time(NULL) - g_start_time;
        long rate = elapsed > 0 ? (long)(files / elapsed) : 0;

        fprintf(stderr,
            "pwalk2: %lld files, %.2f GB, %ld sec, %ld files/sec\n",
            files, (double)bytes / (1024.0 * 1024.0 * 1024.0),
            (long)elapsed, rate);
        fflush(stderr);

        /* Watchdog: signal stuck workers every interval */
        watchdog_check();

        /* Detect stall: if no new files for 6 consecutive intervals
           (30 seconds) AND watchdog couldn't unstick anyone, force exit */
        if (files > 0 && files == prev_files)
            stall_count++;
        else
            stall_count = 0;
        prev_files = files;

        if (stall_count >= PW2_STALL_INTERVALS) {
            fprintf(stderr,
                "pwalk2: no progress for %d seconds, forcing exit "
                "(threads stuck on uninterruptible I/O)\n",
                stall_count * 5);
            print_final_summary();
            exit(0);
        }

        sleep(5);
    }

    return NULL;
}

/* --- Usage / version / header --- */

static void
print_header(void)
{
    printf("inode,parent-inode,directory-depth,\"filename\""
           ",\"fileExtension\",UID,GID,st_size,st_dev,st_blocks"
           ",st_nlink,\"st_mode\",st_atime,st_mtime,st_ctime,pw_fcount"
           ",pw_dirsum\n");
}

static void
print_help(void)
{
    printf("Usage: %s [options] <directory>\n", whoami);
    printf("Flags: --help --version\n");
    printf("       --depth n     Stop walking at depth n\n");
    printf("       --NoSnap      Ignore .snapshot directories\n");
    printf("       --exclude FILE File containing paths to exclude\n");
    printf("       --one-file-system / -x  Don't cross filesystem boundaries\n");
    printf("       --threads n   Worker threads (default %d, max %d)\n",
           PW2_DEFAULT_THREADS, PW2_MAX_THREADS);
    printf("       --ring-depth n io_uring ring size (default %d)\n",
           PW2_DEFAULT_RING_DEPTH);
    printf("       --getdents-max n  Max concurrent getdents64 calls (default threads/4)\n");
    printf("       --header      Print CSV header\n");
    printf("\nOutput: CSV with 17 columns, same format as pwalk.\n");
    printf("File Header:\n");
    print_header();
}

static void
print_version(void)
{
    fprintf(stderr, "%s version %s\n", whoami, Version);
    fprintf(stderr, "io_uring STATX batching for high-performance metadata scanning\n");
}

/* --- Main --- */

int
main(int argc, char *argv[])
{
    struct statx root_stx;
    int print_hdr = 0;

    if (argc < 2) {
        print_help();
        exit(EXIT_FAILURE);
    }

    /* Initialize state */
    memset(&g_state, 0, sizeof(g_state));
    g_state.max_threads = PW2_DEFAULT_THREADS;
    g_state.ring_depth  = PW2_DEFAULT_RING_DEPTH;
    g_state.gd_limit    = 0;  /* 0 = auto (set after parsing) */
    pthread_mutex_init(&g_state.queue.mutex, NULL);
    pthread_cond_init(&g_state.queue.cond, NULL);
    pthread_mutex_init(&g_state.output_mutex, NULL);
    pthread_mutex_init(&g_state.gd_mutex, NULL);
    pthread_cond_init(&g_state.gd_cond, NULL);

    /* Parse arguments */
    argc--; argv++;
    while (argc > 0 && argv[0][0] == '-') {
        if (!strcmp(*argv, "--NoSnap")) {
            g_state.no_snap = 1;
        } else if (!strcmp(*argv, "--depth")) {
            argc--; argv++;
            if (argc <= 0) { fprintf(stderr, "--depth requires argument\n"); exit(1); }
            g_state.max_depth = atoi(*argv);
        } else if (!strcmp(*argv, "--help")) {
            print_help();
            exit(0);
        } else if (!strcmp(*argv, "--version") || !strcmp(*argv, "-v")) {
            print_version();
            exit(0);
        } else if (!strcmp(*argv, "--header")) {
            print_hdr = 1;
        } else if (!strcmp(*argv, "--exclude")) {
            argc--; argv++;
            if (argc <= 0) { fprintf(stderr, "--exclude requires argument\n"); exit(1); }
            pw2_load_exclude_list(*argv);
        } else if (!strcmp(*argv, "--threads")) {
            argc--; argv++;
            if (argc <= 0) { fprintf(stderr, "--threads requires argument\n"); exit(1); }
            g_state.max_threads = atoi(*argv);
            if (g_state.max_threads < 1) g_state.max_threads = 1;
            if (g_state.max_threads > PW2_MAX_THREADS) g_state.max_threads = PW2_MAX_THREADS;
        } else if (!strcmp(*argv, "--ring-depth")) {
            argc--; argv++;
            if (argc <= 0) { fprintf(stderr, "--ring-depth requires argument\n"); exit(1); }
            g_state.ring_depth = atoi(*argv);
            if (g_state.ring_depth < 16) g_state.ring_depth = 16;
            if (g_state.ring_depth > 4096) g_state.ring_depth = 4096;
        } else if (!strcmp(*argv, "--getdents-max")) {
            argc--; argv++;
            if (argc <= 0) { fprintf(stderr, "--getdents-max requires argument\n"); exit(1); }
            g_state.gd_limit = atoi(*argv);
            if (g_state.gd_limit < 1) g_state.gd_limit = 1;
        } else if (!strcmp(*argv, "--one-file-system") || !strcmp(*argv, "-x")) {
            g_state.one_fs = 1;
        } else {
            fprintf(stderr, "Unknown option: %s\n", *argv);
            exit(1);
        }
        argc--; argv++;
    }

    if (argc < 1) {
        fprintf(stderr, "%s: missing directory argument\n", whoami);
        exit(EXIT_FAILURE);
    }

    /* Default getdents concurrency: unlimited (= threads).
       Use --getdents-max to reduce if VFS spinlock contention is observed. */
    if (g_state.gd_limit == 0)
        g_state.gd_limit = g_state.max_threads;

    if (print_hdr)
        print_header();

    /* Try setuid root for full access */
    if (setuid((uid_t)0))
        fprintf(stderr, "unable to setuid root; not all files may be processed\n");

    /* Stat the root directory */
    if (statx(AT_FDCWD, *argv, AT_SYMLINK_NOFOLLOW, PW2_STATX_MASK, &root_stx) < 0) {
        fprintf(stderr, "%s: statx '%s': %s\n", whoami, *argv, strerror(errno));
        exit(errno);
    }
    g_state.root_dev = makedev(root_stx.stx_dev_major, root_stx.stx_dev_minor);

    /* Seed work queue with root directory */
    {
        struct pw2_work *root = calloc(1, sizeof(struct pw2_work));
        if (!root) { perror("calloc"); exit(1); }
        strncpy(root->path, *argv, PATH_MAX - 1);
        root->pathlen = strlen(root->path);
        /* Remove trailing slash if present (unless root "/") */
        if (root->pathlen > 1 && root->path[root->pathlen - 1] == '/') {
            root->path[--root->pathlen] = '\0';
        }
        root->depth = 0;
        root->parent_inode = 0;
        memcpy(&root->dir_stx, &root_stx, sizeof(struct statx));
        atomic_store(&g_state.dirs_in_flight, 1);
        pw2_queue_push(root);
    }

    fflush(stdout);
    g_start_time = time(NULL);

    /* Install SIGUSR1 handler (no SA_RESTART: blocked syscalls get EINTR) */
    {
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = sigusr1_handler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = 0;  /* NO SA_RESTART — this is the key */
        sigaction(SIGUSR1, &sa, NULL);
    }

    /* Launch worker threads */
    {
        struct pw2_worker *workers = calloc(g_state.max_threads, sizeof(struct pw2_worker));
        if (!workers) { perror("calloc"); exit(1); }
        g_state.workers = workers;
        g_state.num_workers = g_state.max_threads;

        for (int i = 0; i < g_state.max_threads; i++) {
            workers[i].id = i;
            pthread_create(&workers[i].thread, NULL, pw2_worker_thread, &workers[i]);
        }

        /* Start progress reporting thread */
        {
            pthread_t prog_tid;
            pthread_attr_t attr;
            pthread_attr_init(&attr);
            pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
            pthread_create(&prog_tid, &attr, progress_thread, NULL);
            pthread_attr_destroy(&attr);
        }

        /* Wait for all workers to finish */
        for (int i = 0; i < g_state.max_threads; i++)
            pthread_join(workers[i].thread, NULL);

        /* Signal progress thread to stop, print final summary */
        atomic_store(&g_state.shutdown, 1);
        print_final_summary();

        free(workers);
    }

    return 0;
}
