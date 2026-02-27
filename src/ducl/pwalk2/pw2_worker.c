/*
 * pw2_worker.c — Worker loop with pipelined io_uring STATX + getdents64
 *
 * Pipeline: double-buffered batches of ~ring_depth entries.
 * While io_uring processes statx for batch A, getdents64 reads batch B.
 * This overlaps the two main sources of network metadata latency and
 * bounds memory to 2 x ring_depth entries per worker (~2MB).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/sysmacros.h>
#include <linux/stat.h>
#include <dirent.h>

#include "pwalk2.h"

#define PROGRESS_BATCH 100

/* Mark entering/leaving a blocking operation for watchdog tracking */
#define ENTER_BLOCKING(w, op, path) do { \
    strncpy((w)->cur_path, (path), PATH_MAX - 1); \
    (w)->cur_path[PATH_MAX - 1] = '\0'; \
    atomic_store(&(w)->op_since, time(NULL)); \
    atomic_store(&(w)->cur_op, (op)); \
} while (0)

#define LEAVE_BLOCKING(w) do { \
    atomic_store(&(w)->cur_op, PW2_OP_IDLE); \
    atomic_store(&(w)->op_since, (time_t)0); \
} while (0)

/* --- Helper: flush progress counters to global atomics --- */
static void
flush_progress(struct pw2_worker *w)
{
    if (w->pending_files > 0) {
        atomic_fetch_add(&g_state.total_files, w->pending_files);
        atomic_fetch_add(&g_state.total_bytes, w->pending_bytes);
        w->pending_files = 0;
        w->pending_bytes = 0;
    }
}

static void
update_progress(struct pw2_worker *w, long long size)
{
    w->pending_files++;
    w->pending_bytes += size;
    if (w->pending_files >= PROGRESS_BATCH)
        flush_progress(w);
}

/* --- Grow a dirents buffer --- */
static int
ensure_dirents_buf(struct pw2_dirent **buf, int *cap, int need)
{
    if (need <= *cap)
        return 0;
    int newcap = *cap ? *cap * 2 : 256;
    while (newcap < need)
        newcap *= 2;
    struct pw2_dirent *p = realloc(*buf, newcap * sizeof(struct pw2_dirent));
    if (!p)
        return -1;
    *buf = p;
    *cap = newcap;
    return 0;
}

/* --- Find file extension in a name (not path) --- */
static const char *
name_extension(const char *name)
{
    const char *dot = NULL;
    const char *s = name;

    if (*s == '.')
        s++;
    for (; *s; s++) {
        if (*s == '.')
            dot = s + 1;
    }
    return dot;
}

/* --- Read up to ~target directory entries via getdents64 ---
 * Calls getdents64 in a loop until *count >= target or EOF.
 * The batch may slightly exceed target (all entries from each getdents64
 * buffer are parsed).
 * Returns 0 on success, -1 on error, -2 if interrupted by watchdog timeout.
 * Sets *eof = 1 when getdents64 returns 0. */
static int
read_directory_batch(struct pw2_worker *w, int dirfd,
                     struct pw2_dirent **buf, int *count, int *cap,
                     int target, int *eof)
{
    int nread;
    *count = 0;
    *eof = 0;

    while (*count < target) {
        nread = syscall(SYS_getdents64, dirfd, w->getdents_buf,
                        sizeof(w->getdents_buf));
        if (nread == 0) {
            *eof = 1;
            break;
        }
        if (nread < 0) {
            if (errno == EINTR) {
                if (atomic_load(&w->timed_out))
                    return -2;  /* watchdog says skip this dir */
                continue;       /* spurious signal, retry */
            }
            return -1;
        }
        int bpos = 0;
        while (bpos < nread) {
            struct linux_dirent64 *de =
                (struct linux_dirent64 *)(w->getdents_buf + bpos);
            bpos += de->d_reclen;

            if (de->d_name[0] == '.') {
                if (de->d_name[1] == '\0')
                    continue;
                if (de->d_name[1] == '.' && de->d_name[2] == '\0')
                    continue;
            }

            if (ensure_dirents_buf(buf, cap, *count + 1) < 0)
                return -1;

            struct pw2_dirent *ent = &(*buf)[*count];
            memset(ent, 0, sizeof(*ent));
            ent->namelen = strlen(de->d_name);
            memcpy(ent->name, de->d_name, ent->namelen + 1);
            ent->d_ino = de->d_ino;
            ent->d_type = de->d_type;
            ent->is_dir = (de->d_type == DT_DIR);
            (*count)++;
        }
    }
    return 0;
}

/* Open a directory with watchdog tracking. Returns fd or -1 on error,
 * -2 if interrupted by watchdog timeout. */
static int
open_directory(struct pw2_worker *w, const char *path)
{
    int fd;
    ENTER_BLOCKING(w, PW2_OP_OPEN, path);
    for (;;) {
        fd = open(path, O_RDONLY | O_DIRECTORY);
        if (fd >= 0)
            break;
        if (errno == EINTR) {
            if (atomic_load(&w->timed_out)) {
                LEAVE_BLOCKING(w);
                return -2;
            }
            continue;  /* spurious signal, retry */
        }
        break;  /* real error */
    }
    LEAVE_BLOCKING(w);
    return fd;
}

/* --- getdents64 concurrency gate --- */
static void
gd_gate_enter(void)
{
    pthread_mutex_lock(&g_state.gd_mutex);
    while (g_state.gd_active >= g_state.gd_limit)
        pthread_cond_wait(&g_state.gd_cond, &g_state.gd_mutex);
    g_state.gd_active++;
    pthread_mutex_unlock(&g_state.gd_mutex);
}

/* Non-blocking: returns 1 if acquired, 0 if gate is full */
static int
gd_gate_try_enter(void)
{
    pthread_mutex_lock(&g_state.gd_mutex);
    if (g_state.gd_active >= g_state.gd_limit) {
        pthread_mutex_unlock(&g_state.gd_mutex);
        return 0;
    }
    g_state.gd_active++;
    pthread_mutex_unlock(&g_state.gd_mutex);
    return 1;
}

static void
gd_gate_leave(void)
{
    pthread_mutex_lock(&g_state.gd_mutex);
    g_state.gd_active--;
    pthread_cond_signal(&g_state.gd_cond);
    pthread_mutex_unlock(&g_state.gd_mutex);
}

/* --- Helper: decrement dirs_in_flight, trigger shutdown if last --- */
static void
finish_directory(void)
{
    if (atomic_fetch_sub(&g_state.dirs_in_flight, 1) == 1) {
        atomic_store(&g_state.shutdown, 1);
        pthread_mutex_lock(&g_state.queue.mutex);
        pthread_cond_broadcast(&g_state.queue.cond);
        pthread_mutex_unlock(&g_state.queue.mutex);
    }
}

/* ================================================================
 * io_uring statx: submit / reap operating on explicit entry buffers
 * ================================================================ */

/* Submit SQEs for entries[start_idx..total).
 * Returns the new position (start_idx + number submitted). */
static int
submit_statx_batch(struct pw2_worker *w, int dirfd,
                   struct pw2_dirent *entries, int start_idx, int total)
{
    int pos = start_idx;

    while (pos < total) {
        struct io_uring_sqe *sqe = pw2_ring_get_sqe(&w->ring);
        if (!sqe)
            break;
        int buf_idx = pos % g_state.ring_depth;
        pw2_ring_prep_statx(sqe, dirfd, entries[pos].name,
                            AT_SYMLINK_NOFOLLOW, PW2_STATX_MASK,
                            &w->statx_buf[buf_idx], (uint64_t)pos);
        pw2_ring_sqe_advance(&w->ring);
        pos++;
    }

    if (pos > start_idx) {
        int ret = pw2_ring_submit(&w->ring);
        if (ret < 0)
            fprintf(stderr, "worker %d: io_uring_submit: %s\n",
                    w->id, strerror(-ret));
    }
    return pos;
}

/* Reap exactly `count` CQEs, storing results into entries[].stx */
static void
reap_statx_batch(struct pw2_worker *w, struct pw2_dirent *entries, int count)
{
    int reaped = 0;
    while (reaped < count) {
        struct io_uring_cqe *cqe = pw2_ring_peek_cqe(&w->ring);
        if (!cqe) {
            int ret = pw2_ring_wait(&w->ring, 1);
            if (ret < 0) {
                fprintf(stderr, "worker %d: io_uring_wait: %s\n",
                        w->id, strerror(-ret));
                return;
            }
            continue;
        }
        int entry_idx = (int)cqe->user_data;
        int buf_idx = entry_idx % g_state.ring_depth;
        if (cqe->res == 0) {
            memcpy(&entries[entry_idx].stx, &w->statx_buf[buf_idx],
                   sizeof(struct statx));
            entries[entry_idx].stx_valid = 1;
            if (entries[entry_idx].d_type == DT_UNKNOWN)
                entries[entry_idx].is_dir =
                    S_ISDIR(entries[entry_idx].stx.stx_mode);
        } else {
            entries[entry_idx].stx_valid = 0;
        }
        pw2_ring_cqe_seen(&w->ring);
        reaped++;
    }
}

/* --- Stat entries using synchronous statx fallback --- */
static void
stat_entries_sync(struct pw2_worker *w, int dirfd,
                  struct pw2_dirent *entries, int count)
{
    (void)w;
    for (int i = 0; i < count; i++) {
        struct statx *stx = &entries[i].stx;
        if (statx(dirfd, entries[i].name, AT_SYMLINK_NOFOLLOW,
                  PW2_STATX_MASK, stx) == 0) {
            entries[i].stx_valid = 1;
            if (entries[i].d_type == DT_UNKNOWN)
                entries[i].is_dir = S_ISDIR(stx->stx_mode);
        } else {
            entries[i].stx_valid = 0;
        }
    }
}

/* ================================================================
 * Prefetch: open + read first batch of next directory while
 * io_uring statx completions for current directory are in-flight.
 * ================================================================ */
static void
try_prefetch(struct pw2_worker *w, int batch_idx)
{
    if (w->has_prefetch || w->use_sync_stat)
        return;

    /* Non-blocking gate: skip prefetch if too many getdents64 active */
    if (!gd_gate_try_enter())
        return;

    struct pw2_work *next = pw2_queue_try_pop();
    if (!next) {
        gd_gate_leave();
        return;
    }

    atomic_store(&w->timed_out, 0);
    int fd = open_directory(w, next->path);
    if (fd < 0) {
        if (fd == -2)
            fprintf(stderr, "Timeout Dir: %s\n", next->path);
        else
            fprintf(stderr, "Locked Dir: %s\n", next->path);
        gd_gate_leave();
        free(next);
        finish_directory();
        return;
    }

    ENTER_BLOCKING(w, PW2_OP_GETDENTS, next->path);
    int ret = read_directory_batch(w, fd, &w->batch[batch_idx],
                                   &w->batch_count[batch_idx],
                                   &w->batch_cap[batch_idx],
                                   g_state.ring_depth, &w->pf_eof);
    LEAVE_BLOCKING(w);

    if (ret < 0) {
        if (ret == -2)
            fprintf(stderr, "Timeout Dir: %s\n", next->path);
        else
            fprintf(stderr, "worker %d: prefetch getdents64 error: %s: %s\n",
                    w->id, next->path, strerror(errno));
        close(fd);
        gd_gate_leave();
        free(next);
        finish_directory();
        return;
    }

    gd_gate_leave();
    w->pf_work = next;
    w->pf_dirfd = fd;
    w->pf_batch_idx = batch_idx;
    w->has_prefetch = 1;
}

/* ================================================================
 * Process a batch of stat'd entries: emit CSV for files, push subdirs.
 * Accumulates file_count and dir_size across calls.
 * ================================================================ */
static void
process_batch(struct pw2_worker *w, struct pw2_work *work,
              struct pw2_dirent *entries, int count,
              long *file_count, long *dir_size)
{
    for (int i = 0; i < count; i++) {
        struct pw2_dirent *ent = &entries[i];

        if (!ent->stx_valid)
            continue;

        dev_t dev = makedev(ent->stx.stx_dev_major, ent->stx.stx_dev_minor);
        if (g_state.one_fs && dev != g_state.root_dev)
            continue;

        (*file_count)++;
        (*dir_size) += ent->stx.stx_size;

        if (ent->is_dir) {
            if (g_state.no_snap && strcmp(ent->name, ".snapshot") == 0)
                continue;
            if (g_state.max_depth && g_state.max_depth == work->depth)
                continue;

            int pathlen = snprintf(w->pathbuf, PATH_MAX, "%s/%s",
                                   work->path, ent->name);
            if (pathlen >= PATH_MAX)
                continue;
            if (g_state.exclude_count && pw2_check_exclude(w->pathbuf))
                continue;

            struct pw2_work *sub = malloc(sizeof(struct pw2_work));
            if (!sub)
                continue;
            memcpy(sub->path, w->pathbuf, pathlen + 1);
            sub->pathlen = pathlen;
            sub->depth = work->depth + 1;
            sub->parent_inode = work->dir_stx.stx_ino;
            memcpy(&sub->dir_stx, &ent->stx, sizeof(struct statx));
            sub->next = NULL;

            atomic_fetch_add(&g_state.dirs_in_flight, 1);
            pw2_queue_push(sub);
        } else {
            int pathlen = snprintf(w->pathbuf, sizeof(w->pathbuf), "%s/%s",
                                   work->path, ent->name);
            if (pathlen < (int)sizeof(w->pathbuf)) {
                const char *ext = name_extension(ent->name);
                pw2_format_file(w, w->pathbuf, ext, &ent->stx,
                                work->dir_stx.stx_ino, work->depth);
                update_progress(w, (long long)ent->stx.stx_size);
            }
        }
    }
}

/* --- Emit the directory summary CSV line --- */
static void
emit_dir_summary(struct pw2_worker *w, struct pw2_work *work,
                 long file_count, long dir_size)
{
    const char *ext = name_extension(
        work->path + (work->pathlen > 0 ?
            (strrchr(work->path, '/') ?
                (strrchr(work->path, '/') - work->path + 1) : 0) : 0));
    pw2_format_dir(w, work->path, ext, &work->dir_stx,
                   work->parent_inode, work->depth,
                   file_count, dir_size);
    update_progress(w, (long long)work->dir_stx.stx_size);
}

/* ================================================================
 * Worker thread main loop
 * ================================================================ */
void *
pw2_worker_thread(void *arg)
{
    struct pw2_worker *w = (struct pw2_worker *)arg;

    /* Initialize io_uring ring */
    w->use_sync_stat = 0;
    int ret = pw2_ring_init(&w->ring, g_state.ring_depth);
    if (ret < 0) {
        fprintf(stderr, "worker %d: io_uring_setup failed (%s), using sync fallback\n",
                w->id, strerror(-ret));
        w->use_sync_stat = 1;
    }

    if (!w->use_sync_stat) {
        w->statx_buf = calloc(g_state.ring_depth, sizeof(struct statx));
        if (!w->statx_buf) {
            fprintf(stderr, "worker %d: calloc statx_buf failed\n", w->id);
            pw2_ring_destroy(&w->ring);
            w->use_sync_stat = 1;
        }
    }

    /* Main loop */
    while (!atomic_load(&g_state.shutdown)) {
        struct pw2_work *work;
        int dirfd;
        int cur, other, eof;
        long file_count = 0, dir_size = 0;

        /* Reset watchdog timeout flag at start of each iteration */
        atomic_store(&w->timed_out, 0);

        /* --- Acquire directory data (from prefetch or fresh) --- */
        if (w->has_prefetch) {
            work = w->pf_work;
            dirfd = w->pf_dirfd;
            cur = w->pf_batch_idx;
            other = 1 - cur;
            eof = w->pf_eof;
            w->has_prefetch = 0;
        } else {
            atomic_store(&w->cur_op, PW2_OP_QUEUE_WAIT);
            work = pw2_queue_pop();
            atomic_store(&w->cur_op, PW2_OP_IDLE);
            if (!work)
                break;

            gd_gate_enter();
            dirfd = open_directory(w, work->path);
            if (dirfd < 0) {
                gd_gate_leave();
                if (dirfd == -2)
                    fprintf(stderr, "Timeout Dir: %s\n", work->path);
                else
                    fprintf(stderr, "Locked Dir: %s\n", work->path);
                free(work);
                finish_directory();
                continue;
            }

            cur = 0;
            other = 1;

            ENTER_BLOCKING(w, PW2_OP_GETDENTS, work->path);
            int rret = read_directory_batch(w, dirfd,
                                             &w->batch[cur],
                                             &w->batch_count[cur],
                                             &w->batch_cap[cur],
                                             g_state.ring_depth, &eof);
            LEAVE_BLOCKING(w);

            if (rret < 0) {
                gd_gate_leave();
                if (rret == -2)
                    fprintf(stderr, "Timeout Dir: %s\n", work->path);
                else
                    fprintf(stderr, "worker %d: getdents64 error: %s: %s\n",
                            w->id, work->path, strerror(errno));
                close(dirfd);
                free(work);
                finish_directory();
                continue;
            }
            gd_gate_leave();
        }

        /* --- Stat + process entries in batches --- */
        if (w->use_sync_stat) {
            /* Sync fallback: no double-buffering, one batch at a time */
            stat_entries_sync(w, dirfd, w->batch[cur], w->batch_count[cur]);
            process_batch(w, work, w->batch[cur], w->batch_count[cur],
                         &file_count, &dir_size);

            while (!eof) {
                gd_gate_enter();
                ENTER_BLOCKING(w, PW2_OP_GETDENTS, work->path);
                int rret = read_directory_batch(w, dirfd,
                                                 &w->batch[cur],
                                                 &w->batch_count[cur],
                                                 &w->batch_cap[cur],
                                                 g_state.ring_depth, &eof);
                LEAVE_BLOCKING(w);
                gd_gate_leave();

                if (rret < 0) {
                    if (rret == -2)
                        fprintf(stderr, "Timeout Dir: %s\n", work->path);
                    else
                        fprintf(stderr, "worker %d: getdents64 error: %s: %s\n",
                                w->id, work->path, strerror(errno));
                    break;
                }
                stat_entries_sync(w, dirfd, w->batch[cur],
                                  w->batch_count[cur]);
                process_batch(w, work, w->batch[cur], w->batch_count[cur],
                             &file_count, &dir_size);
            }
        } else {
            /* io_uring pipeline: double-buffered getdents64 / statx */
            int submitted = submit_statx_batch(w, dirfd, w->batch[cur],
                                               0, w->batch_count[cur]);

            while (!eof) {
                /* Read next batch while statx is in-flight */
                gd_gate_enter();
                ENTER_BLOCKING(w, PW2_OP_GETDENTS, work->path);
                int rret = read_directory_batch(w, dirfd,
                                                 &w->batch[other],
                                                 &w->batch_count[other],
                                                 &w->batch_cap[other],
                                                 g_state.ring_depth, &eof);
                LEAVE_BLOCKING(w);
                gd_gate_leave();

                if (rret < 0) {
                    if (rret == -2)
                        fprintf(stderr, "Timeout Dir: %s\n", work->path);
                    else
                        fprintf(stderr, "worker %d: getdents64 error: %s: %s\n",
                                w->id, work->path, strerror(errno));
                    break;
                }

                /* Reap statx for batch[cur] */
                ENTER_BLOCKING(w, PW2_OP_URING, work->path);
                reap_statx_batch(w, w->batch[cur], submitted);
                LEAVE_BLOCKING(w);

                /* Process batch[cur] */
                process_batch(w, work, w->batch[cur], w->batch_count[cur],
                              &file_count, &dir_size);

                /* Submit statx for batch[other] */
                submitted = submit_statx_batch(w, dirfd, w->batch[other],
                                               0, w->batch_count[other]);

                /* Swap cur/other */
                int tmp = cur;
                cur = other;
                other = tmp;
            }

            /* Overlap: prefetch next dir while last statx is in-flight */
            try_prefetch(w, other);

            /* Reap final statx batch */
            ENTER_BLOCKING(w, PW2_OP_URING, work->path);
            reap_statx_batch(w, w->batch[cur], submitted);
            LEAVE_BLOCKING(w);

            /* Process final batch */
            process_batch(w, work, w->batch[cur], w->batch_count[cur],
                          &file_count, &dir_size);
        }

        /* Directory summary line */
        emit_dir_summary(w, work, file_count, dir_size);

        close(dirfd);
        free(work);
        finish_directory();
    }

    /* Drain any unprocessed prefetch */
    if (w->has_prefetch) {
        close(w->pf_dirfd);
        free(w->pf_work);
        finish_directory();
    }

    /* Final flush */
    pw2_flush_output(w);
    flush_progress(w);

    /* Cleanup */
    if (!w->use_sync_stat) {
        pw2_ring_destroy(&w->ring);
        free(w->statx_buf);
    }
    free(w->batch[0]);
    free(w->batch[1]);

    return NULL;
}
