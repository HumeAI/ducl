/*
 * pwalk2.h — Data structures and constants for pwalk2
 */

#ifndef PWALK2_H
#define PWALK2_H

#include <stdint.h>
#include <stdatomic.h>
#include <pthread.h>
#include <signal.h>
#include <limits.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/stat.h>   /* struct statx */

#include "pw2_uring.h"

#ifndef __NR_statx
#error "__NR_statx not defined: kernel headers too old (need Linux >= 4.11 uapi)"
#endif

/* statx(2) via raw syscall.  The glibc wrapper only exists since 2.28;
 * calling the kernel directly lets the binary run on any glibc (we build
 * against glibc 2.17 for manylinux_2_17 wheels). Kernel >= 4.11 needed. */
static inline int
pw2_statx(int dirfd, const char *path, int flags, unsigned mask,
          struct statx *stx)
{
    return (int)syscall(__NR_statx, dirfd, path, flags, mask, stx);
}

/* Defaults */
#define PW2_DEFAULT_THREADS   64
#define PW2_MAX_THREADS       1024
#define PW2_DEFAULT_RING_DEPTH 2048
#define PW2_GETDENTS_BUFSZ    32768
#define PW2_OUTBUF_SIZE       65536
#define PW2_MAXEXFILES        512
#define PW2_STUCK_TIMEOUT     60   /* seconds before watchdog sends SIGUSR1 */
#define PW2_STALL_INTERVALS   60   /* 60 × 5s = 5 minutes of no progress → exit */

/* Worker blocking-op state (for watchdog diagnostics) */
#define PW2_OP_IDLE       0
#define PW2_OP_OPEN       1
#define PW2_OP_GETDENTS   2
#define PW2_OP_URING      3
#define PW2_OP_QUEUE_WAIT 4

/* statx mask: all fields we need */
#define PW2_STATX_MASK  (STATX_INO | STATX_TYPE | STATX_MODE | STATX_NLINK | \
                         STATX_UID | STATX_GID | STATX_SIZE | STATX_BLOCKS | \
                         STATX_ATIME | STATX_MTIME | STATX_CTIME)

/* --- Work queue entry (one directory to process) --- */
struct pw2_work {
    struct pw2_work *next;
    ino_t            parent_inode;
    struct statx     dir_stx;
    int              depth;
    uint16_t         pathlen;
    char             path[PATH_MAX];
};

/* --- Per-directory entry from getdents64 --- */
struct pw2_dirent {
    char         name[NAME_MAX + 1];
    ino_t        d_ino;
    uint8_t      d_type;
    uint8_t      is_dir;
    uint16_t     namelen;
    struct statx stx;
    int          stx_valid;
};

/* --- Work queue (FIFO, mutex+condvar) --- */
struct pw2_queue {
    struct pw2_work *head;
    struct pw2_work *tail;
    pthread_mutex_t  mutex;
    pthread_cond_t   cond;
};

/* --- Global state --- */
struct pw2_state {
    _Atomic int       dirs_in_flight;
    _Atomic int       shutdown;
    _Atomic long long total_files;
    _Atomic long long total_bytes;

    struct pw2_queue  queue;
    pthread_mutex_t   output_mutex;

    /* getdents64 concurrency gate */
    pthread_mutex_t   gd_mutex;
    pthread_cond_t    gd_cond;
    int               gd_active;
    int               gd_limit;

    /* Config (read-only after init) */
    int   max_threads;
    int   max_depth;
    int   no_snap;
    int   one_fs;
    int   ring_depth;
    dev_t root_dev;

    /* Exclude list */
    char *exclude_paths[PW2_MAXEXFILES];
    int   exclude_count;

    /* Worker array (for watchdog access) */
    struct pw2_worker *workers;
    int               num_workers;
};

/* --- Per-worker state --- */
struct pw2_worker {
    int               id;
    pthread_t         thread;          /* for pthread_kill from watchdog */
    struct pw2_ring   ring;
    int               use_sync_stat;   /* fallback if io_uring fails */
    struct statx     *statx_buf;       /* array[ring_depth] */
    struct pw2_dirent *batch[2];     /* double-buffered dirent batches */
    int               batch_count[2];
    int               batch_cap[2];
    /* Prefetch: first batch of next directory, read during io_uring wait */
    struct pw2_work  *pf_work;
    int               pf_dirfd;
    int               has_prefetch;
    int               pf_eof;        /* whether prefetch batch hit directory EOF */
    int               pf_batch_idx;  /* which batch[] the prefetch wrote to */
    /* Watchdog: track what blocking op the worker is in */
    _Atomic int       cur_op;          /* PW2_OP_* */
    _Atomic time_t    op_since;        /* when cur_op started */
    _Atomic int       timed_out;       /* set by watchdog: skip current op */
    char              cur_path[PATH_MAX]; /* path being opened (for diagnostics) */
    char              getdents_buf[PW2_GETDENTS_BUFSZ];
    char              outbuf[PW2_OUTBUF_SIZE];
    int               outbuf_used;
    long long         pending_files;
    long long         pending_bytes;
    char              pathbuf[PATH_MAX * 2];
};

/* linux_dirent64 for getdents64 syscall */
struct linux_dirent64 {
    uint64_t       d_ino;
    int64_t        d_off;
    unsigned short d_reclen;
    unsigned char  d_type;
    char           d_name[];
};

/* --- Global singleton (defined in pwalk2.c) --- */
extern struct pw2_state g_state;

/* --- pw2_output.c --- */
void pw2_csv_escape(const char *in, char *out, size_t outsize);
void pw2_format_file(struct pw2_worker *w, const char *path,
                     const char *exten, struct statx *stx,
                     ino_t parent_inode, int depth);
void pw2_format_dir(struct pw2_worker *w, const char *path,
                    const char *exten, struct statx *stx,
                    ino_t parent_inode, int depth,
                    long file_count, long dir_size);
void pw2_flush_output(struct pw2_worker *w);
void pw2_load_exclude_list(const char *fname);
int  pw2_check_exclude(const char *path);

/* --- pw2_worker.c --- */
void *pw2_worker_thread(void *arg);

/* --- Watchdog helper: name for PW2_OP_* constants --- */
static inline const char *
pw2_op_name(int op)
{
    switch (op) {
    case PW2_OP_IDLE:       return "idle";
    case PW2_OP_OPEN:       return "open()";
    case PW2_OP_GETDENTS:   return "getdents64()";
    case PW2_OP_URING:      return "io_uring";
    case PW2_OP_QUEUE_WAIT: return "queue_wait";
    default:                return "unknown";
    }
}

/* --- pwalk2.c queue operations --- */
void pw2_queue_push(struct pw2_work *item);
struct pw2_work *pw2_queue_pop(void);
struct pw2_work *pw2_queue_try_pop(void);

#endif /* PWALK2_H */
