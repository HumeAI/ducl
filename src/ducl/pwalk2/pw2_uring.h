/*
 * pw2_uring.h — Minimal io_uring wrapper for IORING_OP_STATX
 *
 * Header-only, raw syscalls, no liburing dependency.
 * Kernel 5.15+ required for IORING_OP_STATX (opcode 21).
 */

#ifndef PW2_URING_H
#define PW2_URING_H

#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/io_uring.h>
#include <errno.h>

/* Syscall numbers (x86_64) */
#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup    425
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter    426
#endif

/* Memory barriers for shared ring head/tail */
#define pw2_smp_store_release(p, v) \
    __atomic_store_n((p), (v), __ATOMIC_RELEASE)
#define pw2_smp_load_acquire(p) \
    __atomic_load_n((p), __ATOMIC_ACQUIRE)

struct pw2_ring {
    int fd;
    unsigned sq_entries;
    unsigned cq_entries;

    /* SQ ring pointers (mapped) */
    unsigned *sq_head;
    unsigned *sq_tail;
    unsigned *sq_ring_mask;
    unsigned *sq_ring_entries;
    unsigned *sq_flags;
    unsigned *sq_array;

    /* CQ ring pointers (mapped) */
    unsigned *cq_head;
    unsigned *cq_tail;
    unsigned *cq_ring_mask;
    unsigned *cq_ring_entries;
    struct io_uring_cqe *cqes;

    /* SQE array (may be separately mapped) */
    struct io_uring_sqe *sqes;

    /* mmap tracking for cleanup */
    void   *sq_ring_ptr;
    size_t  sq_ring_sz;
    void   *cq_ring_ptr;
    size_t  cq_ring_sz;
    void   *sqes_ptr;
    size_t  sqes_sz;
};

static inline int
pw2_ring_init(struct pw2_ring *ring, unsigned entries)
{
    struct io_uring_params p;
    void *sq_ptr, *cq_ptr;

    memset(&p, 0, sizeof(p));
    memset(ring, 0, sizeof(*ring));

    ring->fd = (int)syscall(__NR_io_uring_setup, entries, &p);
    if (ring->fd < 0)
        return -errno;

    ring->sq_entries = p.sq_entries;
    ring->cq_entries = p.cq_entries;

    /* Map SQ ring */
    ring->sq_ring_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    sq_ptr = mmap(NULL, ring->sq_ring_sz, PROT_READ | PROT_WRITE,
                  MAP_SHARED | MAP_POPULATE, ring->fd, IORING_OFF_SQ_RING);
    if (sq_ptr == MAP_FAILED) {
        close(ring->fd);
        return -errno;
    }
    ring->sq_ring_ptr = sq_ptr;
    ring->sq_head         = (unsigned *)((char *)sq_ptr + p.sq_off.head);
    ring->sq_tail         = (unsigned *)((char *)sq_ptr + p.sq_off.tail);
    ring->sq_ring_mask    = (unsigned *)((char *)sq_ptr + p.sq_off.ring_mask);
    ring->sq_ring_entries = (unsigned *)((char *)sq_ptr + p.sq_off.ring_entries);
    ring->sq_flags        = (unsigned *)((char *)sq_ptr + p.sq_off.flags);
    ring->sq_array        = (unsigned *)((char *)sq_ptr + p.sq_off.array);

    /* Map CQ ring */
    ring->cq_ring_sz = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    cq_ptr = mmap(NULL, ring->cq_ring_sz, PROT_READ | PROT_WRITE,
                  MAP_SHARED | MAP_POPULATE, ring->fd, IORING_OFF_CQ_RING);
    if (cq_ptr == MAP_FAILED) {
        munmap(sq_ptr, ring->sq_ring_sz);
        close(ring->fd);
        return -errno;
    }
    ring->cq_ring_ptr = cq_ptr;
    ring->cq_head         = (unsigned *)((char *)cq_ptr + p.cq_off.head);
    ring->cq_tail         = (unsigned *)((char *)cq_ptr + p.cq_off.tail);
    ring->cq_ring_mask    = (unsigned *)((char *)cq_ptr + p.cq_off.ring_mask);
    ring->cq_ring_entries = (unsigned *)((char *)cq_ptr + p.cq_off.ring_entries);
    ring->cqes            = (struct io_uring_cqe *)((char *)cq_ptr + p.cq_off.cqes);

    /* Map SQE array */
    ring->sqes_sz = p.sq_entries * sizeof(struct io_uring_sqe);
    ring->sqes_ptr = mmap(NULL, ring->sqes_sz, PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_POPULATE, ring->fd, IORING_OFF_SQES);
    if (ring->sqes_ptr == MAP_FAILED) {
        munmap(cq_ptr, ring->cq_ring_sz);
        munmap(sq_ptr, ring->sq_ring_sz);
        close(ring->fd);
        return -errno;
    }
    ring->sqes = (struct io_uring_sqe *)ring->sqes_ptr;

    return 0;
}

static inline void
pw2_ring_destroy(struct pw2_ring *ring)
{
    if (ring->sqes_ptr)
        munmap(ring->sqes_ptr, ring->sqes_sz);
    if (ring->cq_ring_ptr)
        munmap(ring->cq_ring_ptr, ring->cq_ring_sz);
    if (ring->sq_ring_ptr)
        munmap(ring->sq_ring_ptr, ring->sq_ring_sz);
    if (ring->fd >= 0)
        close(ring->fd);
    memset(ring, 0, sizeof(*ring));
    ring->fd = -1;
}

/* Get next available SQE slot, or NULL if ring is full */
static inline struct io_uring_sqe *
pw2_ring_get_sqe(struct pw2_ring *ring)
{
    unsigned head = pw2_smp_load_acquire(ring->sq_head);
    unsigned tail = *ring->sq_tail;
    unsigned mask = *ring->sq_ring_mask;

    if (tail - head >= ring->sq_entries)
        return NULL;  /* ring full */

    struct io_uring_sqe *sqe = &ring->sqes[tail & mask];
    memset(sqe, 0, sizeof(*sqe));
    return sqe;
}

/* Advance SQ tail after filling an SQE (call after pw2_ring_get_sqe) */
static inline void
pw2_ring_sqe_advance(struct pw2_ring *ring)
{
    unsigned tail = *ring->sq_tail;
    unsigned mask = *ring->sq_ring_mask;

    ring->sq_array[tail & mask] = tail & mask;
    pw2_smp_store_release(ring->sq_tail, tail + 1);
}

/* Prepare an SQE for IORING_OP_STATX */
static inline void
pw2_ring_prep_statx(struct io_uring_sqe *sqe, int dirfd, const char *name,
                    int flags, unsigned mask, struct statx *stxbuf,
                    uint64_t user_data)
{
    sqe->opcode    = IORING_OP_STATX;
    sqe->fd        = dirfd;
    sqe->addr      = (uint64_t)(unsigned long)name;
    sqe->len       = mask;
    sqe->off       = (uint64_t)(unsigned long)stxbuf;
    sqe->statx_flags = flags;
    sqe->user_data = user_data;
}

/* Submit pending SQEs to the kernel (retries on EINTR from signals) */
static inline int
pw2_ring_submit(struct pw2_ring *ring)
{
    unsigned submitted = *ring->sq_tail - pw2_smp_load_acquire(ring->sq_head);
    if (submitted == 0)
        return 0;
    int ret;
    do {
        ret = (int)syscall(__NR_io_uring_enter, ring->fd, submitted, 0, 0, NULL, 0);
    } while (ret < 0 && errno == EINTR);
    if (ret < 0)
        return -errno;
    return ret;
}

/* Wait for at least min_complete CQEs (retries on EINTR from signals) */
static inline int
pw2_ring_wait(struct pw2_ring *ring, unsigned min_complete)
{
    int ret;
    do {
        ret = (int)syscall(__NR_io_uring_enter, ring->fd, 0, min_complete,
                           IORING_ENTER_GETEVENTS, NULL, 0);
    } while (ret < 0 && errno == EINTR);
    if (ret < 0)
        return -errno;
    return ret;
}

/* Peek at the next CQE without consuming it; returns NULL if none ready */
static inline struct io_uring_cqe *
pw2_ring_peek_cqe(struct pw2_ring *ring)
{
    unsigned head = pw2_smp_load_acquire(ring->cq_head);
    unsigned tail = pw2_smp_load_acquire(ring->cq_tail);

    if (head == tail)
        return NULL;

    return &ring->cqes[head & *ring->cq_ring_mask];
}

/* Mark a CQE as consumed (advance CQ head) */
static inline void
pw2_ring_cqe_seen(struct pw2_ring *ring)
{
    unsigned head = *ring->cq_head;
    pw2_smp_store_release(ring->cq_head, head + 1);
}

#endif /* PW2_URING_H */
