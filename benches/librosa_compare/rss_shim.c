/*
 * Exact peak-RSS tracking for rss_probe.py, loaded with LD_PRELOAD.
 *
 * The kernel's VmHWM is updated from per-CPU approximations of the RSS
 * counters, so after a large free it can be off by a few hundred KB, and a
 * call that touches no new page can still show 8-20 KB of "growth". VmRSS
 * and /proc/self/statm are exact sums. RSS only goes down when memory is
 * returned to the kernel (munmap, madvise, mremap, or a heap trim inside
 * free or realloc), so between those points it never decreases. Reading
 * the exact RSS just before each of them, and once more at the end, gives
 * the exact peak.
 *
 * While tracking is on (between rss_shim_begin() and rss_shim_end()), every
 * free, realloc, munmap, madvise and mremap first reads /proc/self/statm and
 * keeps the largest value. Outside that window the wrappers only test one
 * flag. The library's allocations are untouched: free and realloc forward to
 * glibc's __libc_free and __libc_realloc.
 *
 * Build: cc -O2 -shared -fPIC -o rss_shim.so rss_shim.c
 */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <sys/syscall.h>
#include <unistd.h>

extern void __libc_free(void *);
extern void *__libc_realloc(void *, size_t);

static atomic_int active;
static atomic_long peak_pages;
static int statm_fd = -1;

/* Resident pages from /proc/self/statm (second field); -1 on failure. */
static long resident_pages(void) {
    char buf[128];
    if (statm_fd < 0)
        return -1;
    ssize_t n = pread(statm_fd, buf, sizeof buf - 1, 0);
    if (n <= 0)
        return -1;
    buf[n] = 0;
    char *p = buf;
    while (*p && *p != ' ')
        p++;
    long v = 0;
    while (*p == ' ')
        p++;
    while (*p >= '0' && *p <= '9')
        v = v * 10 + (*p++ - '0');
    return v;
}

static void sample(void) {
    long v = resident_pages();
    long cur = atomic_load(&peak_pages);
    while (v > cur && !atomic_compare_exchange_weak(&peak_pages, &cur, v)) {
    }
}

#define SAMPLE_IF_ACTIVE()                                                     \
    do {                                                                       \
        if (atomic_load_explicit(&active, memory_order_relaxed))               \
            sample();                                                          \
    } while (0)

/* Starts tracking; returns the resident size in KB at the start. */
long rss_shim_begin(void) {
    if (statm_fd < 0)
        statm_fd = open("/proc/self/statm", O_RDONLY | O_CLOEXEC);
    long v = resident_pages();
    atomic_store(&peak_pages, v);
    atomic_store(&active, 1);
    return v * (sysconf(_SC_PAGESIZE) / 1024);
}

/* Stops tracking; returns the peak resident size in KB since rss_shim_begin. */
long rss_shim_end(void) {
    sample();
    atomic_store(&active, 0);
    return atomic_load(&peak_pages) * (sysconf(_SC_PAGESIZE) / 1024);
}

void free(void *p) {
    if (p)
        SAMPLE_IF_ACTIVE();
    __libc_free(p);
}

void *realloc(void *p, size_t n) {
    if (p)
        SAMPLE_IF_ACTIVE();
    return __libc_realloc(p, n);
}

int munmap(void *addr, size_t len) {
    SAMPLE_IF_ACTIVE();
    return (int)syscall(SYS_munmap, addr, len);
}

int madvise(void *addr, size_t len, int advice) {
    SAMPLE_IF_ACTIVE();
    return (int)syscall(SYS_madvise, addr, len, advice);
}

void *mremap(void *old, size_t old_len, size_t new_len, int flags, ...) {
    void *new_addr = NULL;
    if (flags & 2 /* MREMAP_FIXED */) {
        va_list ap;
        va_start(ap, flags);
        new_addr = va_arg(ap, void *);
        va_end(ap);
    }
    SAMPLE_IF_ACTIVE();
    return (void *)syscall(SYS_mremap, old, old_len, new_len, flags, new_addr);
}
