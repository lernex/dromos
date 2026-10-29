// The POSIX calls the engine makes, under their POSIX names on every platform. On Linux and macOS this is the
// system's own headers. On Windows it declares the few the C runtime lacks (dlopen, mmap, pread and their kin),
// implemented in posix_win.cpp over the Win32 API, so a call site reads the same everywhere. Anything that differs
// in meaning between the systems (paths, processes, the terminal) is in platform.h instead.
#pragma once

#if !defined(_WIN32)

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

// Aligned heap memory, returned with lx_aligned_free (Windows keeps aligned blocks apart from malloc's, so free() must
// not see them). Any size, any power-of-two alignment from sizeof(void*) up.
inline void* lx_aligned_alloc(size_t alignment, size_t size) {
  void* p = nullptr;
  return posix_memalign(&p, alignment, size) == 0 ? p : nullptr;
}
inline void lx_aligned_free(void* p) { free(p); }

#else

#include <direct.h>
#include <malloc.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

// The same type cpp-httplib declares, so the two can meet in one file.
using ssize_t = __int64;

#ifndef STDIN_FILENO
#define STDIN_FILENO 0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
#endif
#ifndef F_OK
#define F_OK 0
#endif

// Shared libraries: LoadLibraryExW. A path with a directory loads with that directory first for the library's own
// dependencies (as an $ORIGIN rpath does), then PATH; a bare name searches the system's default order.
#define RTLD_LAZY 0x1
#define RTLD_NOW 0x2
#define RTLD_LOCAL 0x0
#define RTLD_GLOBAL 0x100
void* dlopen(const char* file, int mode);
void* dlsym(void* handle, const char* name);
int dlclose(void* handle);
// The last failure on this thread, then null until the next one (as POSIX says).
char* dlerror(void);

// Memory maps: anonymous read-write memory (VirtualAlloc), and a read-only view of a whole file (a file mapping).
#define PROT_NONE 0x0
#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS
#define MAP_FAILED ((void*)-1)
#define MADV_NORMAL 0
#define MADV_RANDOM 1
#define MADV_SEQUENTIAL 2
#define MADV_WILLNEED 3
#define MADV_DONTNEED 4
#define MADV_HUGEPAGE 14
void* mmap(void* addr, size_t length, int prot, int flags, int fd, int64_t offset);
int munmap(void* addr, size_t length);
// MADV_WILLNEED prefetches (PrefetchVirtualMemory); every other advice is accepted and changes nothing.
int madvise(void* addr, size_t length, int advice);

// A positioned read that leaves no state behind for other threads reading the same descriptor.
ssize_t pread(int fd, void* buf, size_t count, int64_t offset);
// A read-only descriptor that bypasses the system's file cache (FILE_FLAG_NO_BUFFERING): reads must start and end on
// sector boundaries into sector-aligned memory, as O_DIRECT's do. -1 on failure.
int lx_open_unbuffered(const char* path);

int usleep(unsigned int microseconds);
int nanosleep(const struct timespec* request, struct timespec* remaining);

#define _SC_PAGESIZE 30
#define _SC_NPROCESSORS_ONLN 84
long sysconf(int name);

// The absolute form of a path (links are not resolved). Allocated with malloc when resolved is null.
char* realpath(const char* path, char* resolved);

inline void* lx_aligned_alloc(size_t alignment, size_t size) { return _aligned_malloc(size, alignment); }
inline void lx_aligned_free(void* p) { _aligned_free(p); }

inline struct tm* localtime_r(const time_t* t, struct tm* out) { return localtime_s(out, t) == 0 ? out : nullptr; }
inline struct tm* gmtime_r(const time_t* t, struct tm* out) { return gmtime_s(out, t) == 0 ? out : nullptr; }

#endif
