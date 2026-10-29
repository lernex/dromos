// cpuid and xgetbv on x86-64 under the names GCC's <cpuid.h> gives them (__get_cpuid_max, __get_cpuid,
// __cpuid_count), for MSVC too, plus lx_xgetbv0(): XCR0, the register states the operating system saves.
#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>

inline unsigned __get_cpuid_max(unsigned ext, unsigned* sig) {
  int r[4];
  __cpuid(r, static_cast<int>(ext));
  if (sig) *sig = static_cast<unsigned>(r[1]);
  return static_cast<unsigned>(r[0]);
}

inline int __get_cpuid(unsigned leaf, unsigned* a, unsigned* b, unsigned* c, unsigned* d) {
  if (__get_cpuid_max(leaf & 0x80000000u, nullptr) < leaf) return 0;
  int r[4];
  __cpuid(r, static_cast<int>(leaf));
  *a = static_cast<unsigned>(r[0]);
  *b = static_cast<unsigned>(r[1]);
  *c = static_cast<unsigned>(r[2]);
  *d = static_cast<unsigned>(r[3]);
  return 1;
}

#define __cpuid_count(leaf, sub, a, b, c, d)                           \
  do {                                                                  \
    int lx_regs_[4];                                                    \
    __cpuidex(lx_regs_, static_cast<int>(leaf), static_cast<int>(sub)); \
    (a) = static_cast<unsigned>(lx_regs_[0]);                           \
    (b) = static_cast<unsigned>(lx_regs_[1]);                           \
    (c) = static_cast<unsigned>(lx_regs_[2]);                           \
    (d) = static_cast<unsigned>(lx_regs_[3]);                           \
  } while (0)

inline unsigned long long lx_xgetbv0() { return _xgetbv(0); }

#else
#include <cpuid.h>

inline unsigned long long lx_xgetbv0() {
  unsigned lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<unsigned long long>(hi) << 32) | lo;
}
#endif

#endif
