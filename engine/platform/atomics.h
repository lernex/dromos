// The few atomic accesses the engine makes to words it shares with a GPU or a kernel (mapped host memory, or a
// counter the compiler must not drop), spelled once for GCC and Clang (__atomic builtins) and for MSVC. On x64 MSVC
// a volatile access already has acquire (load) and release (store) semantics, its default /volatile:ms.
#pragma once

#include <stdint.h>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

inline uint32_t lx_load_acquire(const volatile uint32_t* p) {
#if defined(_MSC_VER) && !defined(__clang__)
  return *p;
#else
  return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}

inline void lx_store_release(volatile uint32_t* p, uint32_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
  *p = v;
#else
  __atomic_store_n(p, v, __ATOMIC_RELEASE);
#endif
}

inline void lx_fetch_add_relaxed(unsigned long long* p, unsigned long long v) {
#if defined(_MSC_VER) && !defined(__clang__)
  _InterlockedExchangeAdd64(reinterpret_cast<volatile long long*>(p), static_cast<long long>(v));
#else
  __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
#endif
}
