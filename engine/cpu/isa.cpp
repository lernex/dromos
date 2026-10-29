// Which instruction set the CPU backend dispatches on (tunable.h's lx_arch, on the CPU).
//
// Levels, each a kernel table built from kern/isa.cpp with its own flags:
//   1 generic      plain C++ (whatever the compiler makes of it for the architecture's baseline)
//   10 avx2        AVX2 + FMA + F16C            11 avxvnni     + AVX-VNNI (int8 dot products)
//   20 avx512      AVX-512 F/BW/DQ/VL           21 avx512vnni  + AVX512-VNNI
//   22 avx512bf16  + AVX512-BF16 (bf16 dot products)
//   50 neon        Armv8 Advanced SIMD          51 neondot     + the dot product (SDOT)
//   52 neoni8mm    + I8MM and BF16 (BFDOT)
// The real level is the highest one this CPU supports and the library was built with. LX_ARCH_OVERRIDE
// or lx_set_arch_override can lower it to any other level the CPU supports (1 is always allowed), so
// every path can be run and tested on one machine; raising it, or naming a level the CPU lacks, is
// refused.
#include <stdlib.h>

#include <mutex>

#include "cpu/cpu.h"
#include "cpu/dromos_cpu.h"

#if defined(__x86_64__) || defined(_M_X64)
#include "cpu/cpuid.h"
#elif defined(__aarch64__)
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#endif
#endif

namespace lx::cpu {
namespace {

struct Level {
  int code;
  const Kern* (*table)();
};

// Built levels, lowest first. A level missing from the build has no entry.
const Level kLevels[] = {
    {1, kern_generic},
#if defined(LX_CPU_HAVE_AVX2)
    {10, kern_avx2},
#endif
#if defined(LX_CPU_HAVE_AVXVNNI)
    {11, kern_avxvnni},
#endif
#if defined(LX_CPU_HAVE_AVX512)
    {20, kern_avx512},
#endif
#if defined(LX_CPU_HAVE_AVX512VNNI)
    {21, kern_avx512vnni},
#endif
#if defined(LX_CPU_HAVE_AVX512BF16)
    {22, kern_avx512bf16},
#endif
#if defined(LX_CPU_HAVE_NEON)
    {50, kern_neon},
#endif
#if defined(LX_CPU_HAVE_NEONDOT)
    {51, kern_neondot},
#endif
#if defined(LX_CPU_HAVE_NEONI8MM)
    {52, kern_neoni8mm},
#endif
};

#if defined(__x86_64__) || defined(_M_X64)
struct X86 {
  bool avx2 = false, avxvnni = false, avx512 = false, avx512vnni = false, avx512bf16 = false;
};

X86 x86_features() {
  X86 f;
  unsigned a, b, c, d;
  if (!__get_cpuid(1, &a, &b, &c, &d)) return f;
  const bool fma = c & (1u << 12), osxsave = c & (1u << 27), avx = c & (1u << 28), f16c = c & (1u << 29);
  if (!osxsave || !avx) return f;
  const unsigned lo = static_cast<unsigned>(lx_xgetbv0());
  const bool ymm_state = (lo & 0x6u) == 0x6u;
  const bool zmm_state = (lo & 0xe6u) == 0xe6u;
  if (!ymm_state) return f;
  unsigned max_leaf = __get_cpuid_max(0, nullptr);
  if (max_leaf < 7) return f;
  __cpuid_count(7, 0, a, b, c, d);
  const bool avx2 = b & (1u << 5), bmi2 = b & (1u << 8);
  const bool f512 = b & (1u << 16), dq = b & (1u << 17), bw = b & (1u << 30), vl = b & (1u << 31);
  const bool vnni512 = c & (1u << 11);
  unsigned a1 = 0, b1, c1, d1;
  __cpuid_count(7, 1, a1, b1, c1, d1);
  const bool avxvnni = a1 & (1u << 4), bf16_512 = a1 & (1u << 5);
  f.avx2 = avx2 && fma && f16c && bmi2;
  f.avxvnni = f.avx2 && avxvnni;
  f.avx512 = f.avx2 && zmm_state && f512 && dq && bw && vl;
  f.avx512vnni = f.avx512 && vnni512;
  f.avx512bf16 = f.avx512vnni && bf16_512;
  return f;
}
#endif

#if defined(__aarch64__)
struct Arm {
  bool dot = false, i8mm_bf16 = false;
};

#if defined(__APPLE__)
bool sysctl_flag(const char* name) {
  int v = 0;
  size_t len = sizeof(v);
  return sysctlbyname(name, &v, &len, nullptr, 0) == 0 && v != 0;
}
#endif

Arm arm_features() {
  Arm f;
#if defined(__APPLE__)
  f.dot = sysctl_flag("hw.optional.arm.FEAT_DotProd");
  f.i8mm_bf16 = f.dot && sysctl_flag("hw.optional.arm.FEAT_I8MM") && sysctl_flag("hw.optional.arm.FEAT_BF16");
#elif defined(__linux__)
  const unsigned long hw = getauxval(AT_HWCAP), hw2 = getauxval(AT_HWCAP2);
  f.dot = hw & (1ul << 20);  // HWCAP_ASIMDDP
  f.i8mm_bf16 = f.dot && (hw2 & (1ul << 13)) && (hw2 & (1ul << 14));  // HWCAP2_I8MM, HWCAP2_BF16
#endif
  return f;
}
#endif

// Whether this CPU can run a level.
bool supported(int code) {
  if (code == 1) return true;
#if defined(__x86_64__) || defined(_M_X64)
  static const X86 f = x86_features();
  switch (code) {
    case 10: return f.avx2;
    case 11: return f.avxvnni;
    case 20: return f.avx512;
    case 21: return f.avx512vnni;
    case 22: return f.avx512bf16;
    default: return false;
  }
#elif defined(__aarch64__)
  static const Arm f = arm_features();
  switch (code) {
    case 50: return true;
    case 51: return f.dot;
    case 52: return f.i8mm_bf16;
    default: return false;
  }
#else
  return false;
#endif
}

const Level* find_level(int code) {
  for (const Level& l : kLevels)
    if (l.code == code && supported(code) && l.table()) return &l;
  return nullptr;
}

std::mutex g_mu;
std::atomic<const Kern*> g_kern{nullptr};
int g_override = -1;  // -1: not read from the environment yet; 0: none

// The level in effect for an override (0 for none): the override when the CPU supports it and it is
// below the real level, else the real level.
int effective(int over) {
  const int real = real_level();
  if (over > 0 && over < real && find_level(over)) return over;
  return real;
}

const Kern* select() {
  std::lock_guard<std::mutex> g(g_mu);
  if (g_override < 0) {
    const char* env = getenv("LX_ARCH_OVERRIDE");
    g_override = env ? atoi(env) : 0;
  }
  const Kern* k = find_level(effective(g_override))->table();
#if defined(__APPLE__) && defined(__aarch64__)
  // Apple's cores run BFDOT and BFMMLA slower than converting to f32 and using FMA (measured on an M5,
  // one thread, a 512 x 2048 x 2048 BF16 product: 67 and 62 GFLOP/s against 103), so on a Mac the bf16
  // products take the f32 path. Other Arm cores keep the bf16 instructions. LX_CPU_BF16_DOT=1 keeps them
  // here too (to test them).
  const char* keep = getenv("LX_CPU_BF16_DOT");
  if (k->dot_bf16 && !(keep && *keep == '1')) {
    static Kern apple;
    apple = *k;
    apple.dot_bf16 = nullptr;
    k = &apple;
  }
#endif
  g_kern.store(k, std::memory_order_release);
  return k;
}

}  // namespace

int real_level() {
  int best = 1;
  for (const Level& l : kLevels)
    if (supported(l.code) && l.table() && l.code > best) best = l.code;
  return best;
}

const Kern& kern() {
  const Kern* k = g_kern.load(std::memory_order_acquire);
  return k ? *k : *select();
}

const char* level_isa(int code) {
  switch (code) {
    case 10: return "AVX2, FMA, F16C";
    case 11: return "AVX2, FMA, F16C, AVX-VNNI";
    case 20: return "AVX-512 (F, BW, DQ, VL)";
    case 21: return "AVX-512 (F, BW, DQ, VL), AVX512-VNNI";
    case 22: return "AVX-512 (F, BW, DQ, VL), AVX512-VNNI, AVX512-BF16";
    case 50: return "Advanced SIMD";
    case 51: return "Advanced SIMD, dot product";
    case 52: return "Advanced SIMD, dot product, I8MM, BF16";
    default: return "portable C++";
  }
}

int native_math() {
  const Kern& k = kern();
  int m = 0;
  // VNNI and SDOT multiply int8 pairs into int32 sums; below them int8 goes through 16-bit multiplies.
  if (k.code == 11 || k.code == 21 || k.code == 22 || k.code == 51 || k.code == 52) m |= 1 << LX_MATH_INT8;
  // A bf16 dot instruction, where the level has one and uses it (the Mac takes the f32 path: select()).
  if (k.dot_bf16) m |= 1 << LX_MATH_BF16;
  return m;
}

bool amx_present() {
#if defined(__x86_64__) || defined(_M_X64)
  if (__get_cpuid_max(0, nullptr) < 7) return false;
  unsigned a, b, c, d;
  __cpuid_count(7, 0, a, b, c, d);
  const bool tile = d & (1u << 24), bf16 = d & (1u << 22), int8 = d & (1u << 25);
  if (!tile || !(bf16 || int8)) return false;
  __get_cpuid(1, &a, &b, &c, &d);
  if (!(c & (1u << 27))) return false;  // OSXSAVE
  const unsigned lo = static_cast<unsigned>(lx_xgetbv0());
  return (lo & (3u << 17)) == (3u << 17);  // the OS saves the tile state
#else
  return false;
#endif
}

}  // namespace lx::cpu

extern "C" {

int lx_arch(void) { return lx::cpu::kern().code; }

int lx_set_arch_override(int arch) {
  {
    std::lock_guard<std::mutex> g(lx::cpu::g_mu);
    lx::cpu::g_override = arch > 0 ? arch : 0;
    lx::cpu::g_kern.store(nullptr, std::memory_order_release);
  }
  return lx_arch();
}

void lx_cpu_force_scalar(int on) { lx_set_arch_override(on ? 1 : 0); }

}  // extern "C"
