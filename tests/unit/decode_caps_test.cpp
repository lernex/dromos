// The decode program's size decisions (engine/cuda/decode_caps.h) at the x buffer the layout gives. The
// builder's chunk counts, expert tensor-core choices and depth-memory widths are checked at the size they were
// made at, on the shapes Praxis and Logos run, so a change to the shared-memory layout that moves the x buffer
// shows here before it changes any program's bits.
#include <cstdint>
#include <initializer_list>

#include "cuda/decode_caps.h"
#include "doctest.h"

namespace caps = lx::decode::caps;

namespace {
constexpr int64_t kCap = 75392;  // the x buffer the layout gives (decode_ops.h asserts kXOff)
}  // namespace

TEST_CASE("decode x buffer: the layout gives the size the decisions were made at") {
  CHECK(caps::x_cap_of(caps::kSmemOptIn, caps::kXOff) == kCap);
}

TEST_CASE("decode x buffer: Praxis's and Logos's depth-memory entries split as recorded") {
  // Praxis: hidden 2048, route features 256, so the entry's keys are R + 4 = 260 floats wide, 8 tokens.
  CHECK(caps::entry_kc(2048, 260, 8, kCap) == 2048);
  // Logos: hidden 2560, the same keys: its entry splits in half.
  CHECK(caps::entry_kc(2560, 260, 8, kCap) == 1280);
  // The reason the x buffer keeps its size: a smaller one would split Praxis's entry at 1024.
  CHECK(caps::entry_kc(2048, 260, 8, 73344) == 1024);
}

TEST_CASE("decode x buffer: the decisions the sizes give are monotone in the size") {
  // A larger x buffer never splits a product more, nor refuses a shape the smaller one takes.
  int bad = 0;
  for (int K = 8; K <= 16384; K += 8)
    for (int xe : {2, 4}) {
      const caps::KSplit small = caps::gemv_k_split(K, 32, xe, 32, false, 8, 73344);
      const caps::KSplit big = caps::gemv_k_split(K, 32, xe, 32, false, 8, kCap);
      if (big.chunks > small.chunks) ++bad;
    }
  for (int D = 4; D <= 16384; D += 4)
    for (int n = 0; n <= 8; ++n)
      if (caps::memread_fits(n, D, 73344) && !caps::memread_fits(n, D, kCap)) ++bad;
  CHECK(bad == 0);
}
