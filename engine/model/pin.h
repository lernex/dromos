// The persisting L2 window of the decode (kernels.h, lx_rt_l2_persist).
//
// A decode step reads the same weights on both passes, and some of them on every layer (the depth memory's
// routing and retrieval projections), so a few megabytes kept in L2 save DRAM traffic on every token. The window
// covers one contiguous range, so the chosen bytes are packed into one arena: a weight's decode copy is copied into
// it and the model reads the copy from there, and a Mamba state is allocated in it (it is written every token, so
// it cannot be copied). What is chosen is the bytes that save the most traffic per byte held: reads per token of
// each region, highest first, until the budget is full. A region that straddles the budget keeps its head in the
// window; a state never straddles.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace lx::pin {

// What the window may hold: nothing, the weights, the Mamba states, or both.
enum class Hold { Off, Weights, States, Both };

// LX_NO_L2_PIN=1 turns the window off. LX_L2_PIN=weights|states|both picks the hold (default weights).
Hold hold_from_env();

// One device range the decode reads with its siblings (a projection's data, or its scales), and how the model
// is made to read the arena copy instead.
struct Part {
  const void* src = nullptr;
  int64_t bytes = 0;
  std::function<void(const void*)> point;
};

class Arena {
 public:
  Arena();
  // Clears the window and the persisting lines, and frees the arena. Models built after it set their own.
  ~Arena();
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  bool holds_weights() const { return hold_ == Hold::Weights || hold_ == Hold::Both; }
  bool holds_states() const { return hold_ == Hold::States || hold_ == Hold::Both; }

  // A weight's decode bytes, copied into the arena when chosen and then read from there (`point` gives the model
  // the copy's address). `reads` is how many times a token reads each of its bytes.
  void weight(std::vector<Part> parts, double reads);
  // A buffer the model reads and writes every token, zeroed: returns its index (state_buffer gives its address once
  // open() has chosen it, or null when the model should allocate it as before). `reads` counts its traffic per token.
  int state(int64_t bytes, double reads);
  void* state_buffer(int index) const;

  // Chooses what the window holds, allocates the arena and copies the chosen weights into it. Returns false, with
  // nothing changed, when the hold is off, nothing fits, or the backend has no persisting L2 (LX_ERR_NOT_SUPPORTED).
  // Call once, after every weight and state is registered and before any state is allocated.
  bool open(int64_t budget_bytes);

  bool active() const { return window_ > 0; }
  int64_t window_bytes() const { return window_; }
  int64_t arena_bytes() const { return arena_bytes_; }

 private:
  struct Item {
    bool state = false;
    std::vector<Part> parts;  // weights
    int64_t bytes = 0;        // state: the buffer; weight: its parts, each 256-byte aligned
    double reads = 0;
    int64_t offset = 0;       // in the arena, once chosen
    bool chosen = false;
    void* buffer = nullptr;   // state: its address in the arena
  };

  Hold hold_ = Hold::Off;
  std::vector<Item> items_;
  void* base_ = nullptr;
  int64_t arena_bytes_ = 0;
  int64_t window_ = 0;
};

// The budget the window gets: LX_L2_PIN_MB (mebibytes), else three quarters of the device's L2.
int64_t budget_bytes();

}  // namespace lx::pin
