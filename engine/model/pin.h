// The persisting L2 window of the decode (kernels.h, lx_rt_l2_persist).
//
// A decode step reads the same weights on both passes, and some of them on every layer (the depth memory's
// routing and retrieval projections), so a few megabytes kept in L2 save DRAM traffic on every token. The window
// covers one contiguous range, so the chosen bytes are packed into one arena: a weight's decode copy is copied into
// it and the model reads the copy from there, and a Mamba state is allocated in it (it is written every token, so
// it cannot be copied). What is chosen is the bytes that save the most traffic per byte held: reads per token of
// each region, highest first, until the window's grant is full. The grant is what the device allows (open asks
// first, so the arena holds nothing the window cannot keep). A region that straddles the grant keeps its head in the
// window; a state never straddles. The carve-out is process-wide, so it is on only for decode-sized steps (set_on,
// step_pinned): a large batch runs with all of L2, as it does without the window.
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

  // Asks the device what window it grants for the budget, chooses what fits in that grant, allocates the arena and
  // copies the chosen weights into it. Returns false, with nothing changed, when the hold is off, nothing fits, or
  // the backend has no persisting L2 (LX_ERR_NOT_SUPPORTED). Call once, after every weight and state is registered
  // and before any state is allocated. The window is set on return; set_on turns it off between steps.
  bool open(int64_t budget_bytes);

  // Sets the carve-out for the steps that follow: on while a decode-sized step runs, off for the rest. The carve-out
  // is process-wide, so a large batch left with it set would run in what is left of L2.
  void set_on(bool on);

  bool active() const { return window_ > 0; }
  bool on() const { return on_; }
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
  int64_t window_ = 0;  // what the device granted
  bool on_ = false;     // whether the carve-out is set now
};

// The budget the window asks for: LX_L2_PIN_MB (mebibytes), else three quarters of the device's L2. The device grants
// less when its persisting carve-out is smaller (open() reports what it got).
int64_t budget_bytes();

// Whether a step of `tokens` tokens runs with the window. By default only the one-stream decode (one token) is
// pinned: a larger batch or a prefill is not, since the carve-out it would hold costs more than the window saves
// there. LX_L2_PIN_MAX_TOKENS=n raises the cut-off (for measurement: 8 for the 8-stream decode, 32 for the 32-stream).
bool step_pinned(int tokens);

}  // namespace lx::pin
