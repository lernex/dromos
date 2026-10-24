// lernex run MODEL [--think on|off] [--context N]
//
// An interactive chat in the terminal. Replies stream as they are sampled. The model's reasoning
// shows dimmed while it thinks, then folds into one line when the answer starts (/expand reads it
// again). Ctrl-C stops a reply and keeps the chat; Ctrl-D or /quit leaves.
#include <sys/ioctl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cli/common.h"
#include "runtime/engine.h"
#include "text/tokenizer.h"

using namespace lx;

namespace {

std::atomic<bool> g_interrupted{false};

void on_interrupt(int) { g_interrupted.store(true); }

bool stdout_is_terminal() { return isatty(STDOUT_FILENO) != 0; }

int terminal_width() {
  winsize ws{};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) return ws.ws_col;
  return 80;
}

// Counts the rows a dimmed block takes on screen, so it can be erased when it folds.
class Block {
 public:
  explicit Block(int width) : width_(width) {}
  void add(const std::string& text) {
    for (unsigned char c : text) {
      if (c == '\n') {
        rows_ += 1;
        col_ = 0;
      } else if ((c & 0xC0) != 0x80) {  // a UTF-8 lead byte is one column
        if (++col_ > width_) {
          rows_ += 1;
          col_ = 1;
        }
      }
    }
  }
  // Lines to go up from the cursor (which sits on the block's last line) to its first line.
  int lines_up() const { return rows_; }

 private:
  int width_;
  int rows_ = 0;
  int col_ = 0;
};

struct Turn {
  std::string reasoning, content;
  int tokens = 0;
  double seconds = 0.0;
  runtime::FinishReason reason = runtime::FinishReason::Stop;
  std::string error;
};

class Chat {
 public:
  Chat(runtime::Engine& engine, const cli::Model& model, bool thinking)
      : engine_(engine), model_(model), thinking_(thinking), color_(stdout_is_terminal()), width_(terminal_width()) {}

  void run() {
    banner();
    std::string line;
    while (true) {
      g_interrupted.store(false);
      std::fputs(prompt().c_str(), stdout);
      std::fflush(stdout);
      if (!std::getline(std::cin, line)) {
        if (std::cin.eof()) break;
        std::cin.clear();  // Ctrl-C at the prompt clears the line and asks again
        std::fputs("\n", stdout);
        continue;
      }
      std::string text = line;
      if (text == "\"\"\"") text = multiline();
      else text = continued(text);
      if (text.empty()) continue;
      if (text[0] == '/') {
        if (command(text)) break;
        continue;
      }
      reply(text);
    }
    std::fputs("\n", stdout);
  }

 private:
  std::string prompt() const { return color_ ? "\x1b[1m>\x1b[0m " : "> "; }

  void banner() {
    std::printf("%s", color_ ? "\x1b[2m" : "");
    std::printf("%s", model_name().c_str());
    std::printf("  thinking %s", thinking_ ? "on" : "off");
    std::printf("  ·  /help for commands\n");
    std::printf("%s\n", color_ ? "\x1b[0m" : "");
    std::fflush(stdout);
  }

  std::string model_name() const { return model_.name; }

  // A line ending in a backslash continues onto the next.
  std::string continued(std::string text) {
    while (!text.empty() && text.back() == '\\') {
      text.pop_back();
      std::string more;
      std::fputs("... ", stdout);
      std::fflush(stdout);
      if (!std::getline(std::cin, more)) break;
      text += "\n" + more;
    }
    return text;
  }

  std::string multiline() {
    std::string out, line;
    while (std::getline(std::cin, line)) {
      if (line == "\"\"\"") break;
      if (!out.empty()) out += "\n";
      out += line;
    }
    return out;
  }

  // Returns true to leave the chat.
  bool command(const std::string& text) {
    if (text == "/quit" || text == "/exit") return true;
    if (text == "/clear") {
      history_.clear();
      std::printf("%s", color_ ? "\x1b[2mStarted a new conversation.\x1b[0m\n" : "Started a new conversation.\n");
      return false;
    }
    if (text == "/think on" || text == "/think off") {
      thinking_ = text == "/think on";
      std::printf("thinking %s\n", thinking_ ? "on" : "off");
      return false;
    }
    if (text == "/expand") {
      if (last_.reasoning.empty()) std::printf("No reasoning to show.\n");
      else std::printf("%s%s%s\n", color_ ? "\x1b[2m" : "", last_.reasoning.c_str(), color_ ? "\x1b[0m" : "");
      return false;
    }
    if (text == "/help") {
      std::printf(
          "  /think on|off   show or hide the model's reasoning\n"
          "  /expand         read the last reply's reasoning again\n"
          "  /clear          start a new conversation\n"
          "  /quit           leave (Ctrl-D also works)\n"
          "  end a line with \\ to continue it, or type \"\"\" to write several lines\n");
      return false;
    }
    std::printf("Unknown command %s. Type /help.\n", text.c_str());
    return false;
  }

  void reply(const std::string& text) {
    history_.push_back(lx::text::Message{"user", text, "", {}, "", ""});
    lx::text::ChatOptions options;
    options.model_dir = model_.dir.string();
    options.thinking = thinking_;
    const std::string rendered = lx::text::render_chat(history_, options);

    runtime::GenerateRequest req;
    req.prompt = engine_.tokenizer()->encode(rendered, true);
    req.max_tokens = 1 << 20;  // as much as the context allows
    req.thinking = thinking_;
    req.sampling = cli::recommended_sampling(model_);
    req.sampling.seed = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    Turn turn = run_request(std::move(req));
    last_ = turn;

    if (turn.reason == runtime::FinishReason::Error) {
      std::printf("\n%s\n", turn.error.c_str());
      history_.pop_back();
      return;
    }
    history_.push_back(lx::text::Message{"assistant", turn.content, turn.reasoning, {}, "", ""});
    if (turn.reason == runtime::FinishReason::Cancelled) std::printf("%s", color_ ? " \x1b[2m[stopped]\x1b[0m" : " [stopped]");
    std::printf("\n");
    if (turn.tokens > 0 && turn.seconds > 0.0) {
      std::printf("%s", color_ ? "\x1b[2m" : "");
      std::printf("%d tokens, %.0f tok/s", turn.tokens, turn.tokens / turn.seconds);
      std::printf("%s\n", color_ ? "\x1b[0m" : "");
    }
    std::printf("\n");
    std::fflush(stdout);
  }

  Turn run_request(runtime::GenerateRequest req) {
    struct Shared {
      std::mutex mu;
      std::condition_variable cv;
      Turn turn;
      bool done = false;
      bool in_reasoning = false;
      bool folded = false;
      Block block{80};
      std::chrono::steady_clock::time_point reasoning_started{};
      double reasoning_seconds = 0.0;
    };
    auto shared = std::make_shared<Shared>();
    shared->block = Block(width_);
    const bool color = color_;

    runtime::RequestCallbacks cb;
    cb.on_token = [shared, color](const runtime::TokenDelta& d) {
      std::lock_guard<std::mutex> lock(shared->mu);
      if (d.token >= 0) shared->turn.tokens += 1;
      if (!d.reasoning.empty()) {
        if (!shared->in_reasoning) shared->reasoning_started = std::chrono::steady_clock::now();
        shared->in_reasoning = true;
        shared->turn.reasoning += d.reasoning;
        shared->block.add(d.reasoning);
        std::printf("%s%s%s", color ? "\x1b[2m" : "", d.reasoning.c_str(), color ? "\x1b[0m" : "");
        std::fflush(stdout);
      }
      if (!d.content.empty()) {
        if (shared->in_reasoning && !shared->folded) {
          shared->reasoning_seconds =
              std::chrono::duration<double>(std::chrono::steady_clock::now() - shared->reasoning_started).count();
          fold(shared->block, shared->reasoning_seconds, color);
          shared->folded = true;
        }
        shared->turn.content += d.content;
        std::printf("%s", d.content.c_str());
        std::fflush(stdout);
      }
    };
    cb.on_finish = [shared](const runtime::RequestResult& r) {
      std::lock_guard<std::mutex> lock(shared->mu);
      shared->turn.reason = r.reason;
      shared->turn.error = r.error;
      shared->turn.seconds = r.seconds - r.time_to_first_token;
      shared->done = true;
      shared->cv.notify_all();
    };

    g_interrupted.store(false);
    const uint64_t id = engine_.submit(std::move(req), std::move(cb));
    bool cancel_sent = false;
    std::unique_lock<std::mutex> lock(shared->mu);
    while (!shared->done) {
      shared->cv.wait_for(lock, std::chrono::milliseconds(50));
      if (!cancel_sent && g_interrupted.load()) {
        engine_.cancel(id);
        cancel_sent = true;
      }
    }
    return shared->turn;
  }

  // Erases the reasoning as it was printed and leaves one line in its place. It is kept for /expand.
  static void fold(const Block& block, double seconds, bool color) {
    std::printf("\r");
    if (block.lines_up() > 0) std::printf("\x1b[%dA", block.lines_up());
    std::printf("\x1b[0J");
    std::printf("%s", color ? "\x1b[2m" : "");
    std::printf("Thought for %.1f s. /expand to read it.", seconds);
    std::printf("%s\n", color ? "\x1b[0m" : "");
    std::fflush(stdout);
  }

  runtime::Engine& engine_;
  cli::Model model_;
  bool thinking_;
  bool color_;
  int width_;
  std::vector<lx::text::Message> history_;
  Turn last_;
};

}  // namespace

int cmd_run(int argc, char** argv) {
  using namespace lx;
  std::string name;
  int context = 0;
  bool thinking = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--think") {
      const std::string v = value();
      if (v != "on" && v != "off") throw std::runtime_error("--think takes on or off");
      thinking = v == "on";
    } else if (a == "--context") context = std::stoi(value());
    else if (name.empty() && a[0] != '-') name = a;
    else throw std::runtime_error("unknown option " + a);
  }
  if (name.empty()) {
    std::fprintf(stderr, "usage: lernex run MODEL [--think on|off] [--context N]\n");
    return 2;
  }

  const cli::Model model = cli::find_model(name, true);
  const runtime::Plan plan = cli::plan_for(model, 1, context);
  for (const std::string& note : plan.notes) std::fprintf(stderr, "%s\n", note.c_str());
  runtime::Engine engine(model.dir.string(), plan);
  if (!engine.tokenizer()) throw std::runtime_error("the model folder has no tokenizer");

  std::signal(SIGINT, on_interrupt);
  std::signal(SIGPIPE, SIG_IGN);
  Chat chat(engine, model, thinking);
  chat.run();
  return 0;
}
