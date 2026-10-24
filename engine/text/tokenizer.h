// Text in and out of the model: the Metis tokenizer, the chat format, and the parser that splits
// generated text into reasoning, content and tool calls. Matches Transformers exactly (the
// tokenizer.json BPE, chat_template.jinja, the metis tool/reasoning parsers in vllm-metis).
#pragma once

#include <stdint.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace lx::text {

class Tokenizer {
 public:
  // Reads tokenizer.json, tokenizer_config.json and generation_config.json from a model folder.
  static std::unique_ptr<Tokenizer> load(const std::string& model_dir);
  virtual ~Tokenizer() = default;

  // With parse_special, added/special tokens written in the text ("<|im_start|>") become their ids.
  virtual std::vector<int32_t> encode(std::string_view text, bool parse_special = true) const = 0;
  virtual std::string decode(const std::vector<int32_t>& ids, bool skip_special = false) const = 0;

  // Incremental decoding for streaming: push() returns only text that is complete UTF-8 and will
  // not change when later tokens arrive; flush() returns whatever is left.
  class Stream {
   public:
    virtual ~Stream() = default;
    virtual std::string push(int32_t id) = 0;
    virtual std::string flush() = 0;
  };
  virtual std::unique_ptr<Stream> stream(bool skip_special = false) const = 0;

  virtual int vocab_size() const = 0;
  virtual int32_t token_id(std::string_view token) const = 0;  // -1 when absent
  virtual bool is_special(int32_t id) const = 0;
  virtual std::vector<int32_t> stop_ids() const = 0;  // end-of-turn / eos from the configs
};

struct ToolCall {
  std::string id;
  std::string name;
  std::string arguments_json;  // a JSON object, as text
};

struct Message {
  std::string role;  // system | user | assistant | tool
  std::string content;
  std::string reasoning;              // assistant thinking, when replaying a turn
  std::vector<ToolCall> tool_calls;   // assistant
  std::string tool_call_id, name;     // tool results
};

struct ChatOptions {
  bool thinking = true;           // the template's thinking switch
  std::string tools_json;         // OpenAI-style tools array as JSON text, or empty
  bool add_generation_prompt = true;
};

// The prompt text chat_template.jinja renders for these messages (byte-identical).
std::string render_chat(const std::vector<Message>& messages, const ChatOptions& options);

// Splits streamed model output into reasoning, visible content and tool calls, incrementally.
class OutputParser {
 public:
  struct Delta {
    std::string reasoning, content;
    std::vector<ToolCall> tool_calls;  // complete calls only
  };
  explicit OutputParser(bool thinking);
  ~OutputParser();
  Delta push(std::string_view text);
  Delta finish();

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace lx::text
