// lernex: the command line of the Lernex engine.
#include <cstdio>
#include <cstring>

int cmd_inspect(int argc, char** argv);
int cmd_eval(int argc, char** argv);
int cmd_spec(int argc, char** argv);
int cmd_spec_check(int argc, char** argv);
int cmd_spec_cost(int argc, char** argv);
int cmd_features(int argc, char** argv);
int cmd_block_check(int argc, char** argv);

int main(int argc, char** argv) {
  if (argc >= 2) {
    if (!std::strcmp(argv[1], "inspect")) return cmd_inspect(argc - 1, argv + 1);
    if (!std::strcmp(argv[1], "eval")) return cmd_eval(argc - 1, argv + 1);
    if (!std::strcmp(argv[1], "spec")) return cmd_spec(argc - 1, argv + 1);
    if (!std::strcmp(argv[1], "spec-check")) return cmd_spec_check(argc - 1, argv + 1);
    if (!std::strcmp(argv[1], "spec-cost")) return cmd_spec_cost(argc - 1, argv + 1);
    if (!std::strcmp(argv[1], "features")) return cmd_features(argc - 1, argv + 1);
    if (!std::strcmp(argv[1], "block-check")) return cmd_block_check(argc - 1, argv + 1);
  }
  std::fprintf(stderr,
               "usage: lernex <command>\n"
               "  inspect MODEL_DIR                         load a model and describe it\n"
               "  eval MODEL_DIR PROMPTS_JSON OUT_DIR [N]   logits for reference prompts; with N, the\n"
               "                                            first N tokens at once and the rest one by one\n"
               "  spec MODEL_DIR PROMPTS_JSON [options]     speculative decoding against plain decoding\n"
               "  spec-check MODEL_DIR PROMPTS_JSON [N]     verify steps bit-identical to one-token steps\n"
               "  spec-cost MODEL_DIR [CONTEXT]             milliseconds per plain and per verify step\n"
               "  features MODEL_DIR DATA_JSONL OUT_DIR     hidden-state shards for training draft models\n"
               "  block-check MODEL_DIR HEAD PROMPTS OUT    a block drafter's proposal, for check_block.py\n");
  return 2;
}
