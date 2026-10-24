// lernex on the Metal library: the commands of cli/main.cpp that the end-to-end check needs.
#include <cstdio>
#include <cstring>

int cmd_eval(int argc, char** argv);

int main(int argc, char** argv) {
  if (argc >= 2 && !std::strcmp(argv[1], "eval")) return cmd_eval(argc - 1, argv + 1);
  std::fprintf(stderr,
               "usage: lernex <command>\n"
               "  eval MODEL_DIR PROMPTS_JSON OUT_DIR [N]   logits for reference prompts; with N, the\n"
               "                                            first N tokens at once and the rest one by one\n");
  return 2;
}
