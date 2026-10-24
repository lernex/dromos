// lernex: the command line of the Lernex engine.
#include <cstdio>
#include <cstring>

int cmd_inspect(int argc, char** argv);
int cmd_eval(int argc, char** argv);
int cmd_doctor(int argc, char** argv);
int cmd_tune(int argc, char** argv);

int main(int argc, char** argv) {
  if (argc >= 2) {
    if (!std::strcmp(argv[1], "inspect")) return cmd_inspect(argc - 1, argv + 1);
    if (!std::strcmp(argv[1], "eval")) return cmd_eval(argc - 1, argv + 1);
    if (!std::strcmp(argv[1], "doctor")) return cmd_doctor(argc - 1, argv + 1);
    if (!std::strcmp(argv[1], "tune")) return cmd_tune(argc - 1, argv + 1);
  }
  std::fprintf(stderr,
               "usage: lernex <command>\n"
               "  doctor [MODEL_DIR] [--weights F] [--context N] [--sequences N]\n"
               "                                            the GPU, what it can run, and the plan for a model\n"
               "  tune MODEL_DIR [--out PATH] [--log]       time the kernel variants on this GPU, keep the winners\n"
               "  inspect MODEL_DIR                         load a model and describe it\n"
               "  eval MODEL_DIR PROMPTS_JSON OUT_DIR [N]   logits for reference prompts; with N, the\n"
               "                                            first N tokens at once and the rest one by one\n");
  return 2;
}
