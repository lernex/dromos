// dromos: the command line of Dromos.
#include <cstdio>
#include <cstring>
#include <exception>

#include "lx_version.h"
#include "platform/platform.h"

int cmd_inspect(int argc, char** argv);
int cmd_eval(int argc, char** argv);
int cmd_doctor(int argc, char** argv);
int cmd_tune(int argc, char** argv);
int cmd_bench(int argc, char** argv);
int cmd_run(int argc, char** argv);
int cmd_serve(int argc, char** argv);
int cmd_pull(int argc, char** argv);
int cmd_list(int argc, char** argv);
int cmd_offload(int argc, char** argv);
int cmd_spec(int argc, char** argv);
int cmd_spec_check(int argc, char** argv);
int cmd_spec_exactness(int argc, char** argv);
int cmd_spec_cost(int argc, char** argv);
int cmd_pass1(int argc, char** argv);
int cmd_features(int argc, char** argv);
int cmd_block_check(int argc, char** argv);
int cmd_weights(int argc, char** argv);
int cmd_prefill(int argc, char** argv);
int cmd_update(int argc, char** argv);

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: dromos <command>\n"
               "  run MODEL [--think on|off] [--context N]  chat in the terminal\n"
               "  serve MODEL [--host H] [--port P] [--context N] [--sequences N]\n"
               "                                            the OpenAI API on this machine\n"
               "  pull MODEL                                download a model into the cache\n"
               "  list                                      the models and which are installed\n"
               "  update [--check|--rollback]               the latest release of this install, verified and switched to;\n"
               "                                            --check only reports, --rollback returns to the last version\n"
               "  version                                   the version, the release flavor and the commit\n"
               "  bench MODEL [--streams 1,8,32,64] [--prompt-tokens N] [--gen-tokens N]\n"
               "                                            throughput of the engine on this GPU\n"
               "  doctor [MODEL] [--context N] [--sequences N]\n"
               "                                            the GPU, what it can run, and the plan for a model\n"
               "  tune MODEL_DIR [--out PATH] [--log]       time the kernel variants on this GPU, keep the winners\n"
               "  inspect MODEL_DIR                         load a model and describe it\n"
               "  eval MODEL_DIR PROMPTS_JSON OUT_DIR [N]   logits for reference prompts; with N, the\n"
               "                                            first N tokens at once and the rest one by one\n"
               "  offload probe|eval|bench ...              run larger than the GPU (dromos offload for help)\n"
               "  spec MODEL_DIR PROMPTS_JSON [options]     speculative decoding against plain decoding\n"
               "  spec-check MODEL_DIR PROMPTS_JSON [N]     verify steps bit-identical to one-token steps\n"
               "  spec-exactness MODEL_DIR PROMPTS_JSON PROMPT [DRAWS]  next-token draws of verify vs plain sampling\n"
               "  spec-cost MODEL_DIR [CONTEXT]             milliseconds per plain and per verify step\n"
               "  pass1 MODEL_DIR PROMPTS_JSON [options]    free drafters from the first pass: acceptance, runs, timing\n"
               "  features MODEL_DIR DATA_JSONL OUT_DIR     hidden-state shards for training draft models\n"
               "  block-check MODEL_DIR HEAD PROMPTS OUT    a block drafter's proposal, for check_block.py\n"
               "  weights MODEL_DIR bf16|int8|fp8|nvfp4|bf16x [--reference DIR] [--bytes-against DIR]\n"
               "                                            load a folder in a weight format and check it\n"
               "  prefill MODEL_DIR PROMPTS_JSON T [RUNS]   time one T-token prefill\n"
               "run, serve, bench, doctor, eval, prefill and the spec commands take --weights bf16|int8|fp8|nvfp4|bf16x\n"
               "(default: the format the file stores), --no-bf16x (decode reads the plain BF16 tensors,\n"
               "not their BF16X copies) and --batch-invariant (a request's tokens do not depend on the others\n"
               "sharing its steps, at some cost to batched speed; eval always runs this way).\n");
}

}  // namespace

int main(int argc, char** argv) {
  lx::platform::console_init();  // a Windows console: UTF-8 and VT sequences, put back at exit
  if (argc < 2) {
    usage();
    return 2;
  }
  const char* cmd = argv[1];
  // Answered before anything else: no backend is loaded, and no model is looked for.
  if (!std::strcmp(cmd, "--version") || !std::strcmp(cmd, "version")) {
    std::printf("dromos %s (%s, %s)\n", LX_VERSION_STRING, LX_FLAVOR_STRING, LX_GIT_SHA_STRING);
    return 0;
  }
  try {
    if (!std::strcmp(cmd, "inspect")) return cmd_inspect(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "eval")) return cmd_eval(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "doctor")) return cmd_doctor(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "tune")) return cmd_tune(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "bench")) return cmd_bench(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "run")) return cmd_run(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "serve")) return cmd_serve(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "pull")) return cmd_pull(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "list")) return cmd_list(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "offload")) return cmd_offload(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "spec")) return cmd_spec(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "spec-check")) return cmd_spec_check(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "spec-exactness")) return cmd_spec_exactness(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "spec-cost")) return cmd_spec_cost(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "pass1")) return cmd_pass1(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "features")) return cmd_features(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "block-check")) return cmd_block_check(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "weights")) return cmd_weights(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "prefill")) return cmd_prefill(argc - 1, argv + 1);
    if (!std::strcmp(cmd, "update")) return cmd_update(argc - 1, argv + 1);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dromos: %s\n", e.what());
    return 1;
  }
  usage();
  return 2;
}
