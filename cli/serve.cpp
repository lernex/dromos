// lernex serve MODEL [--host H] [--port P] [--weights bf16|fp8|nvfp4] [--context N] [--sequences N]
//
// Serves MODEL over the OpenAI API on this machine until Ctrl-C.
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "cli/common.h"
#include "core/hub.h"
#include "runtime/engine.h"
#include "server/server.h"

namespace {

lx::server::Server* g_server = nullptr;

void on_interrupt(int) {
  if (g_server) g_server->stop();
}

}  // namespace

int cmd_serve(int argc, char** argv) {
  using namespace lx;
  std::string name, served, host = "127.0.0.1", weights;
  int port = 8080, context = 0, sequences = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--host") host = value();
    else if (a == "--port") port = std::stoi(value());
    else if (a == "--weights") weights = value();
    else if (a == "--name") served = value();
    else if (a == "--context") context = std::stoi(value());
    else if (a == "--sequences") sequences = std::stoi(value());
    else if (name.empty() && a[0] != '-') name = a;
    else throw std::runtime_error("unknown option " + a);
  }
  if (name.empty()) {
    std::fprintf(stderr,
                 "usage: lernex serve MODEL [--host 127.0.0.1] [--port 8080] [--name NAME]\n"
                 "                     [--weights bf16|fp8|nvfp4] [--context N] [--sequences N]\n");
    return 2;
  }
  int format = -1;
  if (!weights.empty()) {
    if (weights == "bf16") format = LX_W_BF16;
    else if (weights == "fp8") format = LX_W_FP8;
    else if (weights == "nvfp4") format = LX_W_NVFP4;
    else throw std::runtime_error("--weights takes bf16, fp8 or nvfp4");
  }

  const cli::Model model = cli::find_model(name, true);
  const runtime::Plan plan = cli::plan_for(model, sequences, context, format);
  runtime::Engine engine(model.dir.string(), plan);

  server::ServerOptions options;
  options.host = host;
  options.port = port;
  options.model_name = !served.empty() ? served : cli::looks_like_path(name) ? std::filesystem::path(name).filename().string() : name;
  options.model_dir = model.dir.string();
  options.threads = std::max(64, plan.max_slots * 2);
  server::Server server(engine, options);
  g_server = &server;
  std::signal(SIGINT, on_interrupt);
  std::signal(SIGTERM, on_interrupt);

  for (const std::string& note : plan.notes) std::fprintf(stderr, "%s\n", note.c_str());
  std::fprintf(stderr, "Serving %s on http://%s:%d/v1  (Ctrl-C to stop)\n", name.c_str(), host.c_str(), port);
  const bool ok = server.run();
  g_server = nullptr;
  std::fprintf(stderr, "Stopped.\n");
  return ok ? 0 : 1;
}
