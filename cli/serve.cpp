// lernex serve MODEL [--host H] [--port P] [--weights bf16|int8|fp8|nvfp4|bf16x] [--no-bf16x] [--context N]
//                    [--sequences N]
//
// Serves MODEL over the OpenAI API on this machine until Ctrl-C. The weights run in the format the file stores
// unless --weights names another (runtime/plan.h); a model larger than the GPU runs with its routed experts
// offloaded.
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "cli/common.h"
#include "core/update.h"
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
  std::string name, served, host = "127.0.0.1";
  cli::FormatFlags format;
  int port = 8080, context = 0, sequences = 0;
  bool prefix_cache = true;
  double memory_fraction = 0.92;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (cli::parse_format_flag(argc, argv, i, format)) continue;
    if (a == "--host") host = value();
    else if (a == "--port") port = std::stoi(value());
    else if (a == "--name") served = value();
    else if (a == "--context") context = std::stoi(value());
    else if (a == "--sequences") sequences = std::stoi(value());
    else if (a == "--no-prefix-cache") prefix_cache = false;
    else if (a == "--memory-fraction") memory_fraction = std::stod(value());
    else if (name.empty() && a[0] != '-') name = a;
    else throw std::runtime_error("unknown option " + a);
  }
  if (name.empty()) {
    std::fprintf(stderr,
                 "usage: lernex serve MODEL [--host 127.0.0.1] [--port 8080] [--name NAME]\n"
                 "                     %s\n"
                 "                     [--context N] [--sequences N] [--no-prefix-cache]\n",
                 cli::kFormatUsage);
    return 2;
  }

  update::start_daily_check();
  const cli::Model model = cli::find_model(name, true);
  runtime::EngineOptions engine_options;
  engine_options.prefix_cache = prefix_cache;
  cli::StartedEngine started =
      cli::start_engine(model, sequences, context, format.weights, engine_options, memory_fraction, format.bf16x,
                        format.fast_math);
  const runtime::Plan& plan = started.plan;
  runtime::Engine& engine = *started.engine;

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
  if (!engine.placement().empty()) std::fprintf(stderr, "%s", engine.placement().c_str());
  std::fprintf(stderr, "Serving %s on http://%s:%d/v1  (Ctrl-C to stop)\n", name.c_str(), host.c_str(), port);
  const bool ok = server.run();
  g_server = nullptr;
  std::fprintf(stderr, "Stopped.\n");
  return ok ? 0 : 1;
}
