// The OpenAI-compatible HTTP server: /v1/chat/completions, /v1/completions, /v1/models, /health.
// Requests become engine requests; tokens stream back as server-sent events as they are sampled.
#pragma once

#include <memory>
#include <string>

#include "runtime/engine.h"

namespace httplib {
class Server;
}

namespace lx::server {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string model_name;   // the name clients use; requests naming another model are refused
  std::string model_dir;    // read for generation_config.json: the defaults for sampling
  int threads = 64;         // connections served at once
};

class Server {
 public:
  Server(runtime::Engine& engine, ServerOptions options);
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  // Binds the address and serves until stop() is called. Returns false (with the reason on stderr)
  // when the address cannot be bound.
  bool run();
  // Stops serving; in-flight requests are cancelled as their connections close.
  void stop();

  // The port actually bound, once run() has bound it.
  int port() const { return bound_port_; }

 private:
  runtime::Engine& engine_;
  ServerOptions options_;
  std::unique_ptr<httplib::Server> http_;
  int bound_port_ = 0;
};

}  // namespace lx::server
