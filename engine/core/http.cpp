#include "core/http.h"

#include <dlfcn.h>

#include <cctype>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace lx::http {
namespace {

// The few libcurl entry points and options the client uses, declared here so no curl header is needed.
using CURL = void;
using CURLcode = int;
using CurlWrite = size_t (*)(char*, size_t, size_t, void*);

constexpr long kGlobalDefault = 3;
constexpr int kOptUrl = 10002, kOptWriteData = 10001, kOptHeaderData = 10029, kOptHttpHeader = 10023,
              kOptUserAgent = 10018, kOptWriteFunction = 20011, kOptHeaderFunction = 20079,
              kOptFollowLocation = 52, kOptConnectTimeout = 78, kOptLowSpeedLimit = 19, kOptLowSpeedTime = 20,
              kOptNoSignal = 99;
constexpr int kInfoResponseCode = 0x200002;

struct Curl {
  void* handle = nullptr;
  CURL* (*easy_init)() = nullptr;
  CURLcode (*setopt)(CURL*, int, ...) = nullptr;
  CURLcode (*perform)(CURL*) = nullptr;
  void (*cleanup)(CURL*) = nullptr;
  CURLcode (*getinfo)(CURL*, int, ...) = nullptr;
  const char* (*strerror)(CURLcode) = nullptr;
  void* (*slist_append)(void*, const char*) = nullptr;
  void (*slist_free_all)(void*) = nullptr;
  CURLcode (*global_init)(long) = nullptr;
};

Curl* load() {
  static std::once_flag once;
  static std::unique_ptr<Curl> loaded;
  std::call_once(once, [] {
    void* h = nullptr;
    for (const char* name : {"libcurl.so.4", "libcurl.4.dylib", "libcurl.dylib", "libcurl.so"}) {
      h = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (h) break;
    }
    if (!h) return;
    auto c = std::make_unique<Curl>();
    c->handle = h;
    auto sym = [h](const char* name) { return dlsym(h, name); };
    c->easy_init = reinterpret_cast<decltype(c->easy_init)>(sym("curl_easy_init"));
    c->setopt = reinterpret_cast<decltype(c->setopt)>(sym("curl_easy_setopt"));
    c->perform = reinterpret_cast<decltype(c->perform)>(sym("curl_easy_perform"));
    c->cleanup = reinterpret_cast<decltype(c->cleanup)>(sym("curl_easy_cleanup"));
    c->getinfo = reinterpret_cast<decltype(c->getinfo)>(sym("curl_easy_getinfo"));
    c->strerror = reinterpret_cast<decltype(c->strerror)>(sym("curl_easy_strerror"));
    c->slist_append = reinterpret_cast<decltype(c->slist_append)>(sym("curl_slist_append"));
    c->slist_free_all = reinterpret_cast<decltype(c->slist_free_all)>(sym("curl_slist_free_all"));
    c->global_init = reinterpret_cast<decltype(c->global_init)>(sym("curl_global_init"));
    if (!c->easy_init || !c->setopt || !c->perform || !c->cleanup || !c->getinfo || !c->strerror ||
        !c->slist_append || !c->slist_free_all || !c->global_init)
      return;
    if (c->global_init(kGlobalDefault) != 0) return;
    loaded = std::move(c);
  });
  return loaded.get();
}

std::string lower(std::string s) {
  for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

struct Context {
  const BodySink* body = nullptr;
  Response* response = nullptr;
  bool aborted = false;
};

size_t on_body(char* data, size_t size, size_t count, void* user) {
  auto* ctx = static_cast<Context*>(user);
  const size_t n = size * count;
  if (ctx->aborted) return 0;
  if (!(*ctx->body)(data, n, *ctx->response)) {
    ctx->aborted = true;
    return 0;
  }
  return n;
}

size_t on_header(char* data, size_t size, size_t count, void* user) {
  auto* ctx = static_cast<Context*>(user);
  const size_t n = size * count;
  std::string line(data, n);
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
  if (line.rfind("HTTP/", 0) == 0) {
    // a new status line: redirects and "100 Continue" come before the final one
    ctx->response->headers.clear();
    const size_t space = line.find(' ');
    ctx->response->status = space == std::string::npos ? 0 : std::atol(line.c_str() + space + 1);
    return n;
  }
  const size_t colon = line.find(':');
  if (colon != std::string::npos) {
    std::string value = line.substr(colon + 1);
    const size_t first = value.find_first_not_of(" \t");
    value = first == std::string::npos ? "" : value.substr(first);
    ctx->response->headers[lower(line.substr(0, colon))] = value;
  }
  return n;
}

}  // namespace

bool available() { return load() != nullptr; }

Response get(const std::string& url, const std::vector<std::string>& headers, const BodySink& body) {
  Curl* c = load();
  if (!c) throw std::runtime_error("libcurl is not available on this system");
  CURL* easy = c->easy_init();
  if (!easy) throw std::runtime_error("curl_easy_init failed");
  struct Free {
    Curl* c;
    CURL* easy;
    void* list = nullptr;
    ~Free() {
      if (list) c->slist_free_all(list);
      c->cleanup(easy);
    }
  } guard{c, easy};
  for (const std::string& h : headers) guard.list = c->slist_append(guard.list, h.c_str());

  Response response;
  Context ctx{&body, &response, false};
  c->setopt(easy, kOptUrl, url.c_str());
  c->setopt(easy, kOptFollowLocation, 1L);
  c->setopt(easy, kOptNoSignal, 1L);
  c->setopt(easy, kOptConnectTimeout, 30L);
  // a connection that moves less than a kilobyte a second for a minute is dead
  c->setopt(easy, kOptLowSpeedLimit, 1024L);
  c->setopt(easy, kOptLowSpeedTime, 60L);
  c->setopt(easy, kOptUserAgent, "lernex/0.1");
  if (guard.list) c->setopt(easy, kOptHttpHeader, guard.list);
  c->setopt(easy, kOptWriteFunction, reinterpret_cast<CurlWrite>(&on_body));
  c->setopt(easy, kOptWriteData, &ctx);
  c->setopt(easy, kOptHeaderFunction, reinterpret_cast<CurlWrite>(&on_header));
  c->setopt(easy, kOptHeaderData, &ctx);
  const CURLcode rc = c->perform(easy);
  if (rc != 0 && !ctx.aborted) throw std::runtime_error(std::string("transfer failed: ") + c->strerror(rc));
  long code = 0;
  c->getinfo(easy, kInfoResponseCode, &code);
  response.status = code ? code : response.status;
  return response;
}

Response get_string(const std::string& url, const std::vector<std::string>& headers, std::string& body) {
  body.clear();
  return get(url, headers, [&body](const char* data, size_t size, const Response&) {
    body.append(data, size);
    return true;
  });
}

}  // namespace lx::http
