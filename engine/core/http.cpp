#include "core/http.h"

#include <cctype>
#include <cstdlib>
#include <mutex>
#include <stdexcept>

#include "lx_version.h"
#include "platform/posix.h"

#if defined(_WIN32)
#include <windows.h>
#include <winhttp.h>

#include <algorithm>
#include <chrono>
#endif

namespace lx::http {
#if defined(_WIN32)
// Windows: WinHTTP, the system's own client. TLS comes from Schannel and the system's certificate store, the proxy
// from the system's settings (or WPAD), so nothing is bundled and a machine that browses the web can download.
// Redirects are followed here, not by WinHTTP, so the client behaves as the libcurl one does: up to 20 of them, the
// final URL reported, and an Authorization header dropped when a redirect leaves the host it was meant for (Hugging
// Face sends downloads to its CDN).
namespace {

std::wstring widen(const std::string& s) {
  if (s.empty()) return std::wstring();
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring out(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
  return out;
}

std::string narrow(const wchar_t* s, size_t n) {
  if (n == 0) return std::string();
  const int m = WideCharToMultiByte(CP_UTF8, 0, s, static_cast<int>(n), nullptr, 0, nullptr, nullptr);
  std::string out(static_cast<size_t>(m), '\0');
  WideCharToMultiByte(CP_UTF8, 0, s, static_cast<int>(n), out.data(), m, nullptr, nullptr);
  return out;
}

std::string lower(std::string s) {
  for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

std::runtime_error failure(const char* what) {
  const DWORD code = GetLastError();
  std::string text;
  if (code == ERROR_WINHTTP_TIMEOUT) text = "the connection timed out";
  else if (code == ERROR_WINHTTP_NAME_NOT_RESOLVED) text = "the host name could not be resolved";
  else if (code == ERROR_WINHTTP_CANNOT_CONNECT) text = "could not connect";
  else if (code == ERROR_WINHTTP_SECURE_FAILURE) text = "the TLS connection failed (certificate or protocol)";
  else if (code == ERROR_WINHTTP_CONNECTION_ERROR) text = "the connection was reset";
  else {
    wchar_t* msg = nullptr;
    const DWORD n = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_HMODULE | FORMAT_MESSAGE_FROM_SYSTEM |
                                       FORMAT_MESSAGE_IGNORE_INSERTS,
                                   GetModuleHandleW(L"winhttp.dll"), code, 0, reinterpret_cast<wchar_t*>(&msg), 0, nullptr);
    text = n ? narrow(msg, n) : "error " + std::to_string(code);
    if (msg) LocalFree(msg);
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == '.')) text.pop_back();
  }
  return std::runtime_error(std::string("transfer failed: ") + what + ": " + text + " (" + std::to_string(code) + ")");
}

struct Handle {
  HINTERNET h = nullptr;
  ~Handle() {
    if (h) WinHttpCloseHandle(h);
  }
};

// One session for the process: WinHTTP's sessions are thread-safe, and it keeps connections alive between requests.
HINTERNET session() {
  static HINTERNET s = [] {
    const std::wstring agent = widen("dromos/" LX_VERSION_STRING);
    HINTERNET h = WinHttpOpen(agent.c_str(), WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
                              WINHTTP_NO_PROXY_BYPASS, 0);
    if (!h)  // before Windows 8.1
      h = WinHttpOpen(agent.c_str(), WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (h) {
      DWORD protocols = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2 | WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_3;
      if (!WinHttpSetOption(h, WINHTTP_OPTION_SECURE_PROTOCOLS, &protocols, sizeof protocols)) {
        protocols = WINHTTP_FLAG_SECURE_PROTOCOL_TLS1_2;  // no TLS 1.3 before Windows 11
        WinHttpSetOption(h, WINHTTP_OPTION_SECURE_PROTOCOLS, &protocols, sizeof protocols);
      }
    }
    return h;
  }();
  return s;
}

struct Url {
  std::wstring scheme, host, path;  // path includes the query
  INTERNET_PORT port = 0;
  bool https = false;
};

Url parse(const std::string& url) {
  const std::wstring w = widen(url);
  URL_COMPONENTS c{};
  c.dwStructSize = sizeof c;
  c.dwSchemeLength = c.dwHostNameLength = c.dwUrlPathLength = c.dwExtraInfoLength = static_cast<DWORD>(-1);
  if (!WinHttpCrackUrl(w.c_str(), static_cast<DWORD>(w.size()), 0, &c)) throw std::runtime_error("not a URL: " + url);
  Url u;
  u.scheme.assign(c.lpszScheme, c.dwSchemeLength);
  u.host.assign(c.lpszHostName, c.dwHostNameLength);
  u.path.assign(c.lpszUrlPath, c.dwUrlPathLength);
  if (c.dwExtraInfoLength) u.path.append(c.lpszExtraInfo, c.dwExtraInfoLength);
  if (u.path.empty()) u.path = L"/";
  u.port = c.nPort;
  u.https = c.nScheme == INTERNET_SCHEME_HTTPS;
  return u;
}

// A Location header resolved against the URL it came from: absolute, host-relative (/path) or path-relative.
std::string resolve(const std::string& base, const std::string& location) {
  if (location.find("://") != std::string::npos) return location;
  const size_t scheme_end = base.find("://");
  const size_t host_end = base.find('/', scheme_end == std::string::npos ? 0 : scheme_end + 3);
  const std::string origin = host_end == std::string::npos ? base : base.substr(0, host_end);
  if (location.rfind("//", 0) == 0) return base.substr(0, scheme_end + 1) + location;
  if (!location.empty() && location[0] == '/') return origin + location;
  std::string dir = host_end == std::string::npos ? origin + "/" : base.substr(0, base.find_first_of("?#", host_end));
  dir = dir.substr(0, dir.rfind('/') + 1);
  return dir + location;
}

bool is_authorization(const std::string& header) { return lower(header.substr(0, 14)) == "authorization:"; }

}  // namespace

bool available() { return session() != nullptr; }

Response get(const std::string& url, const std::vector<std::string>& headers, const BodySink& body,
             long timeout_seconds) {
  HINTERNET s = session();
  if (!s) throw failure("WinHttpOpen");
  using Clock = std::chrono::steady_clock;
  const Clock::time_point deadline = Clock::now() + std::chrono::seconds(timeout_seconds > 0 ? timeout_seconds : 0);
  // The libcurl client's limits: 30 s to connect, and a transfer that moves nothing for a minute is dead. A
  // request with a timeout gets no step longer than the time it has left.
  auto ms_left = [&](int cap_ms) {
    if (timeout_seconds <= 0) return cap_ms;
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    return static_cast<int>(std::max<long long>(1, std::min<long long>(cap_ms, left)));
  };
  std::string current = url;
  const std::string first_host = narrow(parse(url).host.c_str(), parse(url).host.size());
  for (int hop = 0; hop <= 20; ++hop) {
    const Url u = parse(current);
    Handle connect{WinHttpConnect(s, u.host.c_str(), u.port, 0)};
    if (!connect.h) throw failure("connect");
    Handle request{WinHttpOpenRequest(connect.h, L"GET", u.path.c_str(), nullptr, WINHTTP_NO_REFERER,
                                      WINHTTP_DEFAULT_ACCEPT_TYPES, u.https ? WINHTTP_FLAG_SECURE : 0)};
    if (!request.h) throw failure("open request");
    DWORD policy = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
    WinHttpSetOption(request.h, WINHTTP_OPTION_REDIRECT_POLICY, &policy, sizeof policy);
    WinHttpSetTimeouts(request.h, ms_left(30000), ms_left(30000), ms_left(60000), ms_left(60000));
    const std::string host = narrow(u.host.c_str(), u.host.size());
    std::wstring extra;
    for (const std::string& h : headers) {
      if (is_authorization(h) && lower(host) != lower(first_host)) continue;
      extra += widen(h) + L"\r\n";
    }
    if (!WinHttpSendRequest(request.h, extra.empty() ? WINHTTP_NO_ADDITIONAL_HEADERS : extra.c_str(),
                            extra.empty() ? 0 : static_cast<DWORD>(-1L), WINHTTP_NO_REQUEST_DATA, 0, 0, 0))
      throw failure("send");
    if (!WinHttpReceiveResponse(request.h, nullptr)) throw failure("receive");

    Response response;
    response.url = current;
    DWORD status = 0, size = sizeof status;
    WinHttpQueryHeaders(request.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX,
                        &status, &size, WINHTTP_NO_HEADER_INDEX);
    response.status = static_cast<long>(status);
    DWORD raw_bytes = 0;
    WinHttpQueryHeaders(request.h, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, nullptr, &raw_bytes,
                        WINHTTP_NO_HEADER_INDEX);
    if (raw_bytes) {
      std::wstring raw(raw_bytes / sizeof(wchar_t), L'\0');
      if (WinHttpQueryHeaders(request.h, WINHTTP_QUERY_RAW_HEADERS_CRLF, WINHTTP_HEADER_NAME_BY_INDEX, raw.data(),
                              &raw_bytes, WINHTTP_NO_HEADER_INDEX)) {
        const std::string text = narrow(raw.c_str(), raw_bytes / sizeof(wchar_t));
        size_t at = text.find("\r\n");  // past the status line
        while (at != std::string::npos && at + 2 < text.size()) {
          const size_t end = text.find("\r\n", at + 2);
          const std::string line = text.substr(at + 2, end == std::string::npos ? std::string::npos : end - at - 2);
          const size_t colon = line.find(':');
          if (colon != std::string::npos) {
            std::string value = line.substr(colon + 1);
            const size_t first = value.find_first_not_of(" \t");
            value = first == std::string::npos ? "" : value.substr(first);
            response.headers[lower(line.substr(0, colon))] = value;
          }
          at = end;
        }
      }
    }
    const bool redirect = status == 301 || status == 302 || status == 303 || status == 307 || status == 308;
    if (redirect && response.headers.count("location")) {
      current = resolve(current, response.headers["location"]);
      continue;
    }
    std::vector<char> buf(1 << 16);
    for (;;) {
      if (timeout_seconds > 0 && Clock::now() >= deadline) {
        SetLastError(ERROR_WINHTTP_TIMEOUT);
        throw failure("read");
      }
      DWORD got = 0;
      if (!WinHttpReadData(request.h, buf.data(), static_cast<DWORD>(buf.size()), &got)) throw failure("read");
      if (got == 0) break;
      if (!body(buf.data(), got, response)) break;  // the caller has what it needs
    }
    return response;
  }
  throw std::runtime_error("transfer failed: more than 20 redirects from " + url);
}
#else
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
constexpr int kInfoEffectiveUrl = 0x100001;  // CURLINFO_EFFECTIVE_URL: the string, after redirects
constexpr int kOptTimeout = 13;              // CURLOPT_TIMEOUT: seconds for the whole transfer

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
  // Never freed: a detached thread (the daily update check) may still be inside libcurl when the process exits.
  static std::once_flag once;
  static Curl* loaded = nullptr;
  std::call_once(once, [] {
    void* h = nullptr;
    for (const char* name : {"libcurl.so.4", "libcurl.4.dylib", "libcurl.dylib", "libcurl.so"}) {
      h = dlopen(name, RTLD_NOW | RTLD_LOCAL);
      if (h) break;
    }
    if (!h) return;
    auto* c = new Curl();
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
    if (c->global_init(kGlobalDefault) != 0) {
      delete c;
      return;
    }
    loaded = c;
  });
  return loaded;
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

Response get(const std::string& url, const std::vector<std::string>& headers, const BodySink& body,
             long timeout_seconds) {
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
  c->setopt(easy, kOptUserAgent, "dromos/" LX_VERSION_STRING);
  if (timeout_seconds > 0) c->setopt(easy, kOptTimeout, timeout_seconds);
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
  const char* effective = nullptr;
  if (c->getinfo(easy, kInfoEffectiveUrl, &effective) == 0 && effective) response.url = effective;
  return response;
}

#endif

Response get_string(const std::string& url, const std::vector<std::string>& headers, std::string& body) {
  body.clear();
  return get(url, headers, [&body](const char* data, size_t size, const Response&) {
    body.append(data, size);
    return true;
  });
}

}  // namespace lx::http
