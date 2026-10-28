// HTTPS client for model downloads, over the system's libcurl. The library is loaded at run time
// (dlopen), so the engine builds and runs on machines with no curl development headers.
#pragma once

#include <stddef.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace lx::http {

// True when a libcurl could be loaded.
bool available();

struct Response {
  long status = 0;  // the last HTTP status line seen (after any redirects)
  std::string url;  // the URL the final response came from, after any redirects
  std::map<std::string, std::string> headers;  // lower-case names
};

// Called with each piece of the body, with the response headers and status seen so far (the final
// status, since they come before the body). Return false to abort the transfer.
using BodySink = std::function<bool(const char* data, size_t size, const Response& so_far)>;

// One GET, following redirects. Throws std::runtime_error when the transfer itself fails (network, timeout), not
// for an HTTP error status, which comes back in Response::status. `timeout_seconds` bounds the whole transfer (0: no
// bound beyond the stall check, which every transfer has).
Response get(const std::string& url, const std::vector<std::string>& headers, const BodySink& body,
             long timeout_seconds = 0);

// Shorthand: the body as a string.
Response get_string(const std::string& url, const std::vector<std::string>& headers, std::string& body);

}  // namespace lx::http
