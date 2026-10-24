// Python's str and number rules, as the Metis template and parser use them: str.strip() (whitespace
// per str.isspace), int() and float() accepting underscores and Unicode decimal digits.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace lx::text {

// s without the leading and trailing characters str.isspace() accepts.
std::string_view py_strip(std::string_view s);

// Byte offset just past the last character that is not whitespace (0 when there is none).
size_t py_last_nonspace_end(std::string_view s);

// str.lower() for the ASCII letters the template and the coercions compare against.
std::string py_lower_ascii(std::string_view s);

// int(s) in base 10: canonical decimal digits (no leading zeros, "0" for zero). False when int()
// would raise.
bool py_int(std::string_view s, std::string& canonical);

// float(s). True when float() would accept the text; out may then be inf or nan, which the caller
// must check.
bool py_float(std::string_view s, double& out);

}  // namespace lx::text
