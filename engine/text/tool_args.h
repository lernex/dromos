// The body of a Metis tool call and its arguments, as the reference parser builds them: the
// <parameter=...> blocks as a JSON object, the name fallbacks, and the typing of each value
// against the request's tool schema.
#pragma once

#include <string>
#include <string_view>

#include "text/json.h"

namespace lx::text {

// The "properties" object of the tool named `name` in an OpenAI-style tools array, or nullptr.
const JValue* find_tool_properties(const JValue& tools, std::string_view name);

// metis_arg_converter on a whole body: the parameter blocks as a JSON object of strings. A block
// that is never closed is left out, as the whole-text parse leaves it out.
std::string convert_parameters(std::string_view body);

// The first "name": "..." string in a body (the regex fallback for a call with no name). Empty when
// there is none, or when the first match is empty.
std::string name_from_body(std::string_view body);

// A body without a name that is a JSON object: its "name" and its arguments (as JSON text, or the
// raw string when "arguments" is a string). False when the body is not a JSON object.
bool name_and_args_from_json(std::string_view body, std::string& name, std::string& args);

// The arguments JSON with each value typed by the schema of the tool named `name`, re-serialized
// only when something changed (_fix_arg_types).
std::string fix_arg_types(const std::string& args, std::string_view name, const JValue& tools);

}  // namespace lx::text
