// What the commands share: finding a model folder, the plan for this GPU, and small formatting helpers.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include "runtime/engine.h"
#include "runtime/plan.h"

namespace lx::cli {

// A model the commands can run: a folder on disk, and the name it was asked for.
struct Model {
  std::filesystem::path dir;
  std::string name;  // what the user typed: "praxis:fp8", a repository, or a folder
  std::string repo;  // the hub repository, when it came from one
};

// Finds the model a name refers to. A folder path is used as it is. An alias or repository must be
// installed in the cache; when `download` is set and it is not, it is pulled first, with progress.
// Throws std::runtime_error with a plain message.
Model find_model(const std::string& name, bool download);

// The plan for this GPU, the model, and the user's wishes (0 and -1 for automatic).
runtime::Plan plan_for(const Model& model, int sequences, int context, int weights = -1);

// The sampling a model recommends (generation_config.json), for a chat that asks for nothing else.
runtime::SamplingParams recommended_sampling(const Model& model);

std::string human_bytes(uint64_t bytes);
std::string human_seconds(double seconds);

// Whether stderr is a terminal that can redraw a progress line.
bool stderr_is_terminal();

// Whether a name is a folder on disk rather than an alias or a repository.
bool looks_like_path(const std::string& name);

}  // namespace lx::cli
