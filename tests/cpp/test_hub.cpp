#include <doctest/doctest.h>

#include <stdexcept>

#include "core/hub.h"

using lx::hub::resolve;

TEST_CASE("aliases resolve to their repositories, bf16 by default") {
  CHECK(resolve("praxis").repo == "Lernex/Metis-1.6-Praxis");
  CHECK(resolve("praxis").format == "bf16");
  CHECK(resolve("Logos").repo == "Lernex/Metis-1.6-Logos");
  CHECK(resolve("logos:bf16").repo == "Lernex/Metis-1.6-Logos");
}

TEST_CASE("the format suffix picks the repository") {
  CHECK(resolve("logos:fp8").repo == "Lernex/Metis-1.6-Logos-FP8");
  CHECK(resolve("praxis:fp8").repo == "Lernex/Metis-1.6-Praxis-FP8");
  CHECK(resolve("praxis:nvfp4").repo == "Lernex/Metis-1.6-Praxis-NVFP4");
  CHECK(resolve("praxis:nvfp4").alias == "praxis");
}

TEST_CASE("an owner/name repository is taken as it is") {
  auto ref = resolve("someone/their-model");
  CHECK(ref.repo == "someone/their-model");
  CHECK(ref.alias.empty());
}

TEST_CASE("unknown names and formats are refused with a reason") {
  CHECK_THROWS_AS(resolve("gpt"), std::invalid_argument);
  CHECK_THROWS_AS(resolve("praxis:q4"), std::invalid_argument);
  CHECK_THROWS_AS(resolve("a/b/c"), std::invalid_argument);
  CHECK_THROWS_AS(resolve("someone/model:fp8"), std::invalid_argument);
}

TEST_CASE("a needed file is a weight shard or a config at the top level") {
  CHECK(lx::hub::needed_file("model.safetensors"));
  CHECK(lx::hub::needed_file("model-00001-of-00002.safetensors"));
  CHECK(lx::hub::needed_file("model.safetensors.index.json"));
  CHECK(lx::hub::needed_file("tokenizer.json"));
  CHECK(lx::hub::needed_file("chat_template.jinja"));
  CHECK_FALSE(lx::hub::needed_file("modeling_metis16.py"));
  CHECK_FALSE(lx::hub::needed_file("NGRAM_CANONICAL_IDS.uint16"));
  CHECK_FALSE(lx::hub::needed_file("conversion_receipt.json"));
  CHECK_FALSE(lx::hub::needed_file("sub/model.safetensors"));
}
