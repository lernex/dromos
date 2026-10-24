# The lernex CLI on the Metal library: the end-to-end check that liblernex_metal runs a real model.
# The model code is the repository's, ported to the runtime ABI by docs/handoff/metal-metis.diff;
# engine/metal/e2e/prepare.sh applies it to a copy, and LX_E2E_TREE names that copy:
#
#   engine/metal/e2e/prepare.sh build-metal/tree
#   cmake -S . -B build-metal -G Ninja -DLX_BACKEND_METAL=ON -DLX_E2E_TREE=$PWD/build-metal/tree
#   ninja -C build-metal lernex
set(LX_E2E_TREE "" CACHE PATH "A copy of engine/core, engine/model and cli with docs/handoff/metal-metis.diff applied")
if(LX_E2E_TREE)
  add_executable(lernex
    ${LX_E2E_TREE}/engine/core/config.cpp
    ${LX_E2E_TREE}/engine/core/safetensors.cpp
    ${LX_E2E_TREE}/engine/core/mapped_weights.cpp
    ${LX_E2E_TREE}/engine/model/metis.cpp
    ${LX_E2E_TREE}/cli/eval.cpp
    ${CMAKE_SOURCE_DIR}/engine/metal/e2e/main.cpp)
  target_include_directories(lernex BEFORE PRIVATE ${LX_E2E_TREE}/engine ${CMAKE_SOURCE_DIR}/engine
                             ${CMAKE_SOURCE_DIR}/third_party)
  target_link_libraries(lernex PRIVATE lernex_metal)
  set_target_properties(lernex PROPERTIES CXX_STANDARD 20 BUILD_RPATH "$<TARGET_FILE_DIR:lernex_metal>")
endif()
