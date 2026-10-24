cmake_minimum_required(VERSION 3.28)
if(APPLE)
  set(APPLE_DEFAULT ON)
  set(NOT_APPLE_DEFAULT OFF)
else()
  set(APPLE_DEFAULT OFF)
  set(NOT_APPLE_DEFAULT ON)
endif()
# Backends: each is one shared library exporting every lx_* function of engine/cuda/kernels.h.
#   LX_BACKEND_CUDA   liblernex_kernels.so (the default off macOS)
#   LX_BACKEND_CPU    liblernex_cpu.so, x86 (AVX2, AVX-VNNI, AVX-512) and ARM (NEON, dot product, I8MM,
#                     BF16), no CUDA needed: cmake -DLX_BACKEND_CUDA=OFF -DLX_BACKEND_CPU=ON. Built by
#                     default next to CUDA too: the offload runtime loads it for its CPU lane.
#   LX_BACKEND_METAL  liblernex_metal.dylib, Apple silicon GPUs (the default on macOS; engine/metal)
# The model, runtime and CLI (lernex) link no backend and build with any of them: lernex opens one at run
# time (engine/dispatch).
option(LX_BACKEND_CUDA "Build the CUDA backend" ${NOT_APPLE_DEFAULT})
option(LX_BACKEND_CPU "Build the CPU backend (liblernex_cpu.so)" ${NOT_APPLE_DEFAULT})
option(LX_BACKEND_METAL "Build the Metal backend (liblernex_metal.dylib)" ${APPLE_DEFAULT})

# The CUDA architectures this build compiles for, chosen with -DLX_ARCHS:
#   120a-real  the development default (RTX 50), so an edit-compile loop stays fast
#   all        every architecture the engine runs on: RTX 30 / A100 (80, 86), RTX 40 (89), H100 (90a),
#              B200 (100a) and RTX 50 (120a). The "a" targets carry the arch-specific instructions
#              (wgmma on 90a, block-scaled FP8 / FP4 on 100a and 120a); code that needs them is guarded
#              by __CUDA_ARCH__ in the sources, so every target compiles.
#   or a CMake architecture list such as 89-real;120a-real
# Real code only: PTX from a toolkit newer than the driver would not load. Set before project(), which
# otherwise fills in an old default.
set(LX_ARCHS "120a-real" CACHE STRING "CUDA architectures: 120a-real (development), all, or a CMake list")
if(LX_ARCHS STREQUAL "all")
  set(LX_ARCH_LIST "80-real;86-real;89-real;90a-real;100a-real;120a-real")
else()
  set(LX_ARCH_LIST "${LX_ARCHS}")
endif()
if(LX_BACKEND_CUDA)
  set(CMAKE_CUDA_ARCHITECTURES ${LX_ARCH_LIST})
endif()
project(lernex_engine LANGUAGES CXX)
if(LX_BACKEND_METAL)
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 20)
endif()

# The compute capabilities this build has code for (80, 86, 89, 90, 100, 120), for the device report.
set(LX_BUILT_ARCHS "")
foreach(arch IN LISTS LX_ARCH_LIST)
  string(REGEX MATCH "^[0-9]+" cc "${arch}")
  list(APPEND LX_BUILT_ARCHS "${cc}")
endforeach()
list(JOIN LX_BUILT_ARCHS "," LX_BUILT_ARCHS_CSV)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(LX_BACKEND_CPU)
  add_subdirectory(engine/cpu)
endif()
if(LX_BACKEND_METAL)
  include(engine/metal/metal.cmake)
endif()

# The dispatcher: the lx_* stubs the host calls, and the loader that opens one backend library at run time
# (engine/dispatch). It holds no GPU code, so it builds with or without CUDA, and lernex_core links it in place of
# liblernex_kernels. tools/gen_dispatch.py writes the stubs and the table from the ABI headers.
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_Interpreter_FOUND)
  message(FATAL_ERROR "python3 is needed to generate the backend dispatch table (tools/gen_dispatch.py)")
endif()
include(GNUInstallDirs)
set(LX_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
if(NOT LX_INSTALL_LIBDIR)
  set(LX_INSTALL_LIBDIR "lib")
endif()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
set(LX_DISPATCH_GEN "${CMAKE_BINARY_DIR}/generated/lx_dispatch_gen.cpp")
add_custom_command(
  OUTPUT "${LX_DISPATCH_GEN}"
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/gen_dispatch.py" "${LX_DISPATCH_GEN}"
          "${CMAKE_SOURCE_DIR}/engine/cuda/kernels.h" "${CMAKE_SOURCE_DIR}/engine/cuda/tunable.h"
  DEPENDS "${CMAKE_SOURCE_DIR}/tools/gen_dispatch.py" "${CMAKE_SOURCE_DIR}/engine/cuda/kernels.h"
          "${CMAKE_SOURCE_DIR}/engine/cuda/tunable.h"
  COMMENT "Generating the backend dispatch table"
  VERBATIM)
add_library(lernex_dispatch STATIC engine/dispatch/dispatch.cpp "${LX_DISPATCH_GEN}")
target_include_directories(lernex_dispatch PUBLIC engine)
target_compile_definitions(lernex_dispatch PRIVATE "LX_INSTALL_LIBDIR=\"${LX_INSTALL_LIBDIR}\"")
target_link_libraries(lernex_dispatch PUBLIC ${CMAKE_DL_LIBS})

if(LX_BACKEND_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -lineinfo")

  find_package(CUDAToolkit REQUIRED)

  # Every kernel, behind a C ABI (engine/cuda/kernels.h) so the tests can call them through ctypes.
  file(GLOB LX_CUDA_SOURCES CONFIGURE_DEPENDS engine/cuda/*.cu)
  add_library(lernex_kernels SHARED ${LX_CUDA_SOURCES})
  target_include_directories(lernex_kernels PUBLIC engine)
  target_link_libraries(lernex_kernels PUBLIC CUDA::cudart CUDA::cublasLt)
endif()

# The prefix cache snapshots the model's slot state. That needs MetisModel::save_slot and restore_slot,
# which live in docs/handoff/serving-metis.diff until the lead applies it to engine/model/metis.cpp.
option(LX_SLOT_STATE "Build the prefix cache on the model's slot state" ON)
if(LX_SLOT_STATE)
  add_compile_definitions(LX_SLOT_STATE=1)
endif()

# The runtime: config, weights, the Metis model, caches, the plan and the tuner.
file(GLOB LX_CORE_SOURCES CONFIGURE_DEPENDS engine/core/*.cpp engine/model/*.cpp engine/model/*.cu engine/runtime/*.cpp
     engine/runtime/*.cu engine/text/*.cpp engine/server/*.cpp)
if(LX_CORE_SOURCES)
  add_library(lernex_core STATIC ${LX_CORE_SOURCES})
  target_include_directories(lernex_core PUBLIC engine third_party third_party/httplib)
  target_link_libraries(lernex_core PUBLIC lernex_dispatch)
  # The architectures this build has code for, for the device report. A header, not a definition: nvcc
  # splits a -D value on its commas.
  file(WRITE "${CMAKE_BINARY_DIR}/generated/lx_build_info.h"
       "#pragma once\n#define LX_BUILT_ARCHS ${LX_BUILT_ARCHS_CSV}\n")
  target_include_directories(lernex_core PUBLIC "${CMAKE_BINARY_DIR}/generated")
  # The tuning tables shipped in the repository (tuning/), found next to the source tree.
  target_compile_definitions(lernex_core PRIVATE "LX_TUNING_DIR=\"${CMAKE_SOURCE_DIR}/tuning\"")
endif()

file(GLOB LX_CLI_SOURCES CONFIGURE_DEPENDS cli/*.cpp)
if(LX_CLI_SOURCES AND TARGET lernex_core)
  add_executable(lernex ${LX_CLI_SOURCES})
  target_include_directories(lernex PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(lernex PRIVATE lernex_core)
  # The backend libraries are opened at run time from next to the executable, so the CLI builds them too.
  foreach(backend lernex_kernels lernex_cpu lernex_metal)
    if(TARGET ${backend})
      add_dependencies(lernex ${backend})
    endif()
  endforeach()
endif()

# The C++ unit tests (doctest): host-side policy, parsing and protocol code, run on the server.
file(GLOB LX_TEST_SOURCES CONFIGURE_DEPENDS tests/cpp/*.cpp)
if(LX_TEST_SOURCES AND TARGET lernex_core)
  add_executable(lernex_tests ${LX_TEST_SOURCES})
  target_include_directories(lernex_tests PRIVATE engine third_party)
  target_link_libraries(lernex_tests PRIVATE lernex_core)
endif()

# GPU checks: the engine end to end on a real model (run by hand on the server, see tests/gpu).
file(GLOB LX_GPU_CHECK_SOURCES CONFIGURE_DEPENDS tests/gpu/*.cpp)
if(LX_GPU_CHECK_SOURCES AND TARGET lernex_core)
  add_executable(lernex_gpu_check ${LX_GPU_CHECK_SOURCES})
  target_link_libraries(lernex_gpu_check PRIVATE lernex_core)
endif()

# The text stack's tests: tokenizer, chat template and output parser. No CUDA; the goldens come from
# tools/text/*.py and are read through LX_TEXT_GOLDEN (see tests/text).
file(GLOB LX_TEXT_TEST_SOURCES CONFIGURE_DEPENDS tests/text/*.cpp)
file(GLOB LX_TEXT_SOURCES CONFIGURE_DEPENDS engine/text/*.cpp)
add_executable(lx_text_tests ${LX_TEXT_TEST_SOURCES} ${LX_TEXT_SOURCES})
target_include_directories(lx_text_tests PRIVATE engine third_party)

# Unit tests for the C++ modules (plan, device report, tuner), with doctest (third_party/doctest).
file(GLOB LX_UNIT_SOURCES CONFIGURE_DEPENDS tests/unit/*.cpp)
if(LX_UNIT_SOURCES AND TARGET lernex_core)
  add_executable(lernex_unit ${LX_UNIT_SOURCES})
  target_include_directories(lernex_unit PRIVATE third_party/doctest)
  target_link_libraries(lernex_unit PRIVATE lernex_core)
endif()

# The prefix cache, end to end on the model (run on the GPU server, see tests/gpu/prefix).
if(LX_SLOT_STATE AND TARGET lernex_core)
  add_executable(lernex_prefix_check tests/gpu/prefix/prefix_check.cpp)
  target_link_libraries(lernex_prefix_check PRIVATE lernex_core)
endif()
