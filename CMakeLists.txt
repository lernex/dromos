cmake_minimum_required(VERSION 3.28)
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
set(CMAKE_CUDA_ARCHITECTURES ${LX_ARCH_LIST})
project(lernex_engine LANGUAGES CXX CUDA)

# The compute capabilities this build has code for (80, 86, 89, 90, 100, 120), for the device report.
set(LX_BUILT_ARCHS "")
foreach(arch IN LISTS LX_ARCH_LIST)
  string(REGEX MATCH "^[0-9]+" cc "${arch}")
  list(APPEND LX_BUILT_ARCHS "${cc}")
endforeach()
list(JOIN LX_BUILT_ARCHS "," LX_BUILT_ARCHS_CSV)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -lineinfo")

find_package(CUDAToolkit REQUIRED)

# Every kernel, behind a C ABI (engine/cuda/kernels.h) so the tests can call them through ctypes.
file(GLOB LX_CUDA_SOURCES CONFIGURE_DEPENDS engine/cuda/*.cu)
add_library(lernex_kernels SHARED ${LX_CUDA_SOURCES})
target_include_directories(lernex_kernels PUBLIC engine)
target_link_libraries(lernex_kernels PUBLIC CUDA::cudart CUDA::cublasLt)

# The runtime: config, weights, the Metis model, caches, the plan and the tuner.
file(GLOB LX_CORE_SOURCES CONFIGURE_DEPENDS engine/core/*.cpp engine/model/*.cpp engine/model/*.cu engine/runtime/*.cpp
     engine/runtime/*.cu engine/text/*.cpp engine/server/*.cpp)
if(LX_CORE_SOURCES)
  add_library(lernex_core STATIC ${LX_CORE_SOURCES})
  target_include_directories(lernex_core PUBLIC engine third_party)
  target_link_libraries(lernex_core PUBLIC lernex_kernels)
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
  target_link_libraries(lernex PRIVATE lernex_core)
endif()

# Unit tests for the C++ modules (plan, device report, tuner), with doctest (third_party/doctest).
file(GLOB LX_UNIT_SOURCES CONFIGURE_DEPENDS tests/unit/*.cpp)
if(LX_UNIT_SOURCES AND TARGET lernex_core)
  add_executable(lernex_unit ${LX_UNIT_SOURCES})
  target_include_directories(lernex_unit PRIVATE third_party/doctest)
  target_link_libraries(lernex_unit PRIVATE lernex_core)
endif()
