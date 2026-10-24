cmake_minimum_required(VERSION 3.28)
# Blackwell consumer (RTX 50) for development; release builds list every architecture. 120a, not
# 120: the block-scaled FP4 / FP8 tensor-core instructions exist only in the arch-specific target.
# Set before project(), which otherwise fills in an old default. Real code only: PTX from a
# toolkit newer than the driver would not load.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a-real)
endif()
project(lernex_engine LANGUAGES CXX CUDA)

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

# The runtime: config, weights, the Metis model, caches and the scheduler.
file(GLOB LX_CORE_SOURCES CONFIGURE_DEPENDS engine/core/*.cpp engine/model/*.cpp engine/model/*.cu engine/runtime/*.cpp
     engine/runtime/*.cu engine/text/*.cpp engine/server/*.cpp)
if(LX_CORE_SOURCES)
  add_library(lernex_core STATIC ${LX_CORE_SOURCES})
  target_include_directories(lernex_core PUBLIC engine third_party)
  target_link_libraries(lernex_core PUBLIC lernex_kernels)
endif()

file(GLOB LX_CLI_SOURCES CONFIGURE_DEPENDS cli/*.cpp)
if(LX_CLI_SOURCES AND TARGET lernex_core)
  add_executable(lernex ${LX_CLI_SOURCES})
  target_link_libraries(lernex PRIVATE lernex_core)
endif()

# The text stack's tests: tokenizer, chat template and output parser. No CUDA; the goldens come from
# tools/text/*.py and are read through LX_TEXT_GOLDEN (see tests/text).
file(GLOB LX_TEXT_TEST_SOURCES CONFIGURE_DEPENDS tests/text/*.cpp)
file(GLOB LX_TEXT_SOURCES CONFIGURE_DEPENDS engine/text/*.cpp)
add_executable(lx_text_tests ${LX_TEXT_TEST_SOURCES} ${LX_TEXT_SOURCES})
target_include_directories(lx_text_tests PRIVATE engine third_party)
