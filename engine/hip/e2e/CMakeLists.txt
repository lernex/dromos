# The lernex CLI and the Metis forward pass on the HIP library: the model code, ported to lx_rt_* (see
# docs/handoff/hip-metis.diff), linked against liblernex_hip.so. Nothing here is compiled into the
# kernel library itself; this is the end-to-end check that the library runs a real model.
#
#   scripts/remote.sh hip run 'cmake -S engine/hip/e2e -B build/e2e -DLX_E2E_TREE=$PWD/proto'
#
# LX_E2E_TREE holds engine/core, engine/model and cli/ as they are in the repository with the handoff
# diff applied. The NVIDIA test build is the only one this file configures so far.
cmake_minimum_required(VERSION 3.21)

set(LX_E2E_TREE "" CACHE PATH "Tree with the ported engine/core, engine/model and cli/")
if(NOT LX_E2E_TREE)
  message(FATAL_ERROR "set LX_E2E_TREE to the ported model tree (engine/core, engine/model, cli)")
endif()
get_filename_component(LX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)

# The HIP library is built in its NVIDIA test configuration, from the same sources as the AMD build.
set(LX_HIP_PLATFORM nvidia)
set(LX_HIP_CUDA_ARCHS "120a")
set(CMAKE_CUDA_ARCHITECTURES ${LX_HIP_CUDA_ARCHS})

project(lernex_e2e LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(${LX_ROOT}/engine/hip ${CMAKE_BINARY_DIR}/hip)

find_package(CUDAToolkit REQUIRED)

set(LX_E2E_SOURCES
  ${LX_E2E_TREE}/engine/core/config.cpp
  ${LX_E2E_TREE}/engine/core/device.cpp
  ${LX_E2E_TREE}/engine/core/safetensors.cpp
  ${LX_E2E_TREE}/engine/model/metis.cpp
  ${LX_E2E_TREE}/engine/model/glue.cu
  ${LX_E2E_TREE}/cli/main.cpp
  ${LX_E2E_TREE}/cli/eval.cpp
  ${LX_E2E_TREE}/cli/inspect.cpp)

add_executable(lernex ${LX_E2E_SOURCES})
target_include_directories(lernex BEFORE PRIVATE
  ${LX_E2E_TREE}/engine
  ${LX_ROOT}/engine                     # cuda/kernels.h, the ABI
  ${LX_ROOT}/engine/hip/nvidia          # the HIP names over the CUDA runtime
  ${LX_ROOT}/engine/hip/include         # lx_hip.h
  ${LX_ROOT}/engine/hip/src             # common.cuh, the HIP port of the kernel helpers
  ${LX_ROOT}/third_party)
target_link_libraries(lernex PRIVATE lernex_hip CUDA::cudart)
