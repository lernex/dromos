# The HIP backend: liblernex_hip.so, the kernel library's C ABI (engine/cuda/kernels.h) on AMD GPUs.
# Standalone from the CUDA build, so a machine without nvcc configures and builds this one alone.
#
#   AMD (ROCm): the sources compile as HIP with ROCm's clang, the compiler hipcc runs underneath.
#     cmake -S engine/hip -B build/amd -DLX_HIP_PLATFORM=amd \
#           -DCMAKE_CXX_COMPILER=<rocm>/lib/llvm/bin/clang++ -DLX_HIP_ROCM_PATH=<rocm> \
#           -DLX_HIP_ARCHS="gfx1100;gfx1201;gfx942"
#   NVIDIA test build (the same sources through the CUDA-backed shim in nvidia/, see README.md):
#     cmake -S engine/hip -B build-hip -DLX_HIP_PLATFORM=nvidia -DLX_HIP_CUDA_ARCHS=120a
#
# The sources in src/ are generated from engine/cuda by translate.py; regenerate, do not hand-edit.
cmake_minimum_required(VERSION 3.21)

set(LX_HIP_PLATFORM "amd" CACHE STRING "amd (ROCm HIP) or nvidia (the CUDA-backed test shim)")
set_property(CACHE LX_HIP_PLATFORM PROPERTY STRINGS amd nvidia)
set(LX_HIP_ARCHS "gfx1100;gfx1201;gfx942" CACHE STRING "AMD GPU targets (LX_HIP_PLATFORM=amd)")
set(LX_HIP_ROCM_PATH "" CACHE PATH "ROCm root for the HIP headers and device libraries (LX_HIP_PLATFORM=amd)")
set(LX_HIP_DEVICE_LIB_PATH "" CACHE PATH "AMDGPU device bitcode, when it is not under the ROCm root (LX_HIP_PLATFORM=amd)")
set(LX_HIP_CUDA_ARCHS "120a" CACHE STRING "CUDA architectures (LX_HIP_PLATFORM=nvidia)")

if(LX_HIP_PLATFORM STREQUAL "nvidia")
  project(lernex_hip LANGUAGES CXX CUDA)
  set(CMAKE_CUDA_ARCHITECTURES ${LX_HIP_CUDA_ARCHS})
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(LX_HIP_LANG CUDA)
  find_package(CUDAToolkit REQUIRED)
elseif(LX_HIP_PLATFORM STREQUAL "amd")
  # CMake's HIP language wants the ROCm CMake packages; the sources are plain C++ with -x hip, which
  # is what hipcc runs, so the CXX language with ROCm's clang does the same without them.
  project(lernex_hip LANGUAGES CXX)
  set(LX_HIP_LANG CXX)
  set(LX_HIP_OFFLOAD "")
  foreach(arch IN LISTS LX_HIP_ARCHS)
    list(APPEND LX_HIP_OFFLOAD --offload-arch=${arch})
  endforeach()
  if(LX_HIP_ROCM_PATH)
    list(APPEND LX_HIP_OFFLOAD --rocm-path=${LX_HIP_ROCM_PATH})
  endif()
  if(LX_HIP_DEVICE_LIB_PATH)
    list(APPEND LX_HIP_OFFLOAD --rocm-device-lib-path=${LX_HIP_DEVICE_LIB_PATH})
  endif()
else()
  message(FATAL_ERROR "LX_HIP_PLATFORM must be amd or nvidia, not ${LX_HIP_PLATFORM}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB LX_HIP_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
set_source_files_properties(${LX_HIP_SOURCES} PROPERTIES LANGUAGE ${LX_HIP_LANG})

add_library(lernex_hip SHARED ${LX_HIP_SOURCES})
set_target_properties(lernex_hip PROPERTIES OUTPUT_NAME lernex_hip)
if(LX_HIP_PLATFORM STREQUAL "amd")
  target_compile_options(lernex_hip PRIVATE -x hip ${LX_HIP_OFFLOAD})
  target_link_options(lernex_hip PRIVATE ${LX_HIP_OFFLOAD})
endif()
target_include_directories(lernex_hip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(lernex_hip PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(LX_HIP_PLATFORM STREQUAL "nvidia")
  # The shim's HIP names go first, so they win over anything else on the path.
  target_include_directories(lernex_hip BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/nvidia)
# The same constexpr rule the CUDA build uses (the root CMakeLists): device lambdas call constexpr host functions.
target_compile_options(lernex_hip PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  target_link_libraries(lernex_hip PRIVATE CUDA::cudart CUDA::cublasLt)
else()
  # hipBLASLt is the dense GEMM. Its headers come from the ROCm install (or LX_HIPBLASLT_INCLUDE).
  find_path(LX_HIPBLASLT_INCLUDE hipblaslt/hipblaslt.h HINTS ${LX_HIPBLASLT_INCLUDE_HINT} /opt/rocm/include)
  if(NOT LX_HIPBLASLT_INCLUDE)
    message(FATAL_ERROR "hipblaslt/hipblaslt.h not found: set LX_HIPBLASLT_INCLUDE to its include directory")
  endif()
  target_include_directories(lernex_hip PRIVATE ${LX_HIPBLASLT_INCLUDE})
endif()
