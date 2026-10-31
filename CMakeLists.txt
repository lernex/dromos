cmake_minimum_required(VERSION 3.28)
if(APPLE)
  set(APPLE_DEFAULT ON)
  set(NOT_APPLE_DEFAULT OFF)
else()
  set(APPLE_DEFAULT OFF)
  set(NOT_APPLE_DEFAULT ON)
endif()
# Backends: each is one shared library exporting every lx_* function of engine/cuda/kernels.h.
#   LX_BACKEND_CUDA   libdromos_kernels.so (the default off macOS)
#   LX_BACKEND_CPU    libdromos_cpu.so, x86 (AVX2, AVX-VNNI, AVX-512) and ARM (NEON, dot product, I8MM,
#                     BF16), no CUDA needed: cmake -DLX_BACKEND_CUDA=OFF -DLX_BACKEND_CPU=ON. Built by
#                     default next to CUDA too: the offload runtime loads it for its CPU lane.
#   LX_BACKEND_METAL  libdromos_metal.dylib, Apple silicon GPUs (the default on macOS; engine/metal)
# The model, runtime and CLI (dromos) link no backend and build with any of them: dromos opens one at run
# time (engine/dispatch).
option(LX_BACKEND_CUDA "Build the CUDA backend" ${NOT_APPLE_DEFAULT})
option(LX_BACKEND_CPU "Build the CPU backend (libdromos_cpu.so)" ${NOT_APPLE_DEFAULT})
option(LX_BACKEND_METAL "Build the Metal backend (libdromos_metal.dylib)" ${APPLE_DEFAULT})
# The release flavors (docs/RELEASE.md, scripts/release/build.sh): LX_BUNDLE_CUDA copies the toolkit's cudart and
# cuBLASLt into lib/ beside the CUDA backend, under their SONAMEs, so the tarball needs only the driver. LX_STATIC_
# LIBSTDCXX links libstdc++ and libgcc statically, so the binaries run on distributions older than the build's.
option(LX_BUNDLE_CUDA "Copy libcudart and libcublasLt into lib/ beside the CUDA backend" OFF)
option(LX_STATIC_LIBSTDCXX "Link libstdc++ and libgcc statically (Linux release builds)" OFF)

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

# The version and the release flavor, baked into `dromos --version` and into `dromos update` (docs/RELEASE.md).
#   -DLX_VERSION=X.Y.Z  the version. Without it, git describe --tags --match 'v*': vX.Y.Z is X.Y.Z, commits after a
#                       tag are X.Y.Z-dev.N+gSHA, and an untagged checkout is 0.0.0-dev+SHA.
#   -DLX_FLAVOR=NAME    the release flavor this build is packaged as, or source (the default).
set(LX_FLAVORS source linux-x86_64-cuda12 linux-x86_64-cuda13 linux-x86_64-cpu linux-aarch64-cpu darwin-arm64
    windows-x86_64-cpu windows-x86_64-cuda12 windows-x86_64-cuda13)
set(LX_FLAVOR "source" CACHE STRING "The release flavor: one of ${LX_FLAVORS}")
set_property(CACHE LX_FLAVOR PROPERTY STRINGS ${LX_FLAVORS})
if(NOT LX_FLAVOR IN_LIST LX_FLAVORS)
  message(FATAL_ERROR "LX_FLAVOR must be one of ${LX_FLAVORS} (got ${LX_FLAVOR})")
endif()
set(LX_VERSION "" CACHE STRING "The version this build reports, X.Y.Z (default: from git describe)")
find_package(Git QUIET)
set(LX_GIT_SHA "unknown")
if(NOT "$ENV{LX_GIT_SHA}" STREQUAL "")
  # A packager without git (a release container) names the commit it was given.
  set(LX_GIT_SHA "$ENV{LX_GIT_SHA}")
elseif(GIT_FOUND)
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short=7 HEAD
                  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" OUTPUT_VARIABLE LX_GIT_SHA
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE LX_GIT_RC)
  if(NOT LX_GIT_RC EQUAL 0 OR LX_GIT_SHA STREQUAL "")
    set(LX_GIT_SHA "unknown")
  endif()
endif()
if(LX_VERSION STREQUAL "")
  set(LX_VERSION "0.0.0-dev+${LX_GIT_SHA}")
  if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" describe --tags --match v* --abbrev=7
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" OUTPUT_VARIABLE LX_DESCRIBE
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE LX_DESCRIBE_RC)
    if(LX_DESCRIBE_RC EQUAL 0)
      if(LX_DESCRIBE MATCHES "^v([0-9]+\\.[0-9]+\\.[0-9]+(-rc[0-9]+)?)$")
        set(LX_VERSION "${CMAKE_MATCH_1}")
      elseif(LX_DESCRIBE MATCHES "^v([0-9]+\\.[0-9]+\\.[0-9]+(-rc[0-9]+)?)-([0-9]+)-g([0-9a-f]+)$")
        set(LX_VERSION "${CMAKE_MATCH_1}-dev.${CMAKE_MATCH_3}+g${CMAKE_MATCH_4}")
      endif()
    endif()
  endif()
endif()
if(NOT LX_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+")
  message(FATAL_ERROR "LX_VERSION must start with X.Y.Z (got ${LX_VERSION})")
endif()
message(STATUS "dromos ${LX_VERSION} (${LX_FLAVOR}, ${LX_GIT_SHA})")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/generated/lx_version.h" CONTENT [[
// Generated by CMakeLists.txt. The version and flavor this build reports.
#pragma once
#define LX_VERSION_STRING "@LX_VERSION@"
#define LX_FLAVOR_STRING "@LX_FLAVOR@"
#define LX_GIT_SHA_STRING "@LX_GIT_SHA@"
]] @ONLY)

project(dromos_engine LANGUAGES CXX)
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

# Windows (MSVC; the CUDA backend through nvcc -ccbin cl). Sources are UTF-8, __cplusplus tells the truth, and the C
# runtime keeps quiet about the POSIX names engine/platform/posix.h relies on. The C runtime is linked statically, so a
# release runs on a Windows that has no Visual C++ redistributable. Every executable and DLL lands in the build
# directory's top, where dromos.exe looks for its backends, as the install tree's bin\ does.
if(WIN32)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0A00 _CRT_SECURE_NO_WARNINGS
                          _CRT_NONSTDC_NO_WARNINGS _USE_MATH_DEFINES)
  if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
                        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor> $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
  endif()
endif()

# The version resource and manifest a Windows binary carries (engine/platform): TARGET's file name, its description,
# and APP or DLL.
function(lx_windows_binary target description kind)
  if(NOT WIN32)
    return()
  endif()
  string(REGEX MATCH "^([0-9]+)\\.([0-9]+)\\.([0-9]+)" _ "${LX_VERSION}")
  set(LX_RC_VERSION "${CMAKE_MATCH_1},${CMAKE_MATCH_2},${CMAKE_MATCH_3},0")
  set(LX_RC_DESCRIPTION "${description}")
  set(LX_RC_NAME "${target}")
  if(kind STREQUAL "APP")
    set(LX_RC_FILETYPE "VFT_APP")
    set(LX_RC_FILE "${target}.exe")
    target_sources(${target} PRIVATE "${PROJECT_SOURCE_DIR}/engine/platform/dromos.manifest")
  else()
    set(LX_RC_FILETYPE "VFT_DLL")
    set(LX_RC_FILE "${target}.dll")
  endif()
  configure_file("${PROJECT_SOURCE_DIR}/engine/platform/version.rc.in" "${CMAKE_BINARY_DIR}/generated/${target}.rc" @ONLY)
  target_sources(${target} PRIVATE "${CMAKE_BINARY_DIR}/generated/${target}.rc")
endfunction()

# engine/platform: what differs between Linux, macOS and Windows (platform.h), and on Windows the POSIX calls its C
# runtime lacks (posix.h, posix_win.cpp). The backends link it on Windows only, for those calls.
add_library(dromos_platform STATIC engine/platform/platform.cpp)
if(WIN32)
  target_sources(dromos_platform PRIVATE engine/platform/posix_win.cpp)
endif()
target_include_directories(dromos_platform PUBLIC engine)

# The static runtime for the Linux release builds: libstdc++ and libgcc are part of the binaries, so the release runs
# on any distribution from the build base's glibc up. The backends hide what they link (their interface is C).
# Each target gets them from the driver that links it. The CUDA kernels library is linked by the host C++ driver (g++),
# not nvcc, so it takes the same flags.
if(LX_STATIC_LIBSTDCXX AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(LX_STATIC_RUNTIME -static-libstdc++ -static-libgcc)
  set(LX_HIDE_LINK_SYMBOLS "LINKER:--exclude-libs,ALL")
else()
  set(LX_STATIC_RUNTIME "")
  set(LX_HIDE_LINK_SYMBOLS "")
endif()

if(LX_BACKEND_CPU)
  add_subdirectory(engine/cpu)
endif()
if(LX_BACKEND_METAL)
  include(engine/metal/metal.cmake)
endif()

# The dispatcher: the lx_* stubs the host calls, and the loader that opens one backend library at run time
# (engine/dispatch). It holds no GPU code, so it builds with or without CUDA, and dromos_core links it in place of
# libdromos_kernels. tools/gen_dispatch.py writes the stubs and the table from the ABI headers.
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_Interpreter_FOUND)
  message(FATAL_ERROR "python3 is needed to generate the backend dispatch table (tools/gen_dispatch.py)")
endif()
include(GNUInstallDirs)
# The install tree is bin/ and lib/ on every platform (docs/RELEASE.md), whatever the distribution's default
# (lib64 on the RPM bases): the CLI looks for its backends in ../lib.
set(LX_INSTALL_LIBDIR "lib")
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
add_library(dromos_dispatch STATIC engine/dispatch/dispatch.cpp "${LX_DISPATCH_GEN}")
target_include_directories(dromos_dispatch PUBLIC engine)
target_compile_definitions(dromos_dispatch PRIVATE "LX_INSTALL_LIBDIR=\"${LX_INSTALL_LIBDIR}\"")
target_link_libraries(dromos_dispatch PUBLIC dromos_platform ${CMAKE_DL_LIBS})

if(LX_BACKEND_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Line tables (-lineinfo) are most of the device code's size: the release flavors drop them
  # (scripts/release/build.sh, -DLX_LINEINFO=OFF). Dev builds keep them, for the profilers and cuda-gdb.
  option(LX_LINEINFO "Keep line tables in the CUDA device code (dev builds; the release flavors drop them)" ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  if(LX_LINEINFO)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
  endif()

  find_package(CUDAToolkit REQUIRED)

  # Every kernel, behind a C ABI (engine/cuda/kernels.h) so the tests can call them through ctypes.
  file(GLOB LX_CUDA_SOURCES CONFIGURE_DEPENDS engine/cuda/*.cu)
  # The persistent decode program (decode_program.cu and the decode_kernel*.cu that run it) is opt-in at run time
  # (LX_PROGRAM) and its kernel is the slowest file to compile by far, about 30 minutes for one architecture. The
  # release flavors leave it out (scripts/release/build.sh); a build without it decodes kernel by kernel, and says
  # so when LX_PROGRAM asks for the program.
  option(LX_DECODE_PROGRAM "Build the persistent decode program (opt-in at run time with LX_PROGRAM)" ON)
  if(NOT LX_DECODE_PROGRAM)
    list(FILTER LX_CUDA_SOURCES EXCLUDE REGEX "/decode_(program|kernel[^/]*)\\.cu$")
  endif()
  add_library(dromos_kernels SHARED ${LX_CUDA_SOURCES})
  target_include_directories(dromos_kernels PUBLIC engine)
  target_link_libraries(dromos_kernels PUBLIC CUDA::cudart CUDA::cublasLt)
  if(WIN32)
    # dromos_kernels.dll exports the kernels.h ABI (LX_API is dllexport while it builds) and takes the POSIX calls it
    # makes (the NVML loader) from engine/platform.
    target_compile_definitions(dromos_kernels PRIVATE LX_BUILDING_BACKEND)
    target_compile_options(dromos_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/Zc:__cplusplus>)
    # CCCL 3, which CUDA 13 ships, refuses MSVC's traditional preprocessor in host code; the C++ sources already use
    # the conforming one. CUDA 12 builds keep the flags they were proven with.
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
      target_compile_options(dromos_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    endif()
    target_link_libraries(dromos_kernels PRIVATE dromos_platform)
    lx_windows_binary(dromos_kernels "Dromos CUDA backend" DLL)
  endif()
  if(LX_STATIC_RUNTIME)
    target_link_options(dromos_kernels PRIVATE ${LX_STATIC_RUNTIME})
  endif()
  if(LX_HIDE_LINK_SYMBOLS)
    target_link_options(dromos_kernels PRIVATE "${LX_HIDE_LINK_SYMBOLS}")
  endif()

  # The release flavors carry the toolkit's runtime libraries in lib/, under the SONAMEs the backend asks for
  # (libcudart.so.13 from libcudart.so.13.2.75): the tarball then needs only the NVIDIA driver.
  if(LX_BUNDLE_CUDA AND WIN32)
    # Windows: the toolkit's cudart64_N.dll and cublasLt64_N.dll go beside dromos_kernels.dll in bin\, which the
    # backend loads them from first. Their license goes with them: the toolkit's EULA.txt from NVIDIA's installer, or
    # the LICENSE of the cudart redistributable archive (scripts/release/cuda-toolkit.ps1), which is the same EULA.
    # CUDA 12 keeps the DLLs in bin\ beside nvcc, and CUDA 13 in bin\x64.
    foreach(pattern IN ITEMS cudart64_*.dll cublasLt64_*.dll)
      file(GLOB lib_dll "${CUDAToolkit_BIN_DIR}/${pattern}" "${CUDAToolkit_BIN_DIR}/x64/${pattern}")
      list(LENGTH lib_dll lib_count)
      if(NOT lib_count EQUAL 1)
        message(FATAL_ERROR "LX_BUNDLE_CUDA: expected one ${pattern} in ${CUDAToolkit_BIN_DIR} or its x64 folder, "
                            "found ${lib_count}")
      endif()
      install(FILES ${lib_dll} DESTINATION bin)
    endforeach()
    set(LX_CUDA_EULA "")
    foreach(candidate IN ITEMS "${CUDAToolkit_LIBRARY_ROOT}/EULA.txt" "${CUDAToolkit_BIN_DIR}/../EULA.txt"
                               "${CUDAToolkit_BIN_DIR}/../licenses/cuda_cudart-LICENSE.txt")
      if(NOT LX_CUDA_EULA AND EXISTS "${candidate}")
        set(LX_CUDA_EULA "${candidate}")
      endif()
    endforeach()
    if(NOT LX_CUDA_EULA)
      message(FATAL_ERROR "LX_BUNDLE_CUDA: the toolkit has no EULA.txt (or redistributable LICENSE) to ship beside "
                          "its libraries")
    endif()
    install(FILES "${LX_CUDA_EULA}" DESTINATION share/dromos/licenses RENAME NVIDIA-CUDA-EULA.txt)
  elseif(LX_BUNDLE_CUDA)
    foreach(target IN ITEMS CUDA::cudart CUDA::cublasLt)
      get_target_property(lib_path ${target} IMPORTED_LOCATION)
      if(NOT lib_path)
        get_target_property(lib_path ${target} IMPORTED_LOCATION_RELEASE)
      endif()
      if(NOT lib_path)
        message(FATAL_ERROR "LX_BUNDLE_CUDA: no file for ${target}")
      endif()
      file(REAL_PATH "${lib_path}" lib_real)
      get_filename_component(lib_name "${lib_real}" NAME)
      string(REGEX REPLACE "^(lib[A-Za-z]+\\.so\\.[0-9]+)\\..*$" "\\1" lib_soname "${lib_name}")
      if(lib_soname STREQUAL lib_name)
        message(FATAL_ERROR "LX_BUNDLE_CUDA: ${lib_name} is not a versioned shared library")
      endif()
      install(FILES "${lib_real}" DESTINATION "${LX_INSTALL_LIBDIR}" RENAME "${lib_soname}")
    endforeach()
    # NVIDIA's license for the libraries goes with them (NOTICE): the toolkit's EULA.txt.
    set(LX_CUDA_EULA "")
    foreach(dir IN ITEMS "${CUDAToolkit_LIBRARY_ROOT}" "${CUDAToolkit_BIN_DIR}/..")
      if(NOT LX_CUDA_EULA AND EXISTS "${dir}/EULA.txt")
        set(LX_CUDA_EULA "${dir}/EULA.txt")
      endif()
    endforeach()
    if(NOT LX_CUDA_EULA)
      message(FATAL_ERROR "LX_BUNDLE_CUDA: the toolkit has no EULA.txt to ship beside its libraries "
                          "(install its documentation package, cuda-documentation-MAJOR-MINOR)")
    endif()
    install(FILES "${LX_CUDA_EULA}" DESTINATION share/dromos/licenses RENAME NVIDIA-CUDA-EULA.txt)
  endif()
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
  add_library(dromos_core STATIC ${LX_CORE_SOURCES})
  target_include_directories(dromos_core PUBLIC engine third_party third_party/httplib)
  find_package(Threads REQUIRED)
  target_link_libraries(dromos_core PUBLIC dromos_dispatch Threads::Threads)
  if(WIN32)
    # The OpenAI server's sockets (cpp-httplib) and the downloads (engine/core/http.cpp, WinHTTP).
    target_link_libraries(dromos_core PUBLIC ws2_32 winhttp)
  endif()
  # The architectures this build has code for, for the device report. A header, not a definition: nvcc
  # splits a -D value on its commas.
  file(WRITE "${CMAKE_BINARY_DIR}/generated/lx_build_info.h"
       "#pragma once\n#define LX_BUILT_ARCHS ${LX_BUILT_ARCHS_CSV}\n")
  target_include_directories(dromos_core PUBLIC "${CMAKE_BINARY_DIR}/generated")
  # The tuning tables shipped in the repository (tuning/), found next to the source tree.
  target_compile_definitions(dromos_core PRIVATE "LX_TUNING_DIR=\"${CMAKE_SOURCE_DIR}/tuning\"")
endif()

file(GLOB LX_CLI_SOURCES CONFIGURE_DEPENDS cli/*.cpp)
if(LX_CLI_SOURCES AND TARGET dromos_core)
  add_executable(dromos ${LX_CLI_SOURCES})
  target_compile_definitions(dromos PRIVATE "LX_CORPUS_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/tools/rivals/corpus.txt\"")
  target_include_directories(dromos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(dromos PRIVATE dromos_core)
  lx_windows_binary(dromos "Dromos" APP)
  if(LX_STATIC_RUNTIME)
    target_link_options(dromos PRIVATE ${LX_STATIC_RUNTIME})
  endif()
  # The backend libraries are opened at run time from next to the executable, so the CLI builds them too.
  foreach(backend dromos_kernels dromos_cpu dromos_metal)
    if(TARGET ${backend})
      add_dependencies(dromos ${backend})
    endif()
  endforeach()
endif()

# The C++ unit tests (doctest): host-side policy, parsing and protocol code, run on the server.
file(GLOB LX_TEST_SOURCES CONFIGURE_DEPENDS tests/cpp/*.cpp)
if(LX_TEST_SOURCES AND TARGET dromos_core)
  add_executable(dromos_tests ${LX_TEST_SOURCES})
  target_include_directories(dromos_tests PRIVATE engine third_party)
  target_link_libraries(dromos_tests PRIVATE dromos_core)
endif()

# GPU checks: the engine end to end on a real model (run by hand on the server, see tests/gpu).
file(GLOB LX_GPU_CHECK_SOURCES CONFIGURE_DEPENDS tests/gpu/*.cpp)
if(LX_GPU_CHECK_SOURCES AND TARGET dromos_core)
  add_executable(dromos_gpu_check ${LX_GPU_CHECK_SOURCES})
  target_link_libraries(dromos_gpu_check PRIVATE dromos_core)
endif()

# The text stack's tests: tokenizer, chat template and output parser. No CUDA; the goldens come from
# tools/text/*.py and are read through LX_TEXT_GOLDEN (see tests/text).
file(GLOB LX_TEXT_TEST_SOURCES CONFIGURE_DEPENDS tests/text/*.cpp)
file(GLOB LX_TEXT_SOURCES CONFIGURE_DEPENDS engine/text/*.cpp)
add_executable(lx_text_tests ${LX_TEXT_TEST_SOURCES} ${LX_TEXT_SOURCES})
target_include_directories(lx_text_tests PRIVATE engine third_party)

# Unit tests for the C++ modules (plan, device report, tuner), with doctest (third_party/doctest).
file(GLOB LX_UNIT_SOURCES CONFIGURE_DEPENDS tests/unit/*.cpp)
if(LX_UNIT_SOURCES AND TARGET dromos_core)
  add_executable(dromos_unit ${LX_UNIT_SOURCES})
  target_include_directories(dromos_unit PRIVATE third_party/doctest)
  target_link_libraries(dromos_unit PRIVATE dromos_core)
endif()

# The prefix cache, end to end on the model (run on the GPU server, see tests/gpu/prefix).
if(LX_SLOT_STATE AND TARGET dromos_core)
  add_executable(dromos_prefix_check tests/gpu/prefix/prefix_check.cpp)
  target_link_libraries(dromos_prefix_check PRIVATE dromos_core)
endif()

# The install tree (docs/RELEASE.md, `cmake --install BUILD --prefix DIR`): bin/dromos, lib/ with the backends (and the
# toolkit's libraries and their EULA, LX_BUNDLE_CUDA), share/dromos/tuning, share/dromos/bench, LICENSE, NOTICE and the
# third-party licenses in share/dromos/licenses. The packager writes share/dromos/channel.
# Relocatable: the CLI finds its backends in ../lib, and each backend finds the libraries beside it ($ORIGIN). The
# build tree keeps CMake's own rpath.
if(APPLE)
  set(LX_RPATH_CLI "@loader_path/../lib")
  set(LX_RPATH_LIB "@loader_path")
else()
  set(LX_RPATH_CLI "$ORIGIN/../lib")
  set(LX_RPATH_LIB "$ORIGIN")
endif()
if(TARGET dromos)
  set_target_properties(dromos PROPERTIES INSTALL_RPATH "${LX_RPATH_CLI}")
  install(TARGETS dromos RUNTIME DESTINATION bin)
endif()
foreach(backend dromos_kernels dromos_cpu dromos_metal)
  if(TARGET ${backend} AND WIN32)
    # Windows: the backend DLLs sit beside dromos.exe in bin\ (where Windows looks for a program's DLLs), and their
    # import libraries are not part of the install.
    install(FILES $<TARGET_FILE:${backend}> DESTINATION bin)
  elseif(TARGET ${backend})
    set_target_properties(${backend} PROPERTIES INSTALL_RPATH "${LX_RPATH_LIB}")
    install(TARGETS ${backend} LIBRARY DESTINATION "${LX_INSTALL_LIBDIR}" RUNTIME DESTINATION "${LX_INSTALL_LIBDIR}")
  endif()
endforeach()
install(DIRECTORY tuning/ DESTINATION share/dromos/tuning FILES_MATCHING PATTERN "*.json")
# dromos bench cuts its prompts from this corpus (cli/bench.cpp), so an install benches the text a build tree does.
install(FILES tools/rivals/corpus.txt DESTINATION share/dromos/bench)
install(FILES LICENSE NOTICE DESTINATION .)
install(DIRECTORY licenses/ DESTINATION share/dromos/licenses)
