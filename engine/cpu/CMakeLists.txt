# libdromos_cpu.so: the CPU backend. Every lx_* function of engine/cuda/kernels.h, no CUDA.
#
# The ops (validation, scratch, threading, the small kernels) are compiled once for the baseline of the
# architecture. The hot kernels (kern/isa.cpp: the dot-product tiles of the GEMV, GEMM and experts, the
# attention, the scan) are compiled once per ISA level, each in its own namespace with its own flags, and
# the library picks one table at run time (isa.cpp). Nothing in a per-ISA object has external linkage
# except its table, so no instruction a CPU lacks can leak into code it runs.
include(CheckCXXCompilerFlag)

set(LX_CPU_SOURCES
  attention.cpp bf16x.cpp elementwise.cpp gemm.cpp glue.cpp isa.cpp mamba.cpp memory.cpp moe.cpp offload.cpp pool.cpp quant.cpp
  runtime.cpp sampler.cpp spec.cpp tunable.cpp)

# One object library per ISA level: name, level code (tunable.h's lx_arch on the CPU), flags.
set(LX_CPU_LEVELS)
macro(lx_cpu_level name code)
  set(flags ${ARGN})
  set(ok TRUE)
  foreach(f IN LISTS flags)
    string(MAKE_C_IDENTIFIER "LX_HAS_FLAG${f}" var)
    check_cxx_compiler_flag("${f}" ${var})
    if(NOT ${var})
      set(ok FALSE)
    endif()
  endforeach()
  if(ok)
    add_library(lx_cpu_${name} OBJECT kern/isa.cpp)
    target_include_directories(lx_cpu_${name} PRIVATE ${PROJECT_SOURCE_DIR}/engine)
    target_compile_options(lx_cpu_${name} PRIVATE ${flags} -fvisibility=hidden)
    target_compile_definitions(lx_cpu_${name} PRIVATE LX_ISA=${name} LX_ISA_CODE=${code})
    list(APPEND LX_CPU_LEVELS ${name})
  else()
    message(STATUS "dromos_cpu: compiler lacks flags for ${name}, level not built")
  endif()
endmacro()

lx_cpu_level(generic 1)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(LX_X86 -mavx2 -mfma -mf16c -mbmi2)
  lx_cpu_level(avx2 10 ${LX_X86})
  lx_cpu_level(avxvnni 11 ${LX_X86} -mavxvnni)
  set(LX_X512 ${LX_X86} -mavx512f -mavx512bw -mavx512vl -mavx512dq)
  lx_cpu_level(avx512 20 ${LX_X512})
  lx_cpu_level(avx512vnni 21 ${LX_X512} -mavx512vnni)
  lx_cpu_level(avx512bf16 22 ${LX_X512} -mavx512vnni -mavx512bf16)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  lx_cpu_level(neon 50 -march=armv8-a)
  lx_cpu_level(neondot 51 -march=armv8.2-a+dotprod+fp16)
  lx_cpu_level(neoni8mm 52 -march=armv8.6-a+dotprod+fp16+i8mm+bf16)
endif()

set(LX_CPU_OBJECTS)
set(LX_CPU_LEVEL_DEFS)
foreach(level IN LISTS LX_CPU_LEVELS)
  list(APPEND LX_CPU_OBJECTS $<TARGET_OBJECTS:lx_cpu_${level}>)
  string(TOUPPER ${level} up)
  list(APPEND LX_CPU_LEVEL_DEFS LX_CPU_HAVE_${up}=1)
endforeach()

add_library(dromos_cpu SHARED ${LX_CPU_SOURCES} ${LX_CPU_OBJECTS})
# libdromos_cpu.so on every platform, so the tests and the model find it by one name.
# Beside libdromos_kernels.so, where the offload runtime looks for its CPU lane.
set_target_properties(dromos_cpu PROPERTIES SUFFIX ".so" LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_include_directories(dromos_cpu PUBLIC ${PROJECT_SOURCE_DIR}/engine)
target_compile_options(dromos_cpu PRIVATE -fvisibility=hidden)
target_compile_definitions(dromos_cpu PRIVATE ${LX_CPU_LEVEL_DEFS})
find_package(Threads REQUIRED)
target_link_libraries(dromos_cpu PRIVATE Threads::Threads)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(dromos_cpu PRIVATE -Wl,--no-undefined)
endif()
if(LX_STATIC_RUNTIME)
  target_link_options(dromos_cpu PRIVATE ${LX_STATIC_RUNTIME})
endif()
if(LX_HIDE_LINK_SYMBOLS)
  target_link_options(dromos_cpu PRIVATE "${LX_HIDE_LINK_SYMBOLS}")
endif()

# A native check of every level the machine can run against the generic level, without PyTorch
# (the Mac has no PyTorch): tests/cpu/isa_check.cpp.
add_executable(lx_cpu_isa_check ${PROJECT_SOURCE_DIR}/tests/cpu/isa_check.cpp)
target_link_libraries(lx_cpu_isa_check PRIVATE dromos_cpu)

# Memory bandwidth and kernel speed on this machine: bench/cpu_bench.cpp.
add_executable(lx_cpu_bench ${PROJECT_SOURCE_DIR}/bench/cpu_bench.cpp)
target_link_libraries(lx_cpu_bench PRIVATE dromos_cpu Threads::Threads)
