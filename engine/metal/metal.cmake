# The Metal backend: libdromos_metal.dylib, one shared library exporting every lx_* function of
# engine/cuda/kernels.h and tunable.h. Objective-C++ host side, Metal kernels compiled at runtime from
# the embedded source (no offline Metal toolchain needed).
set(LX_METAL_DIR "${CMAKE_SOURCE_DIR}/engine/metal")
# The library source, in order: the argument structs, the shared helpers, then every kernel file.
file(GLOB LX_METAL_KERNELS CONFIGURE_DEPENDS "${LX_METAL_DIR}/kernels/*.metal")
list(REMOVE_ITEM LX_METAL_KERNELS "${LX_METAL_DIR}/kernels/common.metal")
list(SORT LX_METAL_KERNELS)
set(LX_METAL_LIBRARY_SOURCES "${LX_METAL_DIR}/kernels/args.h" "${LX_METAL_DIR}/kernels/common.metal" ${LX_METAL_KERNELS})
set(LX_METAL_EMBEDDED "${CMAKE_BINARY_DIR}/generated/lx_metal_source.cpp")
list(JOIN LX_METAL_LIBRARY_SOURCES "|" LX_METAL_SOURCE_LIST)
add_custom_command(
  OUTPUT "${LX_METAL_EMBEDDED}"
  COMMAND ${CMAKE_COMMAND} -DOUT=${LX_METAL_EMBEDDED} -DNAME=lx_metal_source
          "-DSOURCES=${LX_METAL_SOURCE_LIST}" -P "${LX_METAL_DIR}/embed.cmake"
  DEPENDS ${LX_METAL_LIBRARY_SOURCES} "${LX_METAL_DIR}/embed.cmake"
  COMMENT "Embedding the Metal kernel library"
  VERBATIM)

# The second library, for GPUs with neural accelerators (Metal 4 tensor ops): the same argument
# structs and helpers, then kernels/nax/*.metal.
file(GLOB LX_METAL_NAX_KERNELS CONFIGURE_DEPENDS "${LX_METAL_DIR}/kernels/nax/*.metal")
list(SORT LX_METAL_NAX_KERNELS)
set(LX_METAL_NAX_SOURCES "${LX_METAL_DIR}/kernels/args.h" "${LX_METAL_DIR}/kernels/common.metal" ${LX_METAL_NAX_KERNELS})
set(LX_METAL_NAX_EMBEDDED "${CMAKE_BINARY_DIR}/generated/lx_metal_nax_source.cpp")
list(JOIN LX_METAL_NAX_SOURCES "|" LX_METAL_NAX_SOURCE_LIST)
add_custom_command(
  OUTPUT "${LX_METAL_NAX_EMBEDDED}"
  COMMAND ${CMAKE_COMMAND} -DOUT=${LX_METAL_NAX_EMBEDDED} -DNAME=lx_metal_nax_source
          "-DSOURCES=${LX_METAL_NAX_SOURCE_LIST}" -P "${LX_METAL_DIR}/embed.cmake"
  DEPENDS ${LX_METAL_NAX_SOURCES} "${LX_METAL_DIR}/embed.cmake"
  COMMENT "Embedding the Metal neural accelerator library"
  VERBATIM)

file(GLOB LX_METAL_HOST CONFIGURE_DEPENDS "${LX_METAL_DIR}/*.mm")
add_library(dromos_metal SHARED ${LX_METAL_HOST} "${LX_METAL_EMBEDDED}" "${LX_METAL_NAX_EMBEDDED}")
# engine/ for quoted includes only: its dispatch/ directory would shadow the SDK's <dispatch/dispatch.h>,
# which Foundation includes.
target_compile_options(dromos_metal PRIVATE "SHELL:-iquote ${CMAKE_SOURCE_DIR}/engine"
                       $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc> -fvisibility=hidden -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(dromos_metal PUBLIC "-framework Metal" "-framework Foundation" "-framework IOKit")
set_target_properties(dromos_metal PROPERTIES OBJCXX_STANDARD 20 CXX_STANDARD 20)
