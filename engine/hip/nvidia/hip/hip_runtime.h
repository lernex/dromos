// NVIDIA test shim for the HIP port (engine/hip). It supplies the HIP names the translated sources
// use, over the CUDA runtime, so the same sources build and run on an NVIDIA GPU. It is not HIP: AMD's
// HIP-on-NVIDIA headers were removed after ROCm 5.7 and do not build against CUDA 13, so the port is
// tested through this shim and compiled for AMD with the ROCm toolchain. Only the names the port uses
// are here, so a name the port starts to use fails to build until it is added.
#pragma once

#define LX_NVIDIA_SHIM 1

#include <cuda_runtime.h>

typedef cudaError_t hipError_t;
typedef cudaStream_t hipStream_t;
typedef cudaEvent_t hipEvent_t;
typedef cudaGraph_t hipGraph_t;
typedef cudaGraphExec_t hipGraphExec_t;
typedef cudaMemcpyKind hipMemcpyKind;
typedef cudaStreamCaptureMode hipStreamCaptureMode;
typedef cudaFuncAttribute hipFuncAttribute;
typedef cudaDeviceAttr hipDeviceAttribute_t;
typedef cudaDataType hipDataType;

#define hipSuccess cudaSuccess
#define hipErrorInvalidValue cudaErrorInvalidValue
#define hipErrorNotSupported cudaErrorNotSupported
#define hipErrorInvalidDevice cudaErrorInvalidDevice
#define hipErrorUnknown cudaErrorUnknown
#define hipErrorOutOfMemory cudaErrorMemoryAllocation

#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define hipStreamNonBlocking cudaStreamNonBlocking
#define hipEventDisableTiming cudaEventDisableTiming
#define hipStreamCaptureModeThreadLocal cudaStreamCaptureModeThreadLocal
#define hipHostMallocDefault cudaHostAllocDefault
#define hipFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize
#define HIP_R_16BF CUDA_R_16BF
#define HIP_R_32F CUDA_R_32F

#define hipGetLastError cudaGetLastError
#define hipGetDevice cudaGetDevice
#define hipSetDevice cudaSetDevice
#define hipGetDeviceCount cudaGetDeviceCount
#define hipDeviceGetAttribute cudaDeviceGetAttribute
#define hipFuncSetAttribute cudaFuncSetAttribute
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemset cudaMemset
#define hipMemsetAsync cudaMemsetAsync
#define hipMemcpy cudaMemcpy
#define hipMemcpyAsync cudaMemcpyAsync
#define hipHostMalloc cudaHostAlloc
#define hipFreeHost cudaFreeHost
#define hipMallocAsync cudaMallocAsync
#define hipFreeAsync cudaFreeAsync
#define hipStreamCreateWithFlags cudaStreamCreateWithFlags
#define hipStreamDestroy cudaStreamDestroy
#define hipStreamSynchronize cudaStreamSynchronize
#define hipStreamBeginCapture cudaStreamBeginCapture
#define hipStreamEndCapture cudaStreamEndCapture
#define hipDevAttrComputeCapabilityMajor cudaDevAttrComputeCapabilityMajor
#define hipDevAttrComputeCapabilityMinor cudaDevAttrComputeCapabilityMinor

// HIP's instantiate takes an error node and a log; CUDA's takes the flags only.
static inline hipError_t hipGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph, void* error_node, char* log,
                                             size_t log_size) {
  (void)error_node;
  (void)log;
  (void)log_size;
  return cudaGraphInstantiate(exec, graph, 0);
}
#define hipGraphLaunch cudaGraphLaunch
#define hipGraphExecDestroy cudaGraphExecDestroy
#define hipGraphDestroy cudaGraphDestroy
#define hipEventCreateWithFlags cudaEventCreateWithFlags
#define hipEventDestroy cudaEventDestroy
#define hipEventRecord cudaEventRecord
#define hipEventSynchronize cudaEventSynchronize

// Names the translator renames onto HIP's own spellings (translate.py RENAMES).
#define hipDeviceAttributeComputeCapabilityMajor cudaDevAttrComputeCapabilityMajor
#define hipDeviceAttributeComputeCapabilityMinor cudaDevAttrComputeCapabilityMinor
#define hipDeviceAttributeMultiprocessorCount cudaDevAttrMultiProcessorCount
#define hipDeviceAttributeL2CacheSize cudaDevAttrL2CacheSize
#define hipDeviceAttributeMemoryClockRate cudaDevAttrMemoryClockRate
#define hipDeviceAttributeMemoryBusWidth cudaDevAttrGlobalMemoryBusWidth
#define hipOccupancyMaxActiveBlocksPerMultiprocessor cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define hipMemcpyDefault cudaMemcpyDefault
#define hipRuntimeGetVersion cudaRuntimeGetVersion
#define hipDriverGetVersion cudaDriverGetVersion
#define hipGetDeviceProperties cudaGetDeviceProperties
#define hipPointerGetAttributes cudaPointerGetAttributes
#define hipMemoryTypeHost cudaMemoryTypeHost
#define hipGetErrorString cudaGetErrorString
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipEventElapsedTime cudaEventElapsedTime
#define hipEventCreate cudaEventCreate
typedef cudaDeviceProp hipDeviceProp_t;
typedef cudaPointerAttributes hipPointerAttribute_t;
typedef cudaDeviceAttr hipDeviceAttribute_t;
#define hipHostAlloc cudaHostAlloc
#define hipHostAllocPortable cudaHostAllocPortable
#define hipHostAllocMapped cudaHostAllocMapped
#define hipHostRegister cudaHostRegister
#define hipHostRegisterPortable cudaHostRegisterPortable
#define hipHostRegisterMapped cudaHostRegisterMapped
#define hipHostUnregister cudaHostUnregister
#define hipHostGetDevicePointer cudaHostGetDevicePointer
#define hipStreamWaitEvent cudaStreamWaitEvent
#define hipMemGetInfo cudaMemGetInfo
#define hipSharedmemCarveoutMaxShared cudaSharedmemCarveoutMaxShared
#define hipFuncAttributePreferredSharedMemoryCarveout cudaFuncAttributePreferredSharedMemoryCarveout
