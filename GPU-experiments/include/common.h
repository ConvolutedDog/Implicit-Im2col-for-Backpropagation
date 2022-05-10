/*
 * @Author: Yangjie Zhou. https://zenodo.org/record/5535284
 * @Date: 2022-03-11 20:26:34
 * @LastEditTime: 2022-03-19 10:31:43
 * @LastEditors: ConvolutedDog
 * @Description: In User Settings Edit
 * @FilePath: /GPU-experiments/include/common.h
 */

#ifndef COMMON_H
#define COMMON_H

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cuda.h>
#include <cuda_fp16.h>
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <string.h>
#include <cuda_runtime_api.h>
#include <memory.h>
#include <time.h>
#include <random>      
#include "/usr/include/cudnn.h"
#include <mma.h>
#include "helper_cuda.h"
#include <helper_functions.h>

#define _HOST_DEVICE_ __forceinline__ __device__ __host__
#define _DEVICE_ __forceinline__ __device__
#define _PRAGMA_UNROLL_
#define _PRAGMA_NO_UNROLL_
#define _GEMM_LOOP_



#define DTYPE __half
// #define SKEW_HALF 16
#define SKEW_HALF 8

// #define KERNEL_MUL_ALPHA 
// #define KERNEL_MUL_BETA 

#define DOUBLE_BUFFER
// #define UPDATE_SHARE_MEM

#define WARP_ACCESS_M 32

// #define DEBUG


#define CUDA_CALL(f){ \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    CUDA_CALL - Error occurred: " << err << ", error line: " \
        << __LINE__ << " of file " << __FILE__ << ". " << cudaGetErrorString(err) \
        << std::endl; \
    std::exit(EXIT_FAILURE); \
  } \
} 

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    CUDNN_CALL-Error occurred: " << err << ", error line: " \
        << __LINE__ << " of file " << __FILE__ << ". " << cudnnGetErrorString(err) \
        << std::endl; \
    std::exit(EXIT_FAILURE); \
  } \
}


/// C++ exception wrapper for CUDA \p cudaError_t
class cuda_exception : public std::exception {
 public:
  /// Constructor
  cuda_exception(const char* msg = "", cudaError_t err = cudaErrorUnknown) : msg(msg), err(err) {}

  /// Returns the underlying CUDA \p cudaError_t
  cudaError_t cudaError() const { return err; }

 protected:
  /// Explanatory string
  const char* msg;

  /// Underlying CUDA \p cudaError_t
  cudaError_t err;
};


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif