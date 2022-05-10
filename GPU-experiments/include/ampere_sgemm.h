/*
 * @Author: Li.Yinghan https://github.com/Yinghan-Li
 * @Date: 2022-03-14 09:35:55
 * @LastEditTime: 2022-03-19 10:31:14
 * @LastEditors: ConvolutedDog
 * @Description: In User Settings Edit
 * @FilePath: /GPU-experiments/include/ampere_sgemm.h
 */
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cmath>

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

void random_init(float *data, size_t size);

bool check(const float *A,
           const float *B,
           const float *C,
           int m, int n, int k);

__device__ __forceinline__
uint32_t smem_u32addr(const void *smem_ptr) {
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
         : "=r"(addr)
         : "l"(smem_ptr)
    );

    return addr;
}

__device__ __forceinline__
void ldg32_nc(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void ldg32_nc_0(float &reg, const void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @!p mov.b32 %0, 0;\n"
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
        " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
        " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
        : "=f"(reg)
        : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void stg32(const float &reg, void *ptr, bool guard) {
    asm volatile (
        "{.reg .pred p;\n"
        " setp.ne.b32 p, %2, 0;\n"
        " @p st.global.f32 [%0], %1;}\n"
        : : "l"(ptr), "f"(reg), "r"((int)guard)
    );
}

__device__ __forceinline__
void lds128(float &reg0, float &reg1,
            float &reg2, float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
        : "r"(addr)
    );
}

__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr) {
    asm volatile (
        "st.shared.f32 [%0], %1;\n"
        : : "r"(addr), "f"(reg)
    );
}

__device__ __forceinline__
void sts128(const float &reg0, const float &reg1,
            const float &reg2, const float &reg3,
            const uint32_t &addr) {
    asm volatile (
        "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
        : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
    );
}

struct StgFrag {
    float data[4][4];

    __device__ __forceinline__
    StgFrag(const float (&C_frag)[8][8], int tile_x, int tile_y) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                data[i][j] = C_frag[tile_y * 4 + i][tile_x * 4 + j];
            }
        }
    }
};

__device__ __noinline__
void C_tile_wb(StgFrag C_frag,
               float *C_stg_ptr,
               const float *C_lds_ptr,
               uint32_t C_sts_addr,
               uint32_t m,
               uint32_t n,
               uint32_t m_idx,
               uint32_t n_idx);

/*
 * matrix A, B and C: row-major
 *
 * mma block:
 * thread block tile: m128n128k8
 * warp tile: m32n64k8
 * thread tile: m8n8k8
 * thread fragment:
 *     matrixA: 8x1 FP32
 *     matrixB: 1x8 FP32
 *
 * ----------------------------------------------------------------
 * thread block tile map:
 *
 *                                128
 *                    --|---------------------|
 *             B_tile  8|                     |
 *                    --|---------------------|
 *
 *  A_tile   | 8 |      |    64    |
 *         --|---|    --|----------|----------|
 *           |   |    32|  warp_0  |  warp_1  |
 *           |   |    --|----------|----------|
 *           |   |      |  warp_2  |  warp_3  |
 *        128|   |      |----------|----------|
 *           |   |      |  warp_4  |  warp_5  |
 *           |   |      |----------|----------|
 *           |   |      |  warp_6  |  warp_7  |
 *         --|---|      |----------|----------|
 * warp_0 is warp_0_tile ???
 *      warp_0 consists of 4 warps???
 * thread tile: 8 * 8
 * warp size: 4 * 8
 * 
 * 
 * ----------------------------------------------------------------
 * warp tile map:
 *
 * 'z' thread map to avoid LDS.128 shared memory broadcast limitation.
 *
 *              |              32               ||
 *     B_frag --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 *             1|///|   |   |   |   |   |   |   ||///|   |   |   |   |   |   |   |
 *            --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 * A_frag_left  | 4 |                           || 4                                          A_frag_right
 *    | 1 |                                     ||          mma_tid_y-->                         | 1 |  
 *  --|---|--   |---|---|---|---|---|---|---|---||---|---------------------------|             --|---|--
 *    |///|4    |t0 |t2 |t4 |t6 |t8 |t10|t12|t14||t0 |                           |               |///|4
 *    |---|--   |---|---|---|---|---|---|---|---||---|                           |               |---|-- 
 *    |   |     |t1 |t3 |t5 |t7 |t9 |t11|t13|t15||                               |               |   |  
 *  16|---|     |---|---|---|---|---|---|---|---||                               | mma_tid_x   16|---| 
 *    |   |     |t16|t18|t20|t22|t24|t26|t28|t30||                               |     |         |   |
 *    |---|     |---|---|---|---|---|---|---|---||                               |    \|/        |---|   
 *    |   |     |t17|t19|t21|t23|t25|t27|t29|t31||                               |               |   |
 *  ==|===|=====|===|===|===|===|===|===|===|===||===|============================             ==|===|==
 *    |///|4    |t0 |                           ||t0 |                           |               |///|4
 *    |---|     |---|                           ||---|                           |               |---|
 *    |   |     |                               ||                               |               |   |
 *    |---|     |                               ||                               |               |---|
 *    |   |     |                               ||                               |               |   |
 *    |---|     |                               ||                               |               |---|
 *    |   |     |                               ||                               |               |   |
 *    |---|     |-------------------------------||-------------------------------|               |---|
 * 
 * 4 t0 belongs to the same thread
 * 
 * thread idx
 *              |              32               ||
 *     B_frag --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 *             1|///|   |   |   |   |   |   |   ||///|   |   |   |   |   |   |   |
 *            --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 * A_frag       | 4 |                           ||
 *    | 1 |                                     ||          mma_ id_y-->
 *  --|---|--   |---|---|---|---|---|---|---|---||---|---------------------------|
 *    |///|4    | 0 |   |   |   |   |   |   | 7 || 32|                           |
 *    |---|--   |---|---|---|---|---|---|---|---||---|                           |
 *    |   |     |   |   |   |   |   |   |   |   ||                               |
 *  16|---|     |---|---|---|---|---|---|---|---||                               | mma_ id_x
 *    |   |     |   |   |   |   |   |   |   |   ||                               |     |
 *    |---|     |---|---|---|---|---|---|---|---||                               |    \|/
 *    |   |     |   |   |   |   |   |   |   | 31||                           | 63|
 *  ==|===|=====|===|===|===|===|===|===|===|===||===|============================
 *    |///|     | 64|                           || 96|                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                           | 95||                           |127|
 *    |---|     |-------------------------------||-------------------------------|
 * 
 */


// launch_bounds(maxThreadsPerBlock, minBlocksPerMultiprocessor)
// 256 threads are splited to 8 warps, with 32 threads per warp
__global__ __launch_bounds__(256, 2)
void sgemm_128x128x8_kernel(const float *A,
                            const float *B,
                            float *C,
                            uint32_t m,
                            uint32_t n,
                            uint32_t k,
                            uint32_t A_ldg_step,    // k * sizeof(float)
                            uint32_t B_ldg_step);
