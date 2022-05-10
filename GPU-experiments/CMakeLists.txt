cmake_minimum_required(VERSION 2.8)
project(GPU-experiments)

find_package(CUDA QUIET REQUIRED)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_80,code=sm_80 -arch=sm_80 -DCUDA_ARCH_SM=80 --ptxas-options=-v -lcublas
    )


include_directories(
    include
    /usr/include/
    /usr/local/gcc-9.3.0/include/
    /usr/local/cuda/include/
    /usr/local/cuda/samples/common/inc/
    )

cuda_add_executable(
    GPUImplicit
    GPUImplicit.cu
    src/helper.cpp
    src/ampere_sgemm.cu
    src/cudnn.cu
    src/preprocess.cu
    src/nopreprocess.cu
    )


target_link_libraries(
    GPUImplicit
    cudnn
    )
