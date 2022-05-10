// Copyright (c) 2021, Yangjie Zhou. (https://zenodo.org/record/5535284)

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * @Author: Yangjie Zhou. https://zenodo.org/record/5535284
 * @Date: 2022-03-11 20:26:34
 * @LastEditTime: 2022-03-19 10:31:43
 * @LastEditors: ConvolutedDog
 * @Description: In User Settings Edit
 * @FilePath: /GPU-experiments/include/helper.h
 */

#include "../include/common.h"
#include "../include/helper.h"
#include "../include/config.h"

Result test_cudnn(Options &options) {
    Result result;
    int in_n, in_h, in_w, in_c, filt_d, filt_h, filt_w, filt_c;

    // update_param(Option &options);
    in_n = options.input.get_n();
    in_c = options.input.get_c();
    in_h = options.input.get_h();
    in_w = options.input.get_w();
    
    filt_d = options.filter.get_n();
    filt_c = options.filter.get_c();
    filt_h = options.filter.get_h();
    filt_w = options.filter.get_w();
    
    const int pad_h = options.padding.get_n();
    const int pad_w = options.padding.get_h();
    const int str_h = options.conv_stride.get_M();
    const int str_w = options.conv_stride.get_N();
    const int dil_h = options.dilation.get_M();
    const int dil_w = options.dilation.get_N();

    //#######################FORWARD START########################
    

    Tensor4d<DTypeInput> tensor_a(options.input);
    Tensor4d<DTypeFilter> tensor_b(options.filter);
    
    // tensor_a.get_host_data_by_idx(0);

    
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t in_desc;
    cudnnFilterDescriptor_t filt_desc;
    cudnnTensorDescriptor_t out_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CALL(cudnnCreate(&cudnn));
    

    CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
    

    CUDNN_CALL(cudnnSetTensor4dDescriptor(
          in_desc, CUDNN_TENSOR_NCHW, CUDNN_DType_Input,
          in_n, in_c, in_h, in_w));
          
    
    CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DType_Filter, CUDNN_TENSOR_NCHW,
        filt_d, filt_c, filt_h, filt_w)); 

    
    
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    
    CUDNN_CALL(cudnnSetConvolutionMathType(conv_desc, CUDNN_MATH_TYPE));

    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION, CUDNN_DType_OP));
    


    tensor_a.init_host_data_random(); 
    tensor_b.init_host_data_random(); 

    int out_n;
    int out_c;
    int out_h;
    int out_w;
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

    int output_tensor_size = out_n * out_c * out_h * out_w;

    Tensor4d<DTypeOutput> tensor_c(output_tensor_size);

    
    result.fmas = output_tensor_size * int64_t(filt_h * filt_w * filt_c);
    CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DType_Output,
        out_n, out_c, out_h, out_w));

    cudnnConvolutionFwdAlgo_t algo;
    // algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    // algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    algo = (cudnnConvolutionFwdAlgo_t)CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
    
    size_t ws_size;
    CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
          cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));
    
    
    void *ws_data;
    CUDA_CALL(cudaMalloc(&ws_data, ws_size)); 
    
    DtypeBias alpha = 1.f;
    DtypeBias beta = 0.f;
    
    cudaEvent_t events[2];
    float all_time = 0;
    for (auto & event : events) {
      cudaEventCreate(&event);
    }
    
    
    void *in_data = tensor_a.device_data();
    void *filt_data = tensor_b.device_data();
    void *out_data = tensor_c.device_data();

    // warm up
    CUDNN_CALL(cudnnConvolutionForward(
        cudnn,
        &alpha, in_desc, in_data, filt_desc, filt_data,
        conv_desc, algo, ws_data, ws_size,
        &beta, out_desc, out_data));

    cudaEventRecord(events[0]);
    
    for (int iteration = 0; iteration < options.iterations ; ++iteration) {
        CUDNN_CALL(cudnnConvolutionForward(
            cudnn,
            &alpha, in_desc, in_data, filt_desc, filt_data,
            conv_desc, algo, ws_data, ws_size,
            &beta, out_desc, out_data));
        
    }

    cudaEventRecord(events[1]);

    cudaEventSynchronize(events[1]);
    cudaEventElapsedTime(&all_time, events[0], events[1]);
    
    result.runtime_ms = double(all_time) / double(options.iterations);
    result.gflops = 2.0 * result.fmas / double(1.0e9) / (result.runtime_ms / 1000);
    result.workspace = ws_size;

    //#######################FORWARD END##########################

    //#######################LOSS START###########################
    // out_data
    //
    cudnnTensorDescriptor_t loss_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&loss_desc));
    cudnnTensorDescriptor_t din_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&din_desc));

    CUDNN_CALL(cudnnSetTensor4dDescriptor(
          loss_desc, CUDNN_TENSOR_NCHW, CUDNN_DType_Input,
          out_n, out_c, out_h, out_w));

    CUDNN_CALL(cudnnSetTensor4dDescriptor(
          din_desc, CUDNN_TENSOR_NCHW, CUDNN_DType_Input,
          in_n, in_c, in_h, in_w));

    Tensor4d<DTypeOutput> tensor_c_loss(output_tensor_size);
    tensor_c_loss.init_host_data_random(); 
    void *loss_data = tensor_c_loss.device_data();

    Tensor4d<DTypeOutput> tensor_a_loss(options.input);
    // tensor_a_loss.init_host_data_random(); 
    void *ddata = tensor_a_loss.device_data();

    

    cudnnConvolutionBwdDataAlgo_t algo_loss;
    // algo_loss = (cudnnConvolutionBwdDataAlgo_t)CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    algo_loss = (cudnnConvolutionBwdDataAlgo_t)CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    
    // CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
    //       cudnn, in_desc, filt_desc, conv_desc, out_desc, algo_loss, &ws_size));

    size_t ws_size_loss;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
          cudnn, filt_desc, loss_desc, conv_desc, din_desc, algo_loss, &ws_size_loss));

    void *ws_data_loss;
    CUDA_CALL(cudaMalloc(&ws_data_loss, ws_size_loss));

    // warm up
    CUDNN_CALL(cudnnConvolutionBackwardData(
        cudnn, &alpha, filt_desc, filt_data, loss_desc, loss_data, conv_desc, algo_loss, 
        ws_data_loss, ws_size_loss, &beta, din_desc, ddata));

    cudaEventRecord(events[0]);

    for (int iteration = 0; iteration < options.iterations ; ++iteration) {
        CUDNN_CALL(cudnnConvolutionBackwardData(
            cudnn, &alpha, filt_desc, filt_data, loss_desc, loss_data, conv_desc, algo_loss, 
            ws_data_loss, ws_size_loss, &beta, din_desc, ddata));
    }

    cudaEventRecord(events[1]);
    cudaEventSynchronize(events[1]);
    cudaEventElapsedTime(&all_time, events[0], events[1]);

    printf("Loss: %lf ms.\n", float(all_time)/float(options.iterations));
    //#######################LOSS END#############################

    CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(loss_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(din_desc));
    CUDNN_CALL(cudnnDestroy(cudnn));

    return result;
    
}