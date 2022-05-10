//  Copyright 2022 ConvolutedDog (https://github.com/ConvolutedDog/)
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

/*
 * @Author: ConvolutedDog
 * @Date: 2022-03-12 08:40:29
 * @LastEditTime: 2022-04-25 8:05:44
 * @LastEditors: ConvolutedDog
 * @Description: With no preprocess for data.
 * @FilePath: /GPU-experiments/src/nopreprocess.cu
 */
#include "../include/common.h"
#include "../include/helper.h"
#include "../include/config.h"
#include "../include/ampere_sgemm.h"

__global__ void padding_noprocess(float* input_featuremap_data, float* input_featuremap_after_padding, 
                              int warp_num_per_block, int in_n, int in_c, int in_h, int in_w, 
                              int external_pad_h, int external_pad_w, int internal_pad_h, int internal_pad_w,
                              int in_h_after_pad, int in_w_after_pad){
    
    // __shared__ __align__(4*1024) char smem[4*1024];
    // float* tensor_smem = reinterpret_cast<float *>(smem);
    // float wb_reg = 0.;

    // index of thread in a warp, one warp has 1x32 threads
    const uint32_t lane_id = threadIdx.x % 32;
    // index of wrap in a thread block, one warp has 1x32 threads
    const uint32_t warp_id = threadIdx.x / 32;
    
    // Ln means the Ln-th image (N = batch_size) in the last image (after padding).
    // Lc means the Lc-th channel of the Ln-th image in the last image (after padding).
    // Lh means the Lh-th row of the Lc-th channel in the last image (after padding).
    // Lw means the Lw-th column of the Lh-th row in the last image (after padding).
    // [Ln, Lc, Lh, Lw] means the position of the pixel in the last image (after padding).
    const uint32_t Ln = (blockIdx.y * warp_num_per_block + warp_id) / (in_c * in_h_after_pad);
    const uint32_t Lc = (blockIdx.y * warp_num_per_block + warp_id) % (in_c * in_h_after_pad) / in_h_after_pad;
    const uint32_t Lh = (blockIdx.y * warp_num_per_block + warp_id) % (in_c * in_h_after_pad) % in_h_after_pad;
    const uint32_t Lw = blockIdx.x * 32 + lane_id;
    
    // const uint32_t Lhs = blockIdx.y * warp_num_per_block % in_h_after_pad; // start h of this block
    // const uint32_t Lws = blockIdx.x * 32;                                  // start w of this block
    
    // [in, ic, ih, iw]-pixel addr offset in the last image (after padding).
    uint32_t L_addr = Ln*in_c*in_h_after_pad*in_w_after_pad + Lc*in_h_after_pad*in_w_after_pad + Lh*in_w_after_pad + Lw;
    
    // on means the on-th image (N = batch_size) in the old image (no padding).
    // oc means the oc-th channel of the on-th image in the old image (no padding).
    // oh means the oh-th row of the oc-th channel in the old image (no padding).
    // ow means the ow-th column of the oh-th row in the old image (no padding).
    // [on, oc, oh, ow] means the position of the pixel in the old image (no padding).
    const uint32_t on = Ln;
    const uint32_t oc = Lc;
    const uint32_t oh = (Lh - external_pad_h) / (internal_pad_h+1);
    const uint32_t ow = (Lw - external_pad_w) / (internal_pad_w+1);

    // const uint32_t ohs = (Lhs - external_pad_h) / (internal_pad_h+1);
    // const uint32_t ows = (Lws - external_pad_w) / (internal_pad_w+1);

    // [on, oc, oh, ow]-pixel addr offset in the old image (no padding).
    uint32_t o_addr = on*in_c*in_h*in_w + oc*in_h*in_w + oh*in_w + ow;

    // External or Internal padding.
    if(Lw < in_w_after_pad && (blockIdx.y * warp_num_per_block + warp_id) < in_n*in_c*in_h_after_pad){
        if(Lh < external_pad_h || Lw < external_pad_w || Lh >= external_pad_h + in_h+(in_h-1)*internal_pad_h
            || Lw >= external_pad_w + in_w+(in_w-1)*internal_pad_w || 
            (Lh-external_pad_h) % ((internal_pad_h+1)) > 0 || (Lw-external_pad_w) % ((internal_pad_w+1)) > 0 ){
            *(input_featuremap_after_padding + L_addr) = 0.;
        }else{
            *(input_featuremap_after_padding + L_addr) = *(input_featuremap_data + o_addr);
        }
    }
    
    
    // const char* oldTensor_ldg_ptr = (const char*)(input_featuremap_data + oh*in_w + ow);
}


__global__ void im2col_forward(float* input_featuremap_data, float* b_matrix_after_im2col, 
                             int b_row, int b_col, int filt_h, int filt_w, int out_h, int out_w, int str_h, 
                             int str_w, int external_pad_h, int external_pad_w, int internal_pad_h, 
                             int internal_pad_w, int in_h, int in_w, int in_h_after_pad, int in_w_after_pad, 
                             int in_c, int warp_num_per_block, int in_c_in_h_in_w, int in_h_in_w,
                             int tmp_h, int tmp_w){
    
    // index of thread in a warp, one warp has 1x32 threads
    const uint32_t lane_id = threadIdx.x % 32;
    // index of wrap in a thread block, one warp has 1x32 threads
    const uint32_t warp_id = threadIdx.x / 32;

    
    const uint32_t row_b = blockIdx.y * warp_num_per_block + warp_id;
    const uint32_t col_b = blockIdx.x * 32 + lane_id;

    const uint32_t pixel_in_channel_of_image = row_b / (filt_h*filt_w);
    const uint32_t pixel_in_batch_of_image = col_b / (out_h * out_w);
    const uint32_t pixel_row_in_outimage = col_b / out_w % out_h;
    const uint32_t pixel_col_in_outimage = col_b % out_w;
    const uint32_t pixel_row_in_window = row_b / filt_w % filt_h;
    const uint32_t pixel_col_in_window = row_b % filt_w;
    uint32_t pixel_row_in_image = pixel_row_in_outimage*str_h + pixel_row_in_window;
    uint32_t pixel_col_in_image = pixel_col_in_outimage*str_w + pixel_col_in_window;

    uint32_t o_addr;
    
    if(row_b < b_row && col_b < b_col){
        if(pixel_row_in_image < external_pad_h || pixel_col_in_image < external_pad_w ||
        //    pixel_row_in_image >= in_h + (in_h-1)*internal_pad_h + external_pad_h ||
        //    pixel_col_in_image >= in_w + (in_w-1)*internal_pad_w + external_pad_w
            pixel_row_in_image >= tmp_h ||
            pixel_col_in_image >= tmp_w
           ){
            *(b_matrix_after_im2col + row_b*b_col + col_b) = 0.;
        }else{
            // if (lane_id==0 && warp_id == 6 && blockIdx.y==0)printf("%d-%d-%d-%d-%d-%lf.\n", pixel_in_channel_of_image, pixel_in_batch_of_image,
            //     pixel_row_in_image, pixel_col_in_image, row_b*b_col + col_b, *(input_featuremap_data+1));
            
            o_addr = pixel_in_batch_of_image*in_c_in_h_in_w + 
                pixel_in_channel_of_image*in_h_in_w + 
                (pixel_row_in_image-external_pad_h)*in_w + pixel_col_in_image-external_pad_w;
            *(b_matrix_after_im2col + row_b*b_col + col_b) = *(input_featuremap_data+o_addr);
        }
    }

}

__global__ void im2col_loss(float* input_featuremap_data, float* b_matrix_after_im2col, 
                             int b_row, int b_col, int filt_h, int filt_w, int out_h, int out_w, int str_h, 
                             int str_w, int external_pad_h, int external_pad_w, int internal_pad_h, 
                             int internal_pad_w, int in_h, int in_w, int in_h_after_pad, int in_w_after_pad, 
                             int in_c, int warp_num_per_block, int in_c_in_h_in_w, int in_h_in_w,
                             int internal_pad_h_add_1, int internal_pad_w_add_1){
    
    // index of thread in a warp, one warp has 1x32 threads
    const uint32_t lane_id = threadIdx.x % 32;
    // index of wrap in a thread block, one warp has 1x32 threads
    const uint32_t warp_id = threadIdx.x / 32;

    
    const uint32_t row_b = blockIdx.y * warp_num_per_block + warp_id;
    const uint32_t col_b = blockIdx.x * 32 + lane_id;

    const uint32_t pixel_in_channel_of_image = row_b / (filt_h*filt_w);
    const uint32_t pixel_in_batch_of_image = col_b / (out_h * out_w);
    const uint32_t pixel_row_in_outimage = col_b / out_w % out_h;
    const uint32_t pixel_col_in_outimage = col_b % out_w;
    const uint32_t pixel_row_in_window = row_b / filt_w % filt_h;
    const uint32_t pixel_col_in_window = row_b % filt_w;
    uint32_t pixel_row_in_image = pixel_row_in_outimage*str_h + pixel_row_in_window;
    uint32_t pixel_col_in_image = pixel_col_in_outimage*str_w + pixel_col_in_window;

    uint32_t o_addr;
    
    if(row_b < b_row && col_b < b_col){
        if(pixel_row_in_image < external_pad_h || pixel_col_in_image < external_pad_w ||
           (pixel_row_in_image-external_pad_h) % internal_pad_h_add_1 > 0 ||
           (pixel_col_in_image-external_pad_w) % internal_pad_w_add_1 > 0){
            *(b_matrix_after_im2col + row_b*b_col + col_b) = 0.;
        }else{
            pixel_row_in_image = (pixel_row_in_image-external_pad_h)/internal_pad_h_add_1;
            pixel_col_in_image = (pixel_col_in_image-external_pad_w)/internal_pad_w_add_1;

            o_addr = pixel_in_batch_of_image*in_c_in_h_in_w + 
                pixel_in_channel_of_image*in_h_in_w + 
                pixel_row_in_image*in_w + pixel_col_in_image;
            *(b_matrix_after_im2col + row_b*b_col + col_b) = *(input_featuremap_data+o_addr);
        }
    }

    // if (lane_id==1 && warp_id == 0)printf("%d-%d-%d-%d-%d-%lf.\n", pixel_in_channel_of_image, pixel_in_batch_of_image,
    //             pixel_row_in_image, pixel_col_in_image, row_b*b_col + col_b, *(input_featuremap_data+1));
}

__global__ void im2col_noprocess(float* input_featuremap_data, float* b_matrix_after_im2col, 
                             int b_row, int b_col, int filt_h, int filt_w, int out_h, int out_w, int str_h, 
                             int str_w, int external_pad_h, int external_pad_w, int internal_pad_h, 
                             int internal_pad_w, int in_h, int in_w, int in_h_after_pad, int in_w_after_pad, 
                             int in_c, int warp_num_per_block){
    // index of thread in a warp, one warp has 1x32 threads
    const uint32_t lane_id = threadIdx.x % 32;
    // index of wrap in a thread block, one warp has 1x32 threads
    const uint32_t warp_id = threadIdx.x / 32;

    
    const uint32_t row_b = blockIdx.y * warp_num_per_block + warp_id;
    const uint32_t col_b = blockIdx.x * 32 + lane_id;

    const uint32_t pixel_in_channel_of_image = row_b / (filt_h*filt_w);
    const uint32_t pixel_in_batch_of_image = col_b / (out_h * out_w);
    const uint32_t pixel_row_in_outimage = col_b / out_w % out_h;
    const uint32_t pixel_col_in_outimage = col_b % out_w;
    const uint32_t pixel_row_in_window = row_b / filt_w % filt_h;
    const uint32_t pixel_col_in_window = row_b % filt_w;
    uint32_t pixel_row_in_image = pixel_row_in_outimage*str_h + pixel_row_in_window;
    uint32_t pixel_col_in_image = pixel_col_in_outimage*str_w + pixel_col_in_window;

    uint32_t o_addr;
    
    if(row_b < b_row && col_b < b_col){
        if(pixel_row_in_image < external_pad_h || pixel_col_in_image < external_pad_w ||
           pixel_row_in_image >= in_h + (in_h-1)*internal_pad_h + external_pad_h ||
           pixel_col_in_image >= in_w + (in_w-1)*internal_pad_w + external_pad_w ||
           (pixel_row_in_image-external_pad_h) % (internal_pad_h+1) > 0 ||
           (pixel_col_in_image-external_pad_w) % (internal_pad_w+1) > 0){
            *(b_matrix_after_im2col + row_b*b_col + col_b) = 0.;
        }else{
            pixel_row_in_image = (pixel_row_in_image-external_pad_h)/(internal_pad_h+1);
            pixel_col_in_image = (pixel_col_in_image-external_pad_w)/(internal_pad_w+1);

            o_addr = pixel_in_batch_of_image*in_c*in_h*in_w + 
                pixel_in_channel_of_image*in_h*in_w + 
                pixel_row_in_image*in_w + pixel_col_in_image;
            *(b_matrix_after_im2col + row_b*b_col + col_b) = *(input_featuremap_data+o_addr);
        }
    }

    // if (lane_id==1 && warp_id == 0)printf("%d-%d-%d-%d-%d-%lf.\n", pixel_in_channel_of_image, pixel_in_batch_of_image,
    //             pixel_row_in_image, pixel_col_in_image, row_b*b_col + col_b, *(input_featuremap_data+1));
}

Result test_nopreprocess(Options &options) {
    Result result;
    int in_n, in_h, in_w, in_c, filt_d, filt_h, filt_w, filt_c;
    int pad_h, pad_w, str_h, str_w, dil_h, dil_w;
    int out_n, out_c, out_h, out_w;

    // Six warps in one thread block, 32 threads in one warp.
    int warp_num_per_block = 32;
    
    //#######################FORWARD START########################
    printf("\033[32m\nForward Start...\033[0m\n");
    result.gflops = 0;
    result.runtime_ms = 0;
    result.workspace = 0;
    ////////////////////////////FORWARD IM2COL////////////////////////////
    // Some forward parameters of convolution layer.
    in_n = options.input.get_n();
    in_c = options.input.get_c();
    in_h = options.input.get_h();
    in_w = options.input.get_w();
    
    filt_d = options.filter.get_n();
    filt_c = options.filter.get_c();
    filt_h = options.filter.get_h();
    filt_w = options.filter.get_w();

    pad_h = options.padding.get_n();
    pad_w = options.padding.get_h();
    str_h = options.conv_stride.get_M();
    str_w = options.conv_stride.get_N();
    dil_h = options.dilation.get_M();
    dil_w = options.dilation.get_N();

    // Declare tensor_a as Input feature map.
    Tensor4d<DTypeInput> tensor_b(options.input);
    // Declare tensor_b as Filter.
    Tensor4d<DTypeFilter> tensor_a(options.filter);

    // Randomly initial tensor_a values.
    tensor_a.init_host_data_random(); 
    // Randomly initial tensor_b values.
    tensor_b.init_host_data_random(); 
    
    // Sync on the gpu, and call tensor_x.device_data() to
    // get the device address of tensor_x.
    tensor_a.sync_device();
    tensor_b.sync_device();
    
    // Calculate the output shape of forward.
    out_n = in_n;
    out_c = filt_d;
    out_h = (in_h+2*pad_h-filt_h) / str_h + 1;
    out_w = (in_w+2*pad_w-filt_w) / str_w + 1;

#ifdef PRINT
    printf("\n========================Parameters Info========================\n");
    printf("in_h, in_w, filt_h, filt_w, pad_h, pad_w, str_h, str_w: %d, %d, %d, %d, %d, %d, %d, %d.\n", 
            in_h, in_w, filt_h, filt_w, pad_h, pad_w, str_h, str_w);
    printf("out_n, out_c, out_h, out_w: %d, %d, %d, %d.\n", out_n, out_c, out_h, out_w);
    printf("input_featuremap size: %d.\n", options.input.get_size());
    printf("========================Parameters Info========================\n\n");
#endif

    // Declare output_feature_map, and sync like tensor_a/b.
    // int output_featuremap_size = out_n * out_c * out_h * out_w;
    // Tensor4d<DTypeOutput> tensor_y(output_featuremap_size);
    // tensor_y.sync_device();

    // Call .device_data() to get device_addr.
    float *input_featuremap_data = tensor_b.device_data();  //device addr
    float *filter_data = tensor_a.device_data();            //device addr
    // float *output_featuremap_data = tensor_y.device_data(); //device addr
    
    
#ifdef PRINT
    // PRINT is defined in config.h, to see host data of tensor_b.
    printf("\n###########Host Before Padding Start###########\n");
    for(size_t i = 0; i < in_n; ++i){
        printf("=====================%d-th image\n", i);
        for(size_t j = 0; j < in_c; ++j){
            printf("--------------------%d-th channel\n", j);
            for(size_t k = 0; k < in_h; ++k){
                for(size_t l = 0; l < in_w; ++l){
                    printf("%.3f ", *(tensor_b.host_data()+i*in_h*in_w*in_c+j*in_h*in_w+k*in_w+l));
                }
                 printf("\n");
            }
        } 
    }
    printf("###########Host Before Padding End#############\n\n");
#endif


    // This the real H and W after that the image is 
    // padded, consists of external and internal padding.
    // For forward, in_h_after_pad = in_h + 2*pad_h, in_w_after_pad = in_w + 2*pad_w.
    int in_h_after_pad = in_h+2*pad_h;
    int in_w_after_pad = in_w+2*pad_w;
    
    // Record time, ms.
    cudaEvent_t events[2];
    float all_time = 0;
    for (auto & event : events) {
      cudaEventCreate(&event);
    }
    int iter;
  
    ////////////////////////////FORWARD IM2COL////////////////////////////

    // Column of B matrix after im2col. 
    long b_col = in_n * out_h * out_w;
    // Row of B matrix after im2col. 
    long b_row = in_c * filt_h * filt_w;
    
    // The total size of B matrix.
    long b_matrx_size = b_col * b_row;
    
    // Declare 2D B matrix and sync.
    Tensor2d<DTypeInput> b_matrix(b_matrx_size); 
    b_matrix.sync_device();
    
    // Call .device_data() to get device addr of B matrix.
    float* b_matrix_after_im2col = b_matrix.device_data();
    
    // printf("Start to do im2col...\n");

    // Put every pixel on one thread, and one warp has a 1x32 threads grop. So
    // the x-dimensional block num is (b_col + 31) / 32, and y-dimensional
    // block num is (b_row + warp_num_per_block-1) / warp_num_per_block.
    dim3 grid_naive_im2col((b_col + 31) / 32, (b_row + warp_num_per_block-1) / warp_num_per_block);
    
    // warm up
    im2col_forward<<<grid_naive_im2col, warp_num_per_block*32>>>(input_featuremap_data, 
            b_matrix_after_im2col, b_row, b_col, filt_h, filt_w, out_h, out_w, str_h, str_w,
            pad_h, pad_w, 0, 0, in_h, in_w,
            in_h_after_pad, in_w_after_pad, in_c, warp_num_per_block, in_c*in_h*in_w,
            in_h*in_w, in_h + pad_h, in_w + pad_w);
    

    cudaEventRecord(events[0]);

    for(iter=0; iter<options.iterations; iter++){
        im2col_forward<<<grid_naive_im2col, warp_num_per_block*32>>>(input_featuremap_data, 
            b_matrix_after_im2col, b_row, b_col, filt_h, filt_w, out_h, out_w, str_h, str_w,
            pad_h, pad_w, 0, 0, in_h, in_w,
            in_h_after_pad, in_w_after_pad, in_c, warp_num_per_block, in_c*in_h*in_w,
            in_h*in_w, in_h + pad_h, in_w + pad_w);
    }

    cudaEventRecord(events[1]);

    cudaEventSynchronize(events[1]);
    cudaEventElapsedTime(&all_time, events[0], events[1]);
    
    result.runtime_ms += double(all_time) / double(options.iterations);
    result.gflops += 0;
    result.workspace += 0;
    
    // Copy the input featuremap map after im2col back to host.
    float* host_input_featuremap_after_im2col;
    CUDA_CALL(cudaMallocHost(&host_input_featuremap_after_im2col, b_matrx_size * sizeof(float)));
    CUDA_CALL(cudaMemcpy(host_input_featuremap_after_im2col, b_matrix_after_im2col, 
              b_matrx_size * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef PRINT
    // PRINT is defined in config.h, to see host data of tensor_a after im2col.
    printf("\n###########Host Filt Data Start###########\n");
    for(size_t i = 0; i < filt_d; ++i){
        for(size_t j = 0; j < b_row; ++j){
            printf("%.3f ", *(tensor_a.host_data()+
                i*b_row + j));
        }
        printf("\n");
    }
    printf("###########Host Filt Data End#############\n\n");
#endif

#ifdef PRINT
    // PRINT is defined in config.h, to see host data of tensor_b after im2col.
    printf("\n###########Host After Im2col Start###########\n");
    for(size_t i = 0; i < b_row; ++i){
        for(size_t j = 0; j < b_col; ++j){
            printf("%.3f ", *(host_input_featuremap_after_im2col+
                i*b_col + j));
        }
        printf("\n");
    }
    printf("###########Host After Im2col End#############\n\n");
#endif

#ifdef DEBUG
    printf("\033[31mPRENO Forward Im2col time: %lf ms.\033[0m\n", float(all_time)/options.iterations);
#endif

    ////////////////////////////FORWARD MMA////////////////////////////
    size_t m = filt_d;
    size_t k = b_row;
    size_t n = b_col;

    int c_matrx_size = m * n;
    
    // Declare 2D C matrix and sync.
    Tensor2d<DTypeInput> c_matrix(c_matrx_size); 
    c_matrix.sync_device();

    // Call .device_data() to get device addr of C matrix.
    float* c_matrix_result = c_matrix.device_data();

    // printf("Start to do MMA...\n");

    dim3 grid_mma((n + 127) / 128, (m + 127) / 128);

    // warmup
    sgemm_128x128x8_kernel<<<grid_mma, 256>>>(
        filter_data, b_matrix_after_im2col, c_matrix_result, m, n, k, 
        k * sizeof(float), n * sizeof(float) * 8);
    
    cudaEventRecord(events[0]);

    for(iter=0; iter<options.iterations; iter++){
        sgemm_128x128x8_kernel<<<grid_mma, 256>>>(
            filter_data, b_matrix_after_im2col, c_matrix_result, m, n, k, 
            k * sizeof(float), n * sizeof(float) * 8);
    }
    
    cudaEventRecord(events[1]);
    
    cudaEventSynchronize(events[1]);
    cudaEventElapsedTime(&all_time, events[0], events[1]);
    
    result.runtime_ms += double(all_time) / double(options.iterations);
    long workload = m * n * k * 2;
    result.gflops += (double(workload) / 1e9) / (double(result.runtime_ms) / 1e3);
    result.workspace += 0;
    
    // printf("END...\n");

    // Copy the input featuremap map after im2col back to host.
    float* host_output_featuremap_after_mma;
    CUDA_CALL(cudaMallocHost(&host_output_featuremap_after_mma, filt_d * b_col * sizeof(float)));
    CUDA_CALL(cudaMemcpy(host_output_featuremap_after_mma, c_matrix_result, 
              filt_d * b_col * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef PRINT
    // PRINT is defined in config.h, to see host data of c_matrix_result after im2col.
    printf("\n###########Host MMA RESULT Start###########\n");
    for(size_t i = 0; i < m; ++i){
        for(size_t j = 0; j < n; ++j){
            printf("%.3f ", *(host_output_featuremap_after_mma+i*n + j));
        }
        printf("\n");
    }
    printf("###########Host MMA RESULT End#############\n\n");
#endif

#ifdef DEBUG
    printf("\033[31mPRENO Forward MMA time: %lf ms.\033[0m\n", float(all_time)/options.iterations);
#endif

#ifdef DEBUG
    printf("\033[31mPRENO Forward workload: %ld.\033[0m\n", workload);
    printf("\033[31mPRENO Forward Total time: %lf ms.\033[0m\n", result.runtime_ms);
    printf("\033[31mPRENO Forward Total gflops: %lf gflops.\033[0m\n", result.gflops);
    printf("\033[31mPRENO Forward M = %d, K = %d, N = %d.\033[0m\n", m, k, n);
#endif
    
    //#######################FORWARD END##########################

    //#######################LOSS START###########################
    printf("\033[32m\nLoss Start...\033[0m\n");
    result.gflops = 0;
    result.runtime_ms = 0;
    result.workspace = 0;
    
    // Forward parameters.
    in_n = options.input.get_n();
    in_c = options.input.get_c();
    in_h = options.input.get_h();
    in_w = options.input.get_w();
    
    filt_d = options.filter.get_n();
    filt_c = options.filter.get_c();
    filt_h = options.filter.get_h();
    filt_w = options.filter.get_w();

    pad_h = options.padding.get_n();
    pad_w = options.padding.get_h();
    str_h = options.conv_stride.get_M();
    str_w = options.conv_stride.get_N();
    dil_h = options.dilation.get_M();
    dil_w = options.dilation.get_N();

    // Loss: input feature map shape, equals forward 
    //       output feature map shape
    in_n = out_n;
    in_c = out_c;
    in_h = out_h;
    in_w = out_w;

    // Loss: filter feature map, 0-1 dimentional transpose,
    //       rot180 donot affect shape of filter, so no rot180.
    filt_d = options.filter.get_c();
    filt_c = options.filter.get_n();
    filt_h = options.filter.get_h();
    filt_w = options.filter.get_w();
    
    // Loss: pad_loss = filter_size - 1 - pad_forward
    //       str_h_loss = str_w_loss = 1
    //       dil_h = dil_w = 1
    pad_h = filt_h - 1 - options.padding.get_n();
    pad_w = filt_w - 1 - options.padding.get_h();
    str_h = 1;
    str_w = 1;
    dil_h = 1;
    dil_w = 1;
    
    // Loss: output feature map shape = forward input feature map shape
    out_n = options.input.get_n();
    out_c = options.input.get_c();
    out_h = options.input.get_h();
    out_w = options.input.get_w();

#ifdef PRINT
    printf("\n========================Parameters Info========================\n");
    printf("Forward Paras:\n");
    printf("\tin_n, in_c, filter_d, in_h, in_w, filt_h, filt_w, pad_h, pad_w, str_h, str_w: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d.\n", 
            options.input.get_n(), options.input.get_c(), options.filter.get_n(),
            options.input.get_h(), options.input.get_w(), options.filter.get_h(), options.filter.get_w(), 
            options.padding.get_n(), options.padding.get_h(), options.conv_stride.get_M(), options.conv_stride.get_N());
    printf("Loss Paras:\n");
    printf("\tin_h, in_w, filt_h, filt_w, pad_h, pad_w, str_h, str_w: %d, %d, %d, %d, %d, %d, %d, %d.\n", 
            in_h, in_w, filt_h, filt_w, pad_h, pad_w, str_h, str_w);
    printf("\tout_n, out_c, out_h, out_w: %d, %d, %d, %d.\n", out_n, out_c, out_h, out_w);
    printf("========================Parameters Info========================\n\n");
#endif
    
    // Declare LossNextLayer as Input feature map of Loss, loss of next layer.
    Tensor4dParam LossNextLayer_shape(in_n, in_c, in_h, in_w);
    Tensor4d<DTypeFilter> LossNextLayer(LossNextLayer_shape);
    // Declare rot180WT as Filter of Loss.
    Tensor4d<DTypeFilter> rot180WT(tensor_a);
    
    // Randomly initial LossNextLayer values.
    LossNextLayer.init_host_data_random(); 
    // Randomly initial rot180WT values.
    rot180WT.init_host_data_random(); 
    
    // Sync on the gpu, and call tensor_x.device_data() to
    // get the device address of tensor_x.
    LossNextLayer.sync_device();
    rot180WT.sync_device();

    // Call .device_data() to get device_addr.
    input_featuremap_data = LossNextLayer.device_data(); // device addr
    filter_data = rot180WT.device_data();                //device addr

      
#ifdef PRINT
    // PRINT is defined in config.h, to see host data of LossNextLayer.
    printf("\n###########Host Before Padding Start###########\n");
    for(size_t i = 0; i < in_n; ++i){
        printf("=====================%d-th LossNextLayer\n", i);
        for(size_t j = 0; j < in_c; ++j){
            printf("--------------------%d-th channel\n", j);
            for(size_t k = 0; k < in_h; ++k){
                for(size_t l = 0; l < in_w; ++l){
                    printf("%.3f ", *(LossNextLayer.host_data()+i*in_h*in_w*in_c+j*in_h*in_w+k*in_w+l));
                }
                 printf("\n");
            }
        } 
    }
    printf("###########Host Before Padding End#############\n\n");
    // PRINT is defined in config.h, to see host data of rot180WT.
    printf("\n###########Host Before Padding Start###########\n");
    for(size_t i = 0; i < filt_d; ++i){
        printf("=====================%d-th rot180WT\n", i);
        for(size_t j = 0; j < filt_c; ++j){
            printf("--------------------%d-th channel\n", j);
            for(size_t k = 0; k < filt_h; ++k){
                for(size_t l = 0; l < filt_w; ++l){
                    printf("%.3f ", *(rot180WT.host_data()+i*filt_h*filt_w*filt_c+j*filt_h*filt_w+k*filt_w+l));
                }
                 printf("\n");
            }
        } 
    }
    printf("###########Host Before Padding End#############\n\n");
#endif

    // This the real H and W after that the image is 
    // padded, consists of external and internal padding.
    // For loss, in_h_after_pad = in_h + 2*pad_h + (in_h-1)*(str_h_forward-1), 
    //           in_w_after_pad = in_w + 2*pad_w + (in_w-1)*(str_w_forward-1).
    in_h_after_pad = in_h + 2*pad_h + (in_h-1)*(options.conv_stride.get_M()-1);
    in_w_after_pad = in_w + 2*pad_w + (in_w-1)*(options.conv_stride.get_N()-1);

    ////////////////////////////LOSS IM2COL////////////////////////////
    // For the reason that loss's im2col is simple with forward, 
    // we use im2col_forward to caculate loss.

    // Column of B matrix after im2col. 
    b_col = in_n * out_h * out_w;
    // Row of B matrix after im2col. 
    b_row = in_c * filt_h * filt_w;
    
    // The total size of B matrix.
    b_matrx_size = b_col * b_row;
    
    // printf("%d-%d-%d-%d-%d.\n", in_n, out_h, out_w, in_c, filt_h, filt_w);
    // printf("%ld\n", b_matrx_size);
    // printf("%d-%d-%d\n", filt_d, b_row, b_col);

    // Declare 2D B matrix and sync.
    Tensor2d<DTypeInput> b_matrix_loss(b_matrx_size); 
    b_matrix_loss.sync_device();

    // Call .device_data() to get device addr of B matrix.
    b_matrix_after_im2col = b_matrix_loss.device_data();

    // Put every pixel on one thread, and one warp has a 1x32 threads grop. So
    // the x-dimensional block num is (b_col + 31) / 32, and y-dimensional
    // block num is (b_row + warp_num_per_block-1) / warp_num_per_block.
    dim3 grid_naive_im2col_loss((b_col + 31) / 32, (b_row + warp_num_per_block-1) / warp_num_per_block);
    
    // warm up
    im2col_loss<<<grid_naive_im2col_loss, warp_num_per_block*32>>>(input_featuremap_data, 
            b_matrix_after_im2col, b_row, b_col, filt_h, filt_w, out_h, out_w, str_h, str_w,
            pad_h, pad_w, options.conv_stride.get_M()-1, options.conv_stride.get_N()-1,
            in_h, in_w, in_h_after_pad, in_w_after_pad, in_c, warp_num_per_block,
            in_c*in_h*in_w, in_h*in_w, options.conv_stride.get_M(), options.conv_stride.get_N());

#ifdef PRINT
    printf("\n========================Parameters Info========================\n");
    printf("Forward Paras:\n");
    printf("\tin_n, in_c, filter_d, in_h, in_w, filt_h, filt_w, pad_h, pad_w, str_h, str_w: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d.\n", 
            options.input.get_n(), options.input.get_c(), options.filter.get_n(),
            options.input.get_h(), options.input.get_w(), options.filter.get_h(), options.filter.get_w(), 
            options.padding.get_n(), options.padding.get_h(), options.conv_stride.get_M(), options.conv_stride.get_N());
    printf("Loss Paras:\n");
    printf("\tin_h, in_w, filt_h, filt_w, pad_h, pad_w, str_h, str_w: %d, %d, %d, %d, %d, %d, %d, %d.\n", 
            in_h, in_w, filt_h, filt_w, pad_h, pad_w, str_h, str_w);
    printf("\tin_c, out_n, out_c, out_h, out_w: %d, %d, %d, %d.\n", in_c, out_n, out_c, out_h, out_w);
    printf("\tb_row, b_col, filt_h, filt_w, out_h, out_w, str_h, str_w, in_h_after_pad, in_w_after_pad, in_c, warp_num_per_block: %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", 
            b_row, b_col, filt_h, filt_w, out_h, out_w, str_h, str_w,
            in_h_after_pad, in_w_after_pad, in_c, warp_num_per_block);
    printf("========================Parameters Info========================\n\n");
#endif

    // Record time, ms.
    all_time = 0;
    for (auto & event : events) {
      cudaEventCreate(&event);
    }

    cudaEventRecord(events[0]);

    for(iter=0; iter<options.iterations; iter++){
        im2col_loss<<<grid_naive_im2col_loss, warp_num_per_block*32>>>(input_featuremap_data, 
            b_matrix_after_im2col, b_row, b_col, filt_h, filt_w, out_h, out_w, str_h, str_w,
            pad_h, pad_w, options.conv_stride.get_M()-1, options.conv_stride.get_N()-1,
            in_h, in_w, in_h_after_pad, in_w_after_pad, in_c, warp_num_per_block,
            in_c*in_h*in_w, in_h*in_w, options.conv_stride.get_M(), options.conv_stride.get_N());
    }
    // printf("in_h_after_pad, in_w_after_pad: %d-%d\n", in_h_after_pad, in_w_after_pad);
    cudaEventRecord(events[1]);

    cudaEventSynchronize(events[1]);
    cudaEventElapsedTime(&all_time, events[0], events[1]);
    
    result.runtime_ms += double(all_time) / double(options.iterations);
    result.gflops += 0;
    result.workspace += 0;

    // Copy the input featuremap map after im2col back to host.
    CUDA_CALL(cudaMallocHost(&host_input_featuremap_after_im2col, b_matrx_size * sizeof(float)));
    CUDA_CALL(cudaMemcpy(host_input_featuremap_after_im2col, b_matrix_after_im2col, 
              b_matrx_size * sizeof(float), cudaMemcpyDeviceToHost));


#ifdef PRINT
    // PRINT is defined in config.h, to see host data of LossNextLayer.
    printf("\n###########Host Before Padding Start###########\n");
    for(size_t i = 0; i < in_n; ++i){
        printf("=====================%d-th LossNextLayer\n", i);
        for(size_t j = 0; j < in_c; ++j){
            printf("--------------------%d-th channel\n", j);
            for(size_t k = 0; k < in_h; ++k){
                for(size_t l = 0; l < in_w; ++l){
                    printf("%.3f ", *(LossNextLayer.host_data()+i*in_h*in_w*in_c+j*in_h*in_w+k*in_w+l));
                }
                 printf("\n");
            }
        } 
    }
    // PRINT is defined in config.h, to see host data of tensor_b after im2col.
    printf("\n###########Host After Im2col Start###########\n");
    for(size_t i = 0; i < b_row; ++i){
        for(size_t j = 0; j < b_col; ++j){
            printf("%.3f ", *(host_input_featuremap_after_im2col+
                i*b_col + j));
        }
        printf("\n");
    }
    printf("###########Host After Im2col End#############\n\n");
#endif

#ifdef DEBUG
    printf("\033[31mPRENO Loss Im2col time: %lf ms.\033[0m\n", float(all_time)/options.iterations);
#endif

    ////////////////////////////LOSS MMA////////////////////////////
    m = filt_d;
    k = b_row;
    n = b_col;
    
    c_matrx_size = m * n;

    // Declare 2D C matrix and sync.
    Tensor2d<DTypeInput> c_matrix_loss(c_matrx_size); 
    c_matrix_loss.sync_device();

    // Call .device_data() to get device addr of C matrix.
    c_matrix_result = c_matrix_loss.device_data();

    dim3 grid_mma_loss((n + 127) / 128, (m + 127) / 128);

    // warmup
    sgemm_128x128x8_kernel<<<grid_mma_loss, 256>>>(
        filter_data, b_matrix_after_im2col, c_matrix_result, m, n, k, 
        k * sizeof(float), n * sizeof(float) * 8);
    
    cudaEventRecord(events[0]);

    for(iter=0; iter<options.iterations; iter++){
        sgemm_128x128x8_kernel<<<grid_mma_loss, 256>>>(
            filter_data, b_matrix_after_im2col, c_matrix_result, m, n, k, 
            k * sizeof(float), n * sizeof(float) * 8);
    }
    
    cudaEventRecord(events[1]);
    
    cudaEventSynchronize(events[1]);
    cudaEventElapsedTime(&all_time, events[0], events[1]);
    
    result.runtime_ms += double(all_time) / double(options.iterations);
    workload = m * n * k * 2;
    result.gflops += (double(workload) / 1e9) / (double(result.runtime_ms) / 1e3);
    result.workspace += 0;
    
    // printf("END...\n");

    // Copy the input featuremap map after im2col back to host.
    CUDA_CALL(cudaMallocHost(&host_output_featuremap_after_mma, filt_d * b_col * sizeof(float)));
    CUDA_CALL(cudaMemcpy(host_output_featuremap_after_mma, c_matrix_result, 
              filt_d * b_col * sizeof(float), cudaMemcpyDeviceToHost));


#ifdef PRINT
    // PRINT is defined in config.h, to see host data of c_matrix_result after im2col.
    printf("\n###########Host MMA RESULT Start###########\n");
    for(size_t i = 0; i < m; ++i){
        for(size_t j = 0; j < n; ++j){
            printf("%.3f ", *(host_output_featuremap_after_mma+i*n + j));
        }
        printf("\n");
    }
    printf("###########Host MMA RESULT End#############\n\n");
#endif

#ifdef DEBUG
    printf("\033[31mPRENO Loss MMA time: %lf ms.\033[0m\n", float(all_time)/options.iterations);
#endif

#ifdef DEBUG
    printf("\033[31mPRENO Loss workload: %ld.\033[0m\n", workload);
    printf("\033[31mPRENO Loss Total time: %lf ms.\033[0m\n", result.runtime_ms);
    printf("\033[31mPRENO Loss Total gflops: %lf gflops.\033[0m\n", result.gflops);
    printf("\033[31mPRENO Loss M = %d, K = %d, N = %d.\033[0m\n", m, k, n);
#endif
    //#######################LOSS END#############################

    //#######################GRADIENT START ######################
    printf("\033[32m\nGradient Start...\033[0m\n");
    result.gflops = 0;
    result.runtime_ms = 0;
    result.workspace = 0;
    
    // Forward parameters.
    in_n = options.input.get_n();
    in_c = options.input.get_c();
    in_h = options.input.get_h();
    in_w = options.input.get_w();
    
    filt_d = options.filter.get_n();
    filt_c = options.filter.get_c();
    filt_h = options.filter.get_h();
    filt_w = options.filter.get_w();

    pad_h = options.padding.get_n();
    pad_w = options.padding.get_h();
    str_h = options.conv_stride.get_M();
    str_w = options.conv_stride.get_N();
    dil_h = options.dilation.get_M();
    dil_w = options.dilation.get_N();
    
    out_n = in_n;
    out_c = filt_d;
    out_h = (in_h+2*pad_h-filt_h) / str_h + 1;
    out_w = (in_w+2*pad_w-filt_w) / str_w + 1;

    // printf("out_h: %d, out_w: %d\n", out_h, out_w);
    
    int gradient_in_n, gradient_in_c, gradient_in_h, gradient_in_w;
    int gradient_out_n, gradient_out_c, gradient_out_h, gradient_out_w;
    int gradient_pad_h, gradient_pad_w;
    int gradient_str_h, gradient_str_w;
    int gradient_filt_n, gradient_filt_c, gradient_filt_h, gradient_filt_w;

    // Gradient Parameters
    gradient_in_n = in_c;
    gradient_in_c = in_n;
    gradient_in_h = in_h;
    gradient_in_w = in_w;
    
    gradient_out_n = gradient_in_n;
    gradient_out_c = out_c;
    gradient_out_h = filt_h;
    gradient_out_w = filt_w;

    gradient_pad_h = pad_h;
    gradient_pad_w = pad_w;
    gradient_str_h = 1;
    gradient_str_w = 1;
    
    gradient_filt_n = gradient_out_c;
    gradient_filt_c = in_n;
    gradient_filt_h = out_h;
    gradient_filt_w = out_w;
    
    // Declare graident_LossNextLayer as Filter.
    Tensor4dParam gradient_LossNextLayer_shape(gradient_filt_n, gradient_filt_c, gradient_filt_h, gradient_filt_w);
    Tensor4d<DTypeInput> graident_LossNextLayer(gradient_LossNextLayer_shape);

    // Declare gradient_Inputfeaturemap as Input feature map.
    Tensor4dParam Inputfeaturemap_shape(gradient_in_n, gradient_in_c, gradient_in_h, gradient_in_w);
    Tensor4d<DTypeInput> gradient_Inputfeaturemap(Inputfeaturemap_shape);
    
    // Randomly initial and sync.
    graident_LossNextLayer.init_host_data_random(); 
    graident_LossNextLayer.sync_device(); 

    // Randomly initial and sync.
    gradient_Inputfeaturemap.init_host_data_random(); 
    gradient_Inputfeaturemap.sync_device(); 
    
    // Call .device_data() to get device_addr.
    float* d_LossNextLayer_data = graident_LossNextLayer.device_data();  //device addr
    float* d_Inputfeaturemap_data = gradient_Inputfeaturemap.device_data();  //device addr
    
    // Declare gradient_LossNextLayer_after_padding as Filter after padding.
    Tensor4dParam LossNextLayer_after_padding_shape(gradient_filt_n, gradient_filt_c, 
                  out_h+(out_h-1)*(str_h-1), out_w+(out_w-1)*(str_w-1));
    Tensor4d<DTypeInput> gradient_LossNextLayer_after_padding(LossNextLayer_after_padding_shape);
    gradient_LossNextLayer_after_padding.sync_device(); 
    float* d_LossNextLayer_after_padding_data = gradient_LossNextLayer_after_padding.device_data();  //device addr

    // Declare Inputfeaturemap_after_padding as Input feature map after padding.
    Tensor4dParam Inputfeaturemap_after_padding_shape(gradient_in_n, gradient_in_c, 
                  in_h+2*pad_h, in_w+2*pad_w);
    Tensor4d<DTypeInput> Inputfeaturemap_after_padding(Inputfeaturemap_after_padding_shape);
    Inputfeaturemap_after_padding.sync_device(); 
    float* d_Inputfeaturemap_after_padding_data = Inputfeaturemap_after_padding.device_data();  //device addr

#ifdef PRINT
    // PRINT is defined in config.h, to see host data of graident_LossNextLayer before padding.
    printf("\n###########GRADIENT - Host Before Padding Start - graident_LossNextLayer###########\n");
    for(size_t i = 0; i < out_c; ++i){
        printf("=====================%d-th image\n", i);
        for(size_t j = 0; j < in_n; ++j){
            printf("--------------------%d-th channel\n", j);
            for(size_t k = 0; k < out_h; ++k){
                for(size_t l = 0; l < out_w; ++l){
                    printf("%.3f ", *(graident_LossNextLayer.host_data()+i*out_h*out_w*in_n+j*out_h*out_w+k*out_w+l));
                }
                 printf("\n");
            }
        } 
    }
    printf("###########GRADIENT - Host Before Padding End - graident_LossNextLayer#############\n\n");
#endif

#ifdef PRINT
    // PRINT is defined in config.h, to see host data of gradient_Inputfeaturemap before padding.
    printf("\n###########GRADIENT - Host Before Padding Start - gradient_Inputfeaturemap###########\n");
    for(size_t i = 0; i < in_c; ++i){
        printf("=====================%d-th image\n", i);
        for(size_t j = 0; j < in_n; ++j){
            printf("--------------------%d-th channel\n", j);
            for(size_t k = 0; k < in_h; ++k){
                for(size_t l = 0; l < in_w; ++l){
                    printf("%.3f ", *(gradient_Inputfeaturemap.host_data()+i*in_h*in_w*in_n+j*in_h*in_w+k*in_w+l));
                }
                 printf("\n");
            }
        } 
    }
    printf("###########GRADIENT - Host Before Padding End - gradient_Inputfeaturemap#############\n\n");
#endif

    // 1. filt padding.
    // Record time, ms.
    all_time = 0;
    for (auto & event : events) {
      cudaEventCreate(&event);
    }
    
    // printf("Start to add external or internal padding...\n");
    
    // Put every pixel on one thread, and one warp has a 1x32 threads grop. So
    // the x-dimensional block num is (in_w_after_pad + 31) / 32, and y-dimensional
    // block num is (in_h_after_pad*in_c*in_n + warp_num_per_block-1) / warp_num_per_block.
    // ---------------------------------
    // |  1x32 |       |       |       |
    // |6 warps|       |       |       |
    // |block 0|block 1|       |       |
    // ---------------------------------
    // |       |       |       |       |
    // |       |       |       |       |
    // |       |       |       |       |
    // ---------------------------------
    dim3 gradient_grid_naive_padding((out_w+(out_w-1)*(str_w-1) + 31) / 32, ((out_h+(out_h-1)*(str_h-1))*gradient_filt_c*
         gradient_filt_n + warp_num_per_block-1) / warp_num_per_block);
    
    // warm up
    padding_noprocess<<<gradient_grid_naive_padding, warp_num_per_block*32>>>(
            d_LossNextLayer_data, d_LossNextLayer_after_padding_data, warp_num_per_block, 
            gradient_filt_n, gradient_filt_c, gradient_filt_h, gradient_filt_w, 0, 0, 
            str_h-1, str_w-1, out_h+(out_h-1)*(str_h-1), out_h+(out_w-1)*(str_w-1));
    
    // Iteration to seek average time.
    cudaEventRecord(events[0]);
    
    // external or internal padding
    for(iter=0; iter<options.iterations; iter++){
        padding_noprocess<<<gradient_grid_naive_padding, warp_num_per_block*32>>>(
            d_LossNextLayer_data, d_LossNextLayer_after_padding_data, warp_num_per_block, 
            gradient_filt_n, gradient_filt_c, gradient_filt_h, gradient_filt_w, 0, 0, 
            str_h-1, str_w-1, out_h+(out_h-1)*(str_h-1), out_h+(out_w-1)*(str_w-1));
    }
    
    cudaEventRecord(events[1]);

    cudaEventSynchronize(events[1]);
    cudaEventElapsedTime(&all_time, events[0], events[1]);

    result.runtime_ms += double(all_time) / double(options.iterations);
    result.gflops += 0;
    result.workspace += 0;
    
    // Copy the input featuremap map after padding back to host.
    float* host_LossNextLayer_after_padding_data_after_padding;
    CUDA_CALL(cudaMallocHost(&host_LossNextLayer_after_padding_data_after_padding, 
              gradient_filt_n * gradient_filt_c * (out_h+(out_h-1)*(str_h-1)) * 
              (out_h+(out_w-1)*(str_w-1)) * sizeof(float)));
    CUDA_CALL(cudaMemcpy(host_LossNextLayer_after_padding_data_after_padding, 
              d_LossNextLayer_after_padding_data, gradient_filt_n * gradient_filt_c * 
              (out_h+(out_h-1)*(str_h-1)) * (out_h+(out_w-1)*(str_w-1)) * sizeof(float), cudaMemcpyDeviceToHost));


#ifdef PRINT
    // PRINT is defined in config.h, to see host data of host_LossNextLayer_after_padding_data_after_padding.
    printf("\n###########Host After Padding Start###########\n");
    for(size_t i = 0; i < gradient_filt_n; ++i){
        printf("=====================%d-th image\n", i);
        for(size_t j = 0; j < gradient_filt_c; ++j){
            printf("--------------------%d-th channel\n", j);
            for(size_t k = 0; k < out_h+(out_h-1)*(str_h-1); ++k){
                for(size_t l = 0; l < out_w+(out_w-1)*(str_w-1); ++l){
                    printf("%.3f ", *(host_LossNextLayer_after_padding_data_after_padding+
                    i*(out_h+(out_h-1)*(str_h-1))*(out_w+(out_w-1)*(str_w-1))*
                    gradient_filt_c+j*(out_h+(out_h-1)*(str_h-1))*
                    (out_w+(out_w-1)*(str_w-1))+k*(out_w+(out_w-1)*(str_w-1))+l));
                }
                printf("\n");
            }
        }
    }
    printf("###########Host After Padding End#############\n\n");
#endif

#ifdef DEBUG
    printf("\033[31mPRENO Gradient LossNextLayer Padding time: %lf ms.\033[0m\n", float(all_time)/options.iterations);
#endif

    ////////////////////////////GRADIENT IM2COL////////////////////////////
    // For the reason that loss's im2col is simple with forward, 
    // we use im2col_forward to caculate gradient.
    
    // gradient_in_n = in_c;
    // gradient_in_c = in_n;
    // gradient_in_h = in_h;
    // gradient_in_w = in_w;

    // gradient_out_n = gradient_in_n;
    // gradient_out_c = out_c;
    // gradient_out_h = filt_h;
    // gradient_out_w = filt_w;
    
    
    // Column of B matrix after im2col. 
    b_col = gradient_out_n * gradient_out_h * gradient_out_w;
    // Row of B matrix after im2col. 
    b_row = gradient_in_c * (out_h+(out_h-1)*(str_h-1)) * (out_w+(out_w-1)*(str_w-1));
    
    // The total size of B matrix.
    b_matrx_size = b_col * b_row;

#ifdef PRINT
    printf("Input: %d-%d-%d-%d.\n", gradient_in_n, gradient_in_c, gradient_in_h+2*pad_h, 
            gradient_in_w+2*pad_w);
    printf("Weight: %d-%d-%d-%d.\n",  gradient_filt_n, gradient_filt_c, out_h+(out_h-1)*(str_h-1), 
            out_w+(out_w-1)*(str_w-1));
    printf("b_matrx_size: %d.\n", b_matrx_size);
    printf("b_row, b_col: %d-%d.\n", b_row, b_col);
    printf("a_row: %d.\n", gradient_filt_n);
#endif

    // Declare 2D B matrix and sync.
    Tensor2d<DTypeInput> b_matrix_gradient(b_matrx_size); 
    b_matrix_gradient.sync_device();

    // Call .device_data() to get device addr of B matrix.
    b_matrix_after_im2col = b_matrix_gradient.device_data();

    // Put every pixel on one thread, and one warp has a 1x32 threads grop. So
    // the x-dimensional block num is (b_col + 31) / 32, and y-dimensional
    // block num is (b_row + warp_num_per_block-1) / warp_num_per_block.
    dim3 grid_naive_im2col_gradient((b_col + 31) / 32, (b_row + warp_num_per_block-1) / warp_num_per_block);
    
    // im2col_forward(float* input_featuremap_data, float* b_matrix_after_im2col, 
    //                          int b_row, int b_col, int filt_h, int filt_w, int out_h, int out_w, int str_h, 
    //                          int str_w, int external_pad_h, int external_pad_w, int internal_pad_h, 
    //                          int internal_pad_w, int in_h, int in_w, int in_h_after_pad, int in_w_after_pad, 
    //                          int in_c, int warp_num_per_block, int in_c_in_h_in_w, int in_h_in_w,
    //                          int tmp_h, int tmp_w)
    d_Inputfeaturemap_data = gradient_Inputfeaturemap.device_data();
    // warm up
    im2col_forward<<<grid_naive_im2col_gradient, warp_num_per_block*32>>>(d_Inputfeaturemap_data, 
            b_matrix_after_im2col, b_row, b_col, out_h+(out_h-1)*(str_h-1), out_w+(out_w-1)*(str_w-1), 
            gradient_out_h, gradient_out_w, 1, 1, 
            pad_h, pad_w, 0, 0, gradient_in_h, gradient_in_w,
            gradient_in_h+2*pad_h, gradient_in_w+2*pad_w, gradient_in_c, 
            warp_num_per_block, 
            gradient_in_c*gradient_in_h*gradient_in_w, gradient_in_h*gradient_in_w, 
            gradient_in_h + pad_h, gradient_in_w + pad_w
            );

#ifdef PRINT
    printf("Input: %d-%d-%d-%d.\n", gradient_in_n, gradient_in_c, gradient_in_h+2*pad_h, 
            gradient_in_w+2*pad_w);
    printf("Weight: %d-%d-%d-%d.\n",  gradient_filt_n, gradient_filt_c, out_h+(out_h-1)*(str_h-1), 
            out_w+(out_w-1)*(str_w-1));
    printf("filt_h, filt_w: %d, %d.\n", out_h+(out_h-1)*(str_h-1), out_w+(out_w-1)*(str_w-1));
    printf("out_h, out_w: %d, %d.\n", gradient_out_h, gradient_out_w);
    printf("external_pad_h, external_pad_w: %d, %d.\n", pad_h, pad_w);
    printf("in_h, in_w: %d, %d.\n", gradient_in_h, gradient_in_w);
    printf("in_h_after_pad, in_w_after_pad: %d, %d.\n", gradient_in_h+2*pad_h, gradient_in_w+2*pad_w);
    printf("%d-%d-%d-%d.\n", gradient_in_c*gradient_in_h*gradient_in_w, gradient_in_h*gradient_in_w,
          gradient_in_h + pad_h, gradient_in_w + pad_w);

#endif

    cudaEventRecord(events[0]);
    for(iter=0; iter<options.iterations; iter++){
        im2col_forward<<<grid_naive_im2col_gradient, warp_num_per_block*32>>>(d_Inputfeaturemap_data, 
            b_matrix_after_im2col, b_row, b_col, out_h+(out_h-1)*(str_h-1), out_w+(out_w-1)*(str_w-1), 
            gradient_out_h, gradient_out_w, 1, 1, 
            pad_h, pad_w, 0, 0, gradient_in_h, gradient_in_w,
            gradient_in_h+2*pad_h, gradient_in_w+2*pad_w, gradient_in_c, 
            warp_num_per_block, 
            gradient_in_c*gradient_in_h*gradient_in_w, gradient_in_h*gradient_in_w, 
            gradient_in_h + pad_h, gradient_in_w + pad_w
            );
    }
    cudaEventRecord(events[1]);

    cudaEventSynchronize(events[1]);
    cudaEventElapsedTime(&all_time, events[0], events[1]);
    
    result.runtime_ms += double(all_time) / double(options.iterations);
    result.gflops += 0;
    result.workspace += 0;

    // Copy the input featuremap map after im2col back to host.
    CUDA_CALL(cudaMallocHost(&host_input_featuremap_after_im2col, b_matrx_size * sizeof(float)));
    CUDA_CALL(cudaMemcpy(host_input_featuremap_after_im2col, b_matrix_after_im2col, 
              b_matrx_size * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef PRINT
    // PRINT is defined in config.h, to see host data of tensor_b after im2col.
    printf("\n###########Host After Im2col Start###########\n");
    for(size_t i = 0; i < b_row; ++i){
        for(size_t j = 0; j < b_col; ++j){
            printf("%.3f ", *(host_input_featuremap_after_im2col+
                i*b_col + j));
        }
        printf("\n");
    }
    printf("###########Host After Im2col End#############\n\n");
#endif

#ifdef DEBUG
    printf("\033[31mPRENO Gradient Im2col time: %lf ms.\033[0m\n", float(all_time)/options.iterations);
#endif

    ////////////////////////////GRADIENT MMA////////////////////////////
    m = gradient_filt_n;
    k = b_row;
    n = b_col;

    c_matrx_size = m * n;

    // Declare 2D C matrix and sync.
    Tensor2d<DTypeInput> c_matrix_gradient(c_matrx_size); 
    c_matrix_gradient.sync_device();

    // Call .device_data() to get device addr of C matrix.
    c_matrix_result = c_matrix_gradient.device_data();

    dim3 grid_mma_gradient((n + 127) / 128, (m + 127) / 128);

    // warmup
    sgemm_128x128x8_kernel<<<grid_mma_gradient, 256>>>(
        d_LossNextLayer_after_padding_data, b_matrix_after_im2col, c_matrix_result, m, n, k, 
        k * sizeof(float), n * sizeof(float) * 8);
    
    cudaEventRecord(events[0]);
    
    for(iter=0; iter<options.iterations; iter++){
        sgemm_128x128x8_kernel<<<grid_mma_gradient, 256>>>(
            d_LossNextLayer_after_padding_data, b_matrix_after_im2col, c_matrix_result, m, n, k, 
            k * sizeof(float), n * sizeof(float) * 8);
    }

    cudaEventRecord(events[1]);
    
    cudaEventSynchronize(events[1]);
    cudaEventElapsedTime(&all_time, events[0], events[1]);
    
    result.runtime_ms += double(all_time) / double(options.iterations);
    workload = m * n * k * 2;
    result.gflops += (double(workload) / 1e9) / (double(result.runtime_ms) / 1e3);
    result.workspace += 0;

    // printf("END...\n");

    // Copy the input featuremap map after im2col back to host.
    CUDA_CALL(cudaMallocHost(&host_output_featuremap_after_mma, gradient_filt_n * b_col * sizeof(float)));
    CUDA_CALL(cudaMemcpy(host_output_featuremap_after_mma, c_matrix_result, 
              gradient_filt_n * b_col * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef PRINT
    // PRINT is defined in config.h, to see host data of c_matrix_result after im2col.
    printf("\n###########Host MMA RESULT Start###########\n");
    for(size_t i = 0; i < m; ++i){
        for(size_t j = 0; j < n; ++j){
            printf("%.3f ", *(host_output_featuremap_after_mma+i*n + j));
        }
        printf("\n");
    }
    printf("###########Host MMA RESULT End#############\n\n");
#endif

#ifdef DEBUG
    printf("\033[31mPRENO Gradient MMA time: %lf ms.\033[0m\n", float(all_time)/options.iterations);
#endif

#ifdef DEBUG
    printf("\033[31mPRENO Gradient workload: %ld.\033[0m\n", workload);
    printf("\033[31mPRENO Gradient Total time: %lf ms.\033[0m\n", result.runtime_ms);
    printf("\033[31mPRENO Gradient Total gflops: %lf gflops.\033[0m\n", result.gflops);
    printf("\033[31mPRENO Gradient M = %d, K = %d, N = %d.\033[0m\n", m, k, n);
#endif
    //#######################GRADIENT END#########################
    
    return result;
}
