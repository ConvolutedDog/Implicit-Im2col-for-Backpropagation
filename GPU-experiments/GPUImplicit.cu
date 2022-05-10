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

#include "include/common.h"
#include "include/helper.h"
#include "include/config.h"

using namespace std;

int device_shared_limit;
int multiProcessorCount;


int main(int argc, char const *argv[]){

    printf("\033[32mInitializing...\033[0m\n");
    // findCudaDevice return id of your GPU
    int dev = findCudaDevice(argc, (const char **)argv);

    printf("\033[32mGpu Info...\033[0m\n");
    printf("\033[31mUsing device id: %d.\033[0m\n", dev);
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\033[31mCUDACC_VER_MAJOR: %d.\033[0m\n", __CUDACC_VER_MAJOR__);
    printf("\033[31mmaxThreadsPerBlock: %d.\033[0m\n", deviceProp.maxThreadsPerBlock);
    // shared memory per MP
    device_shared_limit = deviceProp.sharedMemPerMultiprocessor;
    // multiProcessorCount is count of multi-processors (MP) on GPU
    multiProcessorCount = deviceProp.multiProcessorCount;
    printf("\033[31mdevice_shared_limit: %d.\033[0m\n", device_shared_limit);
    printf("\033[31mmultiProcessorCount: %d.\033[0m\n", multiProcessorCount);
    Options options;
    options.parse(argc, argv);
    
#ifdef TEST_CUDNN
    options.update();
    if(options.run_cudnn) {
        printf("\033[32m\n==========CUDNN Test...==========\n\033[0m");
        Result cudnn_result = test_cudnn(options);
        printf("cudnn_result.runtime_ms: %f ms.\n", cudnn_result.runtime_ms);
        if(options.cudnn_log_bool == true){
            options.cudnn_log.open(options.cudnn_log_file);
            options.cudnn_log << "cudnn_result.runtime_ms: " << cudnn_result.runtime_ms << std::endl;
            options.cudnn_log << "cudnn_result.gflops: " << cudnn_result.gflops << std::endl;
            options.cudnn_log << "cudnn_result.workspace: " << cudnn_result.workspace << std::endl;
            options.cudnn_log.close();
        }
    }
#endif

#ifdef TEST_PREPROCESS
    options.update();
    if(options.run_preprocess) {
        printf("\033[32m\n==========PREPROCESS Test...==========\033[0m");
        Result preprocess_result = test_preprocess(options);
        if(options.preprocess_log_bool == true){
            options.preprocess_log.open(options.preprocess_log_file);
            options.preprocess_log << "preprocess_result.runtime_ms: " << preprocess_result.runtime_ms << std::endl;
            options.preprocess_log << "preprocess_result.gflops: " << preprocess_result.gflops << std::endl;
            options.preprocess_log << "preprocess_result.workspace: " << preprocess_result.workspace << std::endl;

            options.preprocess_log.close();
        }
    }
#endif

#ifdef TEST_NOPREPROCESS
    options.update();
    if(options.run_nopreprocess) {
        printf("\033[32m\n==========NOPREPROCESS Test...==========\033[0m");
        Result nopreprocess_result = test_nopreprocess(options);
        if(options.nopreprocess_log_bool == true){
            options.nopreprocess_log.open(options.nopreprocess_log_file);
            options.nopreprocess_log << "nopreprocess_result.runtime_ms: " << nopreprocess_result.runtime_ms << std::endl;
            options.nopreprocess_log << "nopreprocess_result.gflops: " << nopreprocess_result.gflops << std::endl;
            options.nopreprocess_log << "nopreprocess_result.workspace: " << nopreprocess_result.workspace << std::endl;

            options.nopreprocess_log.close();
        }
    }
#endif

    return 0;
}