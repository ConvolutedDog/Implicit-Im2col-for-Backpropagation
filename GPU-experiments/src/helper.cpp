/*
 * @Author: ConvolutedDog
 * @Date: 2022-03-11 20:25:14
 * @LastEditTime: 2022-03-19 09:22:41
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /GPU-Implicit/src/helper.cpp
 */

#include "../include/common.h"
#include "../include/helper.h"

void Options::parse(int argc, char const **argv) {

    if((std::stoi(argv[4])+2*std::stoi(argv[9])-std::stoi(argv[7])) % std::stoi(argv[10]) != 0 ||
       (std::stoi(argv[5])+2*std::stoi(argv[9])-std::stoi(argv[8])) % std::stoi(argv[10]) != 0){
        printf("Paras cannot be divided...\n");
        exit(0);
    }
               
    if(strcmp(argv[1],"cudnn") == 0) {
        run_cudnn = true;
        if(argc == inputParamNum){ 
            // input:    N*C*H*W
            // filter:   D*C*Kh*Kw
            // padding:  padding_h_up, padding_h_down, padding_w_left, padding_w_right
            // stride:   stride_h, stride_w
            // dilation: ...
            input.init(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
            filter.init(std::stoi(argv[6]), std::stoi(argv[3]), std::stoi(argv[7]), std::stoi(argv[8]));
            padding.init(std::stoi(argv[9]), std::stoi(argv[9]), std::stoi(argv[9]), std::stoi(argv[9]));
            conv_stride.init(std::stoi(argv[10]), std::stoi(argv[10]));
            dilation.init(std::stoi(argv[11]), std::stoi(argv[11]));

            cudnn_log_file = std::string(argv[12]);
            cudnn_log_bool = true;
        }
        else if(argc == inputParamNum - 1) {
            input.init(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
            filter.init(std::stoi(argv[6]), std::stoi(argv[3]), std::stoi(argv[7]), std::stoi(argv[8]));
            padding.init(std::stoi(argv[9]), std::stoi(argv[9]), std::stoi(argv[9]), std::stoi(argv[9]));
            conv_stride.init(std::stoi(argv[10]), std::stoi(argv[10]));
            dilation.init(std::stoi(argv[11]), std::stoi(argv[11]));

            cudnn_log_bool = false;
        }
        else {
            printf("Cudnn test\n");
            printf("The number of parameters should be %d instead of %d\n", inputParamNum, argc);
            printf("./a.out cudnn n cin h w cout fh fw padding stride dilated [cudnn_log_file]\n");
            exit(0);
        }
    }
    else if(strcmp(argv[1],"preprocess") == 0) {
        run_preprocess= true;
        if(argc == inputParamNum){ 
            // input:    N*C*H*W
            // filter:   D*C*Kh*Kw
            // padding:  padding_h_up, padding_h_down, padding_w_left, padding_w_right
            // stride:   stride_h, stride_w
            // dilation: ...
            input.init(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
            filter.init(std::stoi(argv[6]), std::stoi(argv[3]), std::stoi(argv[7]), std::stoi(argv[8]));
            padding.init(std::stoi(argv[9]), std::stoi(argv[9]), std::stoi(argv[9]), std::stoi(argv[9]));
            conv_stride.init(std::stoi(argv[10]), std::stoi(argv[10]));
            dilation.init(std::stoi(argv[11]), std::stoi(argv[11]));

            preprocess_log_file = std::string(argv[12]);
            preprocess_log_bool = true;
        }
        else if(argc == inputParamNum - 1) {
            input.init(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
            filter.init(std::stoi(argv[6]), std::stoi(argv[3]), std::stoi(argv[7]), std::stoi(argv[8]));
            padding.init(std::stoi(argv[9]), std::stoi(argv[9]), std::stoi(argv[9]), std::stoi(argv[9]));
            conv_stride.init(std::stoi(argv[10]), std::stoi(argv[10]));
            dilation.init(std::stoi(argv[11]), std::stoi(argv[11]));

            preprocess_log_bool = false;
        }
        else {
            printf("Preprocess test\n");
            printf("The number of parameters should be %d instead of %d\n", inputParamNum, argc);
            printf("./a.out preprocess n cin h w cout fh fw padding stride dilated [preprocess_log_file]\n");
            exit(0);
        }
    }
    else if(strcmp(argv[1],"nopreprocess") == 0) {
        run_nopreprocess= true;
        if(argc == inputParamNum){ 
            // input:    N*C*H*W
            // filter:   D*C*Kh*Kw
            // padding:  padding_h_up, padding_h_down, padding_w_left, padding_w_right
            // stride:   stride_h, stride_w
            // dilation: ...
            input.init(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
            filter.init(std::stoi(argv[6]), std::stoi(argv[3]), std::stoi(argv[7]), std::stoi(argv[8]));
            padding.init(std::stoi(argv[9]), std::stoi(argv[9]), std::stoi(argv[9]), std::stoi(argv[9]));
            conv_stride.init(std::stoi(argv[10]), std::stoi(argv[10]));
            dilation.init(std::stoi(argv[11]), std::stoi(argv[11]));

            nopreprocess_log_file = std::string(argv[12]);
            nopreprocess_log_bool = true;
        }
        else if(argc == inputParamNum - 1) {
            input.init(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
            filter.init(std::stoi(argv[6]), std::stoi(argv[3]), std::stoi(argv[7]), std::stoi(argv[8]));
            padding.init(std::stoi(argv[9]), std::stoi(argv[9]), std::stoi(argv[9]), std::stoi(argv[9]));
            conv_stride.init(std::stoi(argv[10]), std::stoi(argv[10]));
            dilation.init(std::stoi(argv[11]), std::stoi(argv[11]));

            nopreprocess_log_bool = false;
        }
        else {
            printf("Nopreprocess test\n");
            printf("The number of parameters should be %d instead of %d\n", inputParamNum, argc);
            printf("./a.out preprocess n cin h w cout fh fw padding stride dilated [nopreprocess_log_file]\n");
            exit(0);
        }
    }
    else { //TODO
        printf("The number of parameters should be %d instead of %d\n", inputParamNum, argc);
        printf("./a.out cudnn/preprocess/nopreprocess/all n cin h w cout fh fw padding stride dilated *_log_file\n");
        printf("argv[1]:%s\n", argv[1]);
        exit(0);
    }

}


void Options::update(){
        filter_size = filter.get_size();

        size_t IC,KH,KW,ON,OC,OH,OW;
        IC = input.get_c();
        KH = filter.get_h();
        KW = filter.get_w();
        ON = input.get_n();
        OC = filter.get_n();
        OH = (input.get_h()+padding.get_n()+padding.get_c() - filter.get_h()) / conv_stride.get_M() + 1;
        OW = (input.get_w()+padding.get_h()+padding.get_w() - filter.get_w()) / conv_stride.get_N() + 1;
        
        output.init(ON, OC, OH, OW); 

        size_t gemm_m = OC;
        size_t gemm_k = IC * KH * KW;
        size_t gemm_n = ON * OH * OW;

        gemm.init(gemm_m, gemm_k, gemm_n);
        problem_size = gemm.get_size();
}

double Options::gflops(double runtime_s){
    int64_t fmas = problem_size;
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
}