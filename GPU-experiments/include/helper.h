/*
 * @Author: Yangjie Zhou. https://zenodo.org/record/5535284
 * @Date: 2022-03-11 20:26:34
 * @LastEditTime: 2022-03-19 10:31:43
 * @LastEditors: ConvolutedDog
 * @Description: In User Settings Edit
 * @FilePath: /GPU-experiments/include/helper.h
 */

#ifndef HELPER_H
#define HELPER_H

#include "common.h"
#include "device_memory.h"

class Tensor4dParam{
    size_t N = 1;
    size_t C = 1;
    size_t H = 1;
    size_t W = 1;
    
public:
	void init(size_t n=0, size_t c=0, size_t h=0, size_t w=0){
		this->N = n;
        this->C = c;
		this->H = h;
		this->W = w;
	}
    Tensor4dParam() {}
    Tensor4dParam(size_t n, size_t c, size_t h, size_t w) :N(n), C(c), H(h), W(w){}
    size_t get_size() { return N * C * H * W; }
    size_t get_n() {return N;}
    size_t get_c() {return C;}
    size_t get_h() {return H;}
    size_t get_w() {return W;}
    ~Tensor4dParam() {}
};


class Tensor2dParam{
    size_t M = 1;
    size_t N = 1;

public:
	void init(size_t m=0, size_t n=0){
		this->M = m;
        this->N = n;
	}
    Tensor2dParam() {}
    Tensor2dParam(size_t m, size_t n) :M(m), N(n){}

    size_t get_size() { return M * N; }
    size_t get_M() { return M; }
    size_t get_N() { return N; }

    ~Tensor2dParam() {}

};

class Tensor3dParam{
    size_t M = 1;
    size_t K = 1;
    size_t N = 1;

public:
	void init(size_t m, size_t k, size_t n){
		this->M = m;
        this->K = k;
        this->N = n;
	}
    Tensor3dParam() {}
    Tensor3dParam(size_t m, size_t k, size_t n) :M(m), K(k), N(n){}

    size_t get_size() { return M * K * N; }
    size_t get_M() { return M; }
    size_t get_K() { return K; }
    size_t get_N() { return N; }

    ~Tensor3dParam() {}

};

class Options {
public:
    const int inputParamNum = 13;

    Tensor4dParam input;
    Tensor4dParam filter;
    Tensor4dParam output;
    Tensor4dParam padding; 
    Tensor2dParam conv_stride;
    Tensor2dParam dilation;

    Tensor3dParam gemm;

    size_t input_size;
    size_t filter_size;
    size_t output_size;
    size_t problem_size;

    std::string cudnn_log_file;
    std::string preprocess_log_file;
    std::string nopreprocess_log_file;
    std::ofstream cudnn_log;
    std::ofstream preprocess_log;
    std::ofstream nopreprocess_log;
    bool cudnn_log_bool;
    bool preprocess_log_bool;
    bool nopreprocess_log_bool;

    int my_algo; //???
    int iterations;

    bool run_cudnn;
    bool run_preprocess;
    bool run_nopreprocess;

    Options():
        input(1, 32, 32, 32),
        filter(32, 3, 3, 32),
        padding(0, 0, 0, 0),
        conv_stride(1, 1), //(1,1) no expand of filter
        dilation(1, 1),
        gemm(1,1,1),
        cudnn_log_bool(false),
        preprocess_log_bool(false),
        nopreprocess_log_bool(false),
        iterations(50),
        my_algo(0),
        run_cudnn(false),
        run_preprocess(false),
        run_nopreprocess(false)
        {}

    void parse(int argc, char const **argv);
    // void update(int padding_size = 16);
    void update();
    double gflops(double runtime_s);
};

class Result{
public:
    double runtime_ms;
    
    int64_t fmas;
    double gflops;
    bool run_success;
    bool check_success;
    size_t workspace;
    cudaError_t error;

    Result(): runtime_ms(0), gflops(0), run_success(true), check_success(true), error(cudaSuccess) {}
    ~Result() {}

    std::ostream & print_header(std::ostream &out, Options options) {
        out << "Layer,N,C,H,W,D,Kh,Kw,Runtime,GFLOPs";

        return out;
    }

    std::ostream & print_result(std::ostream &out, int layer_idx, Options options) {
        out 
        << "conv_" << layer_idx << ","
        << options.input.get_n() << ","
        << options.input.get_c() << ","
        << options.input.get_h() << ","
        << options.input.get_w() << ","
        << options.filter.get_n() << ","
        << options.filter.get_h() << ","
        << options.filter.get_w() << ","
        << runtime_ms << ","
        << gflops;

        return out;
    }

    void update_param(Options &options);
};

Result test_cudnn(Options &options);
Result test_preprocess(Options &options);
Result test_nopreprocess(Options &options);


template <
  typename DType_ 
>
class Tensor4d {
public:
    Tensor4dParam param_;
    size_t tensor_size; 

    std::vector<DType_> host_;
    device_memory::allocation<DType_> device_;


    Tensor4d() {}
    ~Tensor4d() {}

    Tensor4d(Tensor4dParam param): param_(param) {
        host_.clear();
        tensor_size = param_.get_size();
        device_.reset(param_.get_size());
    }

    Tensor4d(size_t tensor_size): tensor_size(tensor_size){
        param_.init();
        host_.clear();
        device_.reset(tensor_size);
    }

    void reset() {
        param_.init();
        host_.clear();
        device_.reset();
        tensor_size = 0;
    }

    void reset(Tensor4dParam param) {
        param_ = param;
        host_.clear();
        tensor_size = param.get_size();
        device_.reset(param.get_size());
    }

    bool device_backed() const {
        return (device_.get() == nullptr) ? false : true;
    }
  
    size_t size() const {
        return host_.size();
    }

    DType_ * host_data() { return host_.data(); }
    DType_ * device_data() { return device_.get(); }

    void init_host_data_random() {
        host_.clear();
        srand (time(NULL));
        size_t vec_size = param_.get_size();
        for(int i = 0; i < vec_size; i++){
            host_.push_back((DType_)rand() / RAND_MAX);
        }
    }

    void init_half_host_data_random(int max, int min) {
        host_.clear();
        srand (time(NULL));
        float tmp;
        size_t vec_size = param_.get_size();
        for(int i = 0; i < vec_size; i++){
            tmp = (min + rand() % (( max + 1 ) - min));
            host_.push_back(__float2half(tmp));
        }
    }
    void init_half_host_data_order() {
        host_.clear();
        size_t vec_size = param_.get_size();
        for(int i = 0; i < vec_size; i++){
            host_.push_back(__float2half((float)i/100));
        } 
    }

    void init_half_host_data_order_add_bias(float add) {
        host_.clear();
        size_t vec_size = param_.get_size();
        for(int i = 0; i < vec_size; i++){
            host_.push_back(__float2half((float)i/100 + add));
        } 
    }


    void init_half_host_data_static(float m) {
        host_.clear();
        size_t vec_size = param_.get_size();
        for(int i = 0; i < vec_size; i++){
            host_.push_back(__float2half((float)m));
        } 
    }

    void init_host_data_order() {
        host_.clear();
        size_t vec_size = param_.get_size();
        for(int i = 0; i < vec_size; i++){
            host_.push_back((DType_)i);
        } 
    }

    DType_ get_host_data_by_idx(size_t idx) {
        if(idx < host_.size())
            return host_.at(idx);
        else
            return DType_(0xffff);
    }

    void print_host_data() {
        for (auto i = host_.begin(); i != host_.end(); ++i) {
            std::cout << *i << ' ';
        }
        std::cout << std::endl;
    }

    void print_half_host_data() {
        for (auto i = host_.begin(); i != host_.end(); ++i) {
                std::cout << __half2float(*i) << ' ';
        }
        std::cout << std::endl;
    }

    void save_host_data(std::string file_name) {
        std::ofstream file_(file_name, std::ofstream::out | std::ofstream::app);
        for (auto i = host_.begin(); i != host_.end(); ++i) {
            file_ << *i << ' ';
        }
        file_ << std::endl;
        file_.close();
    }

    void save_half_host_data(std::string file_name) {
        std::ofstream file_(file_name, std::ofstream::out | std::ofstream::app);
        for (auto i = host_.begin(); i != host_.end(); ++i) {
            file_ << __half2float(*i) << ' ';
        }
        file_ << std::endl;
        file_.close();
    }

    void sync_host() {
        if (device_backed()) {
        device_memory::copy_to_host(
            host_data(), device_data(), size());
        }
    }

    void sync_device() {
        if (device_backed()) {
            device_memory::copy_to_device(
                device_data(), host_data(), size());
        }
    }
};



template <
  typename DType_ 
>
class Tensor2d {
public:
    Tensor2dParam param_;
    size_t tensor_size; 
    
    std::vector<DType_> host_;
    device_memory::allocation<DType_> device_;

    
    Tensor2d() {}
    ~Tensor2d() {}

    Tensor2d(Tensor2dParam param): param_(param) {
        host_.clear();
        tensor_size = param_.get_size();
        device_.reset(param_.get_size());
    }
    
    Tensor2d(size_t tensor_size): tensor_size(tensor_size){
        param_.init();
        host_.clear();
        device_.reset(tensor_size);
    }

    void reset() {
        param_.init();
        host_.clear();
        device_.reset();
        tensor_size = 0;
    }

    void reset(Tensor2dParam param) {
        param_ = param;
        host_.clear();
        tensor_size = param.get_size();
        device_.reset(param.get_size());
    }

    bool device_backed() const {
        return (device_.get() == nullptr) ? false : true;
    }
  
    size_t size() const {
        return host_.size();
    }

    DType_ * host_data() { return host_.data(); }
    DType_ * device_data() { return device_.get(); }
    
    void init_host_data_random() {
        host_.clear();
        srand (time(NULL));
        size_t vec_size = param_.get_size();
        for(int i = 0; i < vec_size; i++){
            host_.push_back((DType_)rand() / RAND_MAX);
        }
    }
    
    DType_ get_host_data_by_idx(size_t idx) {
        if(idx < host_.size())
            return host_.at(idx);
        else
            return DType_(0xffff);
    }

    void print_host_data() {
        for (auto i = host_.begin(); i != host_.end(); ++i) {
            std::cout << *i << ' ';
        }
        std::cout << std::endl;
    }

    void sync_host() {
        if (device_backed()) {
        device_memory::copy_to_host(
            host_data(), device_data(), size());
        }
    }

    void sync_device() {
        if (device_backed()) {
            device_memory::copy_to_device(
                device_data(), host_data(), size());
        }
    }
};






#endif