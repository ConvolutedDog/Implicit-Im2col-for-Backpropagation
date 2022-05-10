# Copyright (c) 2021, Yangjie Zhou.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
from math import *
from  multiprocessing import Process,Pool
import gc
import numpy as np
import os

if __name__ == '__main__':
    DEVICE_NO = str(1)
    config_dir = "./config/"
    my_implicit = " ../build/backup "



    result_dir = "./result_accu_fp16/"
    os.system("mkdir " + result_dir)

    log_dir = "./log/"
    os.system("mkdir " + log_dir)
    cudnn_log_dir = log_dir + "cudnn/"
    my_log_dir = log_dir + "my_implicit/"
    cudnn_padding_cout_log_dir = log_dir + "cudnn_padding_cout/"
    cudnn_padding_cin_log_dir = log_dir + "cudnn_padding_cin/"
    cudnn_padding_cin_out_log_dir = log_dir + "cudnn_padding_cin_out/"
    cudnn_padding_all_log_dir = log_dir + "cudnn_padding_all/"

    os.system("mkdir " + cudnn_log_dir)
    os.system("mkdir " + my_log_dir)
    os.system("mkdir " + cudnn_padding_cout_log_dir)
    os.system("mkdir " + cudnn_padding_cin_log_dir)
    os.system("mkdir " + cudnn_padding_cin_out_log_dir)
    os.system("mkdir " + cudnn_padding_all_log_dir)

    for Network in ["AlexNet", "ResNet18", "GoogleNet", "Overfeat", "VGG", "YOLO", "ZFNet", "DenseNet"]:

        result =  Network + ".csv"
        result_file = open(result_dir + result, "w")
        

        dict_writer = csv.DictWriter(result_file, fieldnames=["Layer_no", "N", "H", "W", "Cin", "Cout", "Fh", "Fw", "Stride", "padding", 
        "cudnn_time(ms)", "cudnn_GFLOPS", "cudnn_workspace", 
        "my_time", "my_GFLOPS", "my_workspace", "my_algo", 
        "my_time_update", "my_GFLOPS_update", "my_workspace_update", "my_algo_update", 
        "cudnn_padding_out_time", "cudnn_padding_out_GFLOPS", "cudnn_padding_out_workspace",
        "cudnn_padding_in_time", "cudnn_padding_in_GFLOPS", "cudnn_padding_in_workspace",
        "cudnn_padding_in_out_time", "cudnn_padding_in_out_GFLOPS", "cudnn_padding_in_out_workspace",
        "cudnn_padding_all_time", "cudnn_padding_all_GFLOPS", "cudnn_padding_all_workspace"]
        )
        dict_writer.writeheader()

        os.system("rm " + cudnn_log_dir + Network + " -rf")
        os.system("rm " + my_log_dir + Network + " -rf")

        os.system("mkdir " + cudnn_log_dir + Network)
        os.system("mkdir " + my_log_dir + Network)


        os.system("rm " + cudnn_padding_cout_log_dir + Network + " -rf")
        os.system("mkdir " + cudnn_padding_cout_log_dir + Network)
        os.system("rm " + cudnn_padding_cin_log_dir + Network + " -rf")
        os.system("mkdir " + cudnn_padding_cin_log_dir + Network)
        os.system("rm " + cudnn_padding_cin_out_log_dir + Network + " -rf")
        os.system("mkdir " + cudnn_padding_cin_out_log_dir + Network)
        os.system("rm " + cudnn_padding_all_log_dir + Network + " -rf")
        os.system("mkdir " + cudnn_padding_all_log_dir + Network)
        
        
        config_file = "config_"+ Network +".csv"
        config = open(config_dir + "/" + config_file, "r")
        

        layer_no = 0
    
        N = [8, 16, 32, 64, 128, 256]

        for n_ in N:
            print(n_)
            config = open(config_dir + "/" + config_file, "r")
            dict_reader = csv.DictReader(config)
            for row in dict_reader:
                if (row["type"] == "convolution layer"):
                    h_ = int(row["H"])
                    w_ = int(row["W"])
                    k_ = int(row["Fh"])
                    cin_ = int(row["Cin"])
                    cout_ = int(row["Cout"])
                    pad_ = int(row["padding"])
                    stride_ = int(row["stride"])
                    dilated = 1

                    h_ = h_ + 2 * pad_
                    w_ = w_ + 2 * pad_
                    pad_ = 0

                    result_dict = {}
                    result_dict["Layer_no"] = layer_no
                    result_dict["N"] = n_
                    result_dict["H"] = h_
                    result_dict["W"] = w_
                    result_dict["Cin"] = cin_
                    result_dict["Cout"] = cout_
                    result_dict["Fh"] = k_
                    result_dict["Fw"] = k_
                    result_dict["Stride"] = stride_
                    result_dict["padding"] = pad_

                    cin_cudnn = cin_
                    if(cin_cudnn < 8):
                        cin_cudnn = 8
                    cout_cudnn = cout_
                    if(cout_cudnn < 8):
                        cout_cudnn = 8



                    cudnn_log_file = cudnn_log_dir + Network + "/" + str(layer_no)
                    my_log_file = my_log_dir + Network + "/" + str(layer_no)

                    cudnn_padding_cout_log_file = cudnn_padding_cout_log_dir + Network + "/" + str(layer_no)
                    cudnn_padding_cin_log_file = cudnn_padding_cin_log_dir + Network + "/" + str(layer_no)
                    cudnn_padding_in_out_log_file = cudnn_padding_cin_out_log_dir + Network + "/" + str(layer_no)
                    cudnn_padding_all_log_file = cudnn_padding_all_log_dir + Network + "/" + str(layer_no)

                    os.system("CUDA_VISIBLE_DEVICES=" + DEVICE_NO +  my_implicit + "cudnn " + str(n_) + " " +  str(h_) + " " +  str(w_) + " " + str(cin_cudnn) + " " +  str(cout_cudnn) + " " + str(k_) + " " + str(k_) + " " +  str(pad_) + " " + str(stride_) + " " + str(dilated) + " " +  cudnn_log_file);     

                    print("CUDA_VISIBLE_DEVICES=" + DEVICE_NO +  my_implicit + "cudnn " + str(n_) + " " +  str(h_) + " " +  str(w_) + " " + str(cin_) + " " +  str(cout_) + " " + str(k_) + " " + str(k_) + " " +  str(pad_) + " " + str(stride_) + " " + str(dilated) + " " +  cudnn_log_file)

                    if(os.path.exists(cudnn_log_file)):
                        with open(cudnn_log_file,'r') as f:
                            lines=f.readlines()
                            if(len(lines) > 2):
                                cudnn_time = lines[0].strip('\n')
                                cudnn_gflos = lines[1].strip('\n')
                                cudnn_workspace = lines[2].strip('\n')
                            else:
                                cudnn_time = -1
                                cudnn_gflos = -1
                                cudnn_workspace = -1
                    else:
                        cudnn_time = -1
                        cudnn_gflos = -1
                        cudnn_workspace = -1 
                            
                    result_dict["cudnn_time(ms)"] = cudnn_time
                    result_dict["cudnn_GFLOPS"] = cudnn_gflos
                    result_dict["cudnn_workspace"] = cudnn_workspace



                    my_cin = cin_
                    if(my_cin % 32 != 0):
                        my_cin = (floor((my_cin + 31) / 32)) * 32
                    
                    my_cout = cout_
                    if(my_cout % 128 != 0):
                        my_cout = (floor((my_cout + 127) / 128)) * 128


                    ow_ = floor((w_ - k_)/stride_) + 1
                    ow_ = (floor((ow_ + 15)/16)) * 16
                    my_h_ = my_w_ = (ow_ - 1) * stride_ + k_ 

                    os.system("CUDA_VISIBLE_DEVICES=" + DEVICE_NO +  my_implicit + "my_implicit " + str(n_) + " " +  str(my_h_) + " " +  str(my_w_) + " " + str(my_cin) + " " +  str(my_cout) + " " + str(k_) + " " + str(k_) + " " +  str(pad_) + " " + str(stride_) + " " + str(dilated) + " " + my_log_file);     
                    print("CUDA_VISIBLE_DEVICES=" + DEVICE_NO +  my_implicit + "my_implicit " + str(n_) + " " +  str(my_h_) + " " +  str(my_w_) + " " + str(my_cin) + " " +  str(my_cout) + " " + str(k_) + " " + str(k_) + " " +  str(pad_) + " " + str(stride_) + " " + str(dilated) + " " + my_log_file)

                    ######
                    if(os.path.exists(my_log_file)):
                        with open(my_log_file,'r') as f:
                            lines=f.readlines()
                            if(len(lines) > 7):
                                my_time = lines[0].strip('\n')
                                my_gflos = lines[1].strip('\n')
                                my_workspace = lines[2].strip('\n')
                                my_algo = lines[3].strip('\n')

                                my_time_update = lines[4].strip('\n')
                                my_gflos_update = lines[5].strip('\n')
                                my_workspace_update = lines[6].strip('\n')
                                my_algo_update = lines[7].strip('\n')
                            
                            elif(len(lines) > 3):
                                my_time = lines[0].strip('\n')
                                my_gflos = lines[1].strip('\n')
                                my_workspace = lines[2].strip('\n')
                                my_algo = lines[3].strip('\n')

                                my_time_update = -1
                                my_gflos_update = -1
                                my_workspace_update = -1
                                my_algo_update = -1
                            
                            else:
                                my_time = -1
                                my_gflos = -1
                                my_workspace = -1
                                my_algo = -1

                                my_time_update = -1
                                my_gflos_update = -1
                                my_workspace_update = -1
                                my_algo_update = -1

                    else:
                        my_time = -1
                        my_gflos = -1
                        my_workspace = -1
                        my_algo = -1

                        my_time_update = -1
                        my_gflos_update = -1
                        my_workspace_update = -1
                        my_algo_update = -1
                            
                    result_dict["my_time"] = my_time
                    result_dict["my_GFLOPS"] = my_gflos
                    result_dict["my_workspace"] = my_workspace
                    result_dict["my_algo"] = my_algo

                    result_dict["my_time_update"] = my_time_update
                    result_dict["my_GFLOPS_update"] = my_gflos_update
                    result_dict["my_workspace_update"] = my_workspace_update
                    result_dict["my_algo_update"] = my_algo_update

                    
                    #####
                    #just padding cout

                    os.system("CUDA_VISIBLE_DEVICES=" + DEVICE_NO +  my_implicit + "cudnn " + str(n_) + " " +  str(h_) + " " +  str(w_) + " " + str(cin_cudnn) + " " +  str(my_cout) + " " + str(k_) + " " + str(k_) + " " +  str(pad_) + " " + str(stride_) + " " + str(dilated) + " " +  cudnn_padding_cout_log_file);     

                    print("CUDA_VISIBLE_DEVICES=" + DEVICE_NO +  my_implicit + "cudnn " + str(n_) + " " +  str(h_) + " " +  str(w_) + " " + str(cin_cudnn) + " " +  str(my_cout) + " " + str(k_) + " " + str(k_) + " " +  str(pad_) + " " + str(stride_) + " " + str(dilated) + " " +  cudnn_padding_cout_log_file)

                    if(os.path.exists(cudnn_padding_cout_log_file)):
                        with open(cudnn_padding_cout_log_file,'r') as f:
                            lines=f.readlines()
                            if(len(lines) > 2):
                                cudnn_padding_cout_time = lines[0].strip('\n')
                                cudnn_padding_cout_gflos = lines[1].strip('\n')
                                cudnn_padding_cout_workspace = lines[2].strip('\n')
                            else:
                                cudnn_padding_cout_time = -1
                                cudnn_padding_cout_gflos = -1
                                cudnn_padding_cout_workspace = -1
                    else:
                        cudnn_padding_cout_time = -1
                        cudnn_padding_cout_gflos = -1
                        cudnn_padding_cout_workspace = -1
                            
                    result_dict["cudnn_padding_out_time"] = cudnn_padding_cout_time
                    result_dict["cudnn_padding_out_GFLOPS"] = cudnn_padding_cout_gflos
                    result_dict["cudnn_padding_out_workspace"] = cudnn_padding_cout_workspace


                    #####
                    #just padding cin

                    os.system("CUDA_VISIBLE_DEVICES=" + DEVICE_NO +  my_implicit + "cudnn " + str(n_) + " " +  str(h_) + " " +  str(w_) + " " + str(my_cin) + " " +  str(cout_cudnn) + " " + str(k_) + " " + str(k_) + " " +  str(pad_) + " " + str(stride_) + " " + str(dilated) + " " +  cudnn_padding_cin_log_file);     

                    print("CUDA_VISIBLE_DEVICES=" + DEVICE_NO +  my_implicit + "cudnn " + str(n_) + " " +  str(h_) + " " +  str(w_) + " " + str(my_cin) + " " +  str(cout_cudnn) + " " + str(k_) + " " + str(k_) + " " +  str(pad_) + " " + str(stride_) + " " + str(dilated) + " " +  cudnn_padding_cin_log_file)

                    if(os.path.exists(cudnn_padding_cin_log_file)):
                        with open(cudnn_padding_cin_log_file,'r') as f:
                            lines=f.readlines()
                            if(len(lines) > 2):
                                cudnn_padding_in_time = lines[0].strip('\n')
                                cudnn_padding_in_GFLOPS = lines[1].strip('\n')
                                cudnn_padding_in_workspace = lines[2].strip('\n')
                            else:
                                cudnn_padding_in_time = -1
                                cudnn_padding_in_GFLOPS = -1
                                cudnn_padding_in_workspace = -1
                    else:
                        cudnn_padding_in_time = -1
                        cudnn_padding_in_GFLOPS = -1
                        cudnn_padding_in_workspace = -1
                            
                    result_dict["cudnn_padding_in_time"] = cudnn_padding_in_time
                    result_dict["cudnn_padding_in_GFLOPS"] = cudnn_padding_in_GFLOPS
                    result_dict["cudnn_padding_in_workspace"] = cudnn_padding_in_workspace

                    #####
                    #padding cin & cout

                    os.system("CUDA_VISIBLE_DEVICES=" + DEVICE_NO +  my_implicit + "cudnn " + str(n_) + " " +  str(h_) + " " +  str(w_) + " " + str(my_cin) + " " +  str(my_cout) + " " + str(k_) + " " + str(k_) + " " +  str(pad_) + " " + str(stride_) + " " + str(dilated) + " " +  cudnn_padding_in_out_log_file);     

                    print("CUDA_VISIBLE_DEVICES=" + DEVICE_NO +  my_implicit + "cudnn " + str(n_) + " " +  str(h_) + " " +  str(w_) + " " + str(my_cin) + " " +  str(my_cout) + " " + str(k_) + " " + str(k_) + " " +  str(pad_) + " " + str(stride_) + " " + str(dilated) + " " +  cudnn_padding_in_out_log_file)

                    if(os.path.exists(cudnn_padding_in_out_log_file)):
                        with open(cudnn_padding_in_out_log_file,'r') as f:
                            lines=f.readlines()
                            if(len(lines) > 2):
                                cudnn_padding_in_out_time = lines[0].strip('\n')
                                cudnn_padding_in_out_GFLOPS = lines[1].strip('\n')
                                cudnn_padding_in_out_workspace = lines[2].strip('\n')
                            else:
                                cudnn_padding_in_out_time = -1
                                cudnn_padding_in_out_GFLOPS = -1
                                cudnn_padding_in_out_workspace = -1
                    else:
                        cudnn_padding_in_out_time = -1
                        cudnn_padding_in_out_GFLOPS = -1
                        cudnn_padding_in_out_workspace = -1
                            
                    result_dict["cudnn_padding_in_out_time"] = cudnn_padding_in_out_time
                    result_dict["cudnn_padding_in_out_GFLOPS"] = cudnn_padding_in_out_GFLOPS
                    result_dict["cudnn_padding_in_out_workspace"] = cudnn_padding_in_out_workspace

                    #####
                    #padding cin & cout & w

                    os.system("CUDA_VISIBLE_DEVICES=" + DEVICE_NO +  my_implicit + "cudnn " + str(n_) + " " +  str(my_h_) + " " +  str(my_h_) + " " + str(my_cin) + " " +  str(my_cout) + " " + str(k_) + " " + str(k_) + " " +  str(pad_) + " " + str(stride_) + " " + str(dilated) + " " +  cudnn_padding_all_log_file);     

                    print("CUDA_VISIBLE_DEVICES=" + DEVICE_NO +  my_implicit + "cudnn " + str(n_) + " " +  str(my_h_) + " " +  str(my_h_) + " " + str(my_cin) + " " +  str(my_cout) + " " + str(k_) + " " + str(k_) + " " +  str(pad_) + " " + str(stride_) + " " + str(dilated) + " " +  cudnn_padding_all_log_file)

                    if(os.path.exists(cudnn_padding_all_log_file)):
                        with open(cudnn_padding_all_log_file,'r') as f:
                            lines=f.readlines()
                            if(len(lines) > 2):
                                cudnn_padding_all_time = lines[0].strip('\n')
                                cudnn_padding_all_GFLOPS = lines[1].strip('\n')
                                cudnn_padding_all_workspace = lines[2].strip('\n')
                            else:
                                cudnn_padding_all_time = -1
                                cudnn_padding_all_GFLOPS = -1
                                cudnn_padding_all_workspace = -1
                    else:
                        cudnn_padding_all_time = -1
                        cudnn_padding_all_GFLOPS = -1
                        cudnn_padding_all_workspace = -1
                            
                    result_dict["cudnn_padding_all_time"] = cudnn_padding_all_time
                    result_dict["cudnn_padding_all_GFLOPS"] = cudnn_padding_all_GFLOPS
                    result_dict["cudnn_padding_all_workspace"] = cudnn_padding_all_workspace

                    print("Layer " + str(layer_no) + " success")
                    layer_no = layer_no + 1
                    dict_writer.writerow(result_dict)

                    # os.exit(1)
           
            config.close()

        result_file.close()
        print("result csv: " +  result_dir + result)
