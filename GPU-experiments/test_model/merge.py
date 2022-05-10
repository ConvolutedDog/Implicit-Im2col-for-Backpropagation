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

Network_list = ["AlexNet", "GoogleNet", "Overfeat", "ResNet18", "VGG", "YOLO", "ZFNet", "DenseNet"]
result_dir = "./result"
def merge_layer():

    merge_file = open(result_dir + "Network_layer_Result.csv", "w")

    dict_writer = csv.DictWriter(merge_file, fieldnames=["Network",
    "Layer_no", "N", "H", "W", "Cin", "Cout", "Fh", "Fw", "Stride", "padding", 
        "cudnn_time(ms)", "cudnn_GFLOPS", "cudnn_workspace", 
        "my_time", "my_GFLOPS", "my_workspace", "my_algo", 
        "my_time_2", "my_GFLOPS_2", "my_workspace_2", "my_algo_2", 
        "my_time_3", "my_GFLOPS_3", "my_workspace_3", "my_algo_3", 
        "my_time_4", "my_GFLOPS_4", "my_workspace_4", "my_algo_4", 
        "my_time_5", "my_GFLOPS_5", "my_workspace_5", "my_algo_5", 
        "my_time_6", "my_GFLOPS_6", "my_workspace_6", "my_algo_6", 
        "my_time_update", "my_GFLOPS_update", "my_workspace_update", "my_algo_update", 
        "cudnn_padding_out_time", "cudnn_padding_out_GFLOPS", "cudnn_padding_out_workspace",
        "cudnn_padding_in_time", "cudnn_padding_in_GFLOPS", "cudnn_padding_in_workspace",
        "cudnn_padding_w_time", "cudnn_padding_w_GFLOPS", "cudnn_padding_w_workspace",
        "cudnn_padding_in_out_time", "cudnn_padding_in_out_GFLOPS", "cudnn_padding_in_out_workspace",
        "cudnn_padding_all_time", "cudnn_padding_all_GFLOPS", "cudnn_padding_all_workspace"
        ])


    dict_writer.writeheader()

    for Network in Network_list:
            result_file = open(result_dir + Network + ".csv", "r")
            dict_reader = csv.DictReader(result_file)
            for row in dict_reader:
                row["Network"] = Network
                dict_writer.writerow(row)

if __name__ == '__main__':
    merge_layer()
    