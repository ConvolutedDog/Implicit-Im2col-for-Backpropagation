/*
 * @Author: ConvolutedDog
 * @Date: 2022-03-11 21:39:43
 * @LastEditTime: 2022-03-19 18:05:07
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /GPU-Implicit/include/config.h
 */

#ifndef CONFIG_H
#define CONFIG_H

#include "common.h"

using DTypeInput = float;
using DTypeFilter = float;
using DTypeOutput = float;
using DTypeAccumulator = float;
using DtypeBias = float;

#define CUDNN_DType_Input CUDNN_DATA_FLOAT
#define CUDNN_DType_Filter CUDNN_DATA_FLOAT
#define CUDNN_DType_Output CUDNN_DATA_FLOAT
#define CUDNN_DType_OP CUDNN_DATA_FLOAT
#define CUDNN_MATH_TYPE CUDNN_TENSOR_OP_MATH


#define DEBUG
// #define PRINT

#define TEST_CUDNN
#define TEST_PREPROCESS
#define TEST_NOPREPROCESS

#endif 