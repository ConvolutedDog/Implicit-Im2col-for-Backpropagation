#  Copyright 2022 ConvolutedDog (https://github.com/ConvolutedDog/)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

#!/usr/bin/python3
import csv
import torch
import torchvision as tv
from torchsummary import summary

def calculate_sparsity(original_H, original_W, outer_pad_H_up, outer_pad_H_down, outer_pad_W_left, outer_pad_W_right, \
                       internal_pad_H, internal_pad_W, channel, Ksize_H, Ksize_W, Stride_h, Stride_w, allpixels, debug):
    
    _H = original_H + (original_H-1)*internal_pad_H
    _W = original_W + (original_W-1)*internal_pad_W
    __H = original_H + (original_H-1)*internal_pad_H + outer_pad_H_up + outer_pad_H_down
    __W = original_W + (original_W-1)*internal_pad_W + outer_pad_W_left + outer_pad_W_right

    ZeroNum = 0
    for h in range(__H):
        for w in range(__W):
            # location of pixel (h,w)
            covered_h = 0
            covered_w = 0
            for i in range(Ksize_H):
                if h >= i and h <= __H-(Ksize_H-i) and (h-i)%Stride_h == 0:
                    covered_h += 1
            for j in range(Ksize_W):
                if w >= j and w <= __W-(Ksize_W-j) and (w-j)%Stride_w == 0:
                    covered_w += 1
            
            if h < outer_pad_H_up or w < outer_pad_W_left or (h-outer_pad_H_up)%(internal_pad_H+1) > 0 or (w-outer_pad_W_left)%(internal_pad_W+1) > 0\
                or h >= outer_pad_H_up+_H or w >= outer_pad_W_left+_W:
                isZero = 1
                # 1,3
            else:
                isZero = 0
            if debug == 1:
                print('location: ', h, w, end=' | ')
                print('\tcovered: ', covered_h, covered_w, end=' | ')
                print('\tisZero: ', isZero)
            ZeroNum += isZero*channel*covered_h*covered_w
    
    return ZeroNum/allpixels, ZeroNum


def addrSparseForward(image_size_h, image_size_w, inchannel, outchannel, ksize_h, ksize_w, \
                      pad_h_up, pad_h_down, pad_w_left, pad_w_right, stride_h, stride_w):
    print('\tForward...')
    outputsize_h = int((image_size_h+pad_h_up+pad_h_down-ksize_h)/stride_h+1)
    outputsize_w = int((image_size_w+pad_w_left+pad_w_right-ksize_w)/stride_w+1)
    b_col = outputsize_h*outputsize_w
    b_row = inchannel*ksize_h*ksize_w
    last_B = [[0 for i in range(b_col)] for j in range(b_row)]
    image = [[[0 for i in range(image_size_w)] for j in range(image_size_h)] for k in range(inchannel)]
    pixel = 1
    for k in range(inchannel):
        pixel = 1
        for j in range(image_size_h):
            for i in range(image_size_w):
                image[k][j][i] = pixel
                pixel += 1

    # print('IMAGE BEFORE PADDING>>>>>>')
    # for i in range(inchannel):
    #    for j in range(len(image[i])):
    #        print(image[i][j])
    #    print('--------------')
    # print('<<<<<<<<<<<<<<<<<<<<<<<<<<')

    # padding
    for i in range(inchannel):
        for _ in range(pad_h_up):
            image[i].insert(0, [0 for j in range(len(image[i][0]))])
    for i in range(inchannel):
        for _ in range(pad_h_down):
            image[i].append([0 for j in range(len(image[i][0]))])
    for i in range(inchannel):
        for j in range(len(image[i])):
            for _ in range(pad_w_left):
                image[i][j].insert(0, 0)
    for i in range(inchannel):
        for j in range(len(image[i])):
            for _ in range(pad_w_right):
                image[i][j].append(0)
    
    # print('IMAGE AFTER PADDING>>>>>>>')
    # for i in range(inchannel):
    #     for j in range(len(image[i])):
    #         print(image[i][j])
    #     print('--------------')
    # print('<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    # img2col
    prefetch_length = image_size_h*image_size_w
    for r in range(b_row):
        if r // (ksize_h*ksize_w) >= 1.:
            prefetch_length_t = prefetch_length
        else:
            prefetch_length_t = 0
        for c in range(b_col):
            pixel_in_channel_of_image = r // (ksize_h*ksize_w)
            pixel_row_in_outimage = int(c // outputsize_w) % outputsize_h
            pixel_col_in_outimage = c % outputsize_w
            pixel_row_in_window = int(r // ksize_w) % ksize_h
            pixel_col_in_window = r % ksize_w
            pixel_row_in_image = pixel_row_in_outimage*stride_h+pixel_row_in_window
            pixel_col_in_image = pixel_col_in_outimage*stride_w+pixel_col_in_window
            if image[pixel_in_channel_of_image][pixel_row_in_image][pixel_col_in_image] == 0:
                last_B[r][c] = 0
            else:
                last_B[r][c] = image[pixel_in_channel_of_image][pixel_row_in_image][pixel_col_in_image] + pixel_in_channel_of_image*prefetch_length_t
    
    # print('MATRIX B>>>>>>>>>>>>>>>>>>')
    # for r in range(len(last_B)):
    #     print(last_B[r])
    # print('<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    zero_num = 0
    for r in range(len(last_B)):
        for c in range(len(last_B[r])):
            if last_B[r][c] == 0:
                zero_num += 1
    print('\t\tzero_rate: ', zero_num/(len(last_B)*len(last_B[0])))

    ZeroNumRate_after_im2col = \
        calculate_sparsity(image_size_h, image_size_w, pad_h_up, pad_h_down, pad_w_left, pad_w_right, 0, \
                           0, inchannel, ksize_h, ksize_w, stride_h, stride_w, len(last_B)*len(last_B[0]), debug=0)
    print('\t\tzero_rate: ', ZeroNumRate_after_im2col[0])
    
    if zero_num == ZeroNumRate_after_im2col[1]:
        print('\t\tRight!!!')
    else:
        print('\t\t\033[0;31;40mError!!!\033[0m')

    return last_B, len(last_B)*len(last_B[0]), zero_num, ZeroNumRate_after_im2col[0]

def addrSparseLOSS(image_size_h, image_size_w, inchannel, outchannel, ksize_h, ksize_w, \
                   pad_h_up, pad_h_down, pad_w_left, pad_w_right, stride_h, stride_w):
    print('\tLoss...')
    forward_outputsize_h = int((image_size_h+pad_h_up+pad_h_down-ksize_h)/stride_h+1)
    forward_outputsize_w = int((image_size_w+pad_w_left+pad_w_right-ksize_w)/stride_w+1)
    _outputsize_h = forward_outputsize_h+(forward_outputsize_h-1)*(stride_h-1)
    _outputsize_w = forward_outputsize_w+(forward_outputsize_w-1)*(stride_w-1)
    
    outputsize_h = image_size_h
    outputsize_w = image_size_w

    image_size_h = _outputsize_h
    image_size_w = _outputsize_w

    b_col = outputsize_h*outputsize_w
    b_row = outchannel*ksize_h*ksize_w
    last_B = [[0 for i in range(b_col)] for j in range(b_row)]
    image = [[[0 for i in range(image_size_w)] for j in range(image_size_h)] for k in range(outchannel)]
    pixel = 1
    for k in range(outchannel):
        pixel = 1
        for j in range(image_size_h):
            for i in range(image_size_w):
                if j % stride_h == 0 and i % stride_w == 0:
                    image[k][j][i] = pixel
                    pixel += 1
                else:
                    image[k][j][i] = 0
    
    pimage = image
    
    # print('IMAGE BEFORE PADDING>>>>>>')
    # for i in range(outchannel):
    #     for j in range(len(pimage[i])):
    #         print(pimage[i][j])
    #     print('--------------')
    # print('<<<<<<<<<<<<<<<<<<<<<<<<<<')

    # padding
    for i in range(outchannel):
        for _ in range(ksize_h-1-pad_h_up):
            pimage[i].insert(0, [0 for j in range(len(pimage[i][0]))])
    for i in range(outchannel):
        for _ in range(ksize_h-1-pad_h_down):
            pimage[i].append([0 for j in range(len(pimage[i][0]))])
    for i in range(outchannel):
        for j in range(len(pimage[i])):
            for _ in range(ksize_w-1-pad_w_left):
                pimage[i][j].insert(0, 0)
    for i in range(outchannel):
        for j in range(len(pimage[i])):
            for _ in range(ksize_w-1-pad_w_right):
                pimage[i][j].append(0)
    
    # print('IMAGE AFTER PADDING>>>>>>>')
    # for i in range(outchannel):
    #     for j in range(len(pimage[i])):
    #         print(pimage[i][j])
    #     print('--------------')
    # print('<<<<<<<<<<<<<<<<<<<<<<<<<<')

    # img2col
    prefetch_length = _outputsize_h*_outputsize_w

    for r in range(b_row):
        if r // (ksize_h*ksize_w) >= 1.:
            prefetch_length_t = prefetch_length
        else:
            prefetch_length_t = 0
        for c in range(b_col):
            pixel_in_channel_of_image = r // (ksize_h*ksize_w)
            pixel_row_in_outimage = int(c // outputsize_w) % outputsize_h
            pixel_col_in_outimage = c % outputsize_w
            pixel_row_in_window = int(r // ksize_w) % ksize_h
            pixel_col_in_window = r % ksize_w
            pixel_row_in_image = pixel_row_in_outimage*1+pixel_row_in_window
            pixel_col_in_image = pixel_col_in_outimage*1+pixel_col_in_window
            if pimage[pixel_in_channel_of_image][pixel_row_in_image][pixel_col_in_image] == 0:
                last_B[r][c] = 0
            else:
                last_B[r][c] = pimage[pixel_in_channel_of_image][pixel_row_in_image][pixel_col_in_image] + pixel_in_channel_of_image*prefetch_length_t
    
    # debug:  
    #     from gen_comparefile_new import gencomparefile_loss
    #     gencomparefile_loss(4,4,2,2,2,2,2,2,4,4,2,3)
    
    # print('MATRIX B>>>>>>>>>>>>>>>>>>')
    # for r in range(len(last_B)):
    #     print(last_B[r])
    # print('<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    zero_num = 0
    for r in range(len(last_B)):
        for c in range(len(last_B[r])):
            if last_B[r][c] == 0:
                zero_num += 1
    print('\t\tzero_rate: ', zero_num/(len(last_B)*len(last_B[0])))

    ZeroNumRate_after_im2col = \
        calculate_sparsity(forward_outputsize_h, forward_outputsize_w, ksize_h-1-pad_h_up, ksize_h-1-pad_h_down, \
                           ksize_w-1-pad_w_left, ksize_w-1-pad_w_right, stride_h-1, stride_w-1, outchannel, ksize_h, \
                           ksize_w, 1, 1, len(last_B)*len(last_B[0]), debug=0)
    print('\t\tzero_rate: ', ZeroNumRate_after_im2col[0])
    if zero_num == ZeroNumRate_after_im2col[1]:
        print('\t\tRight!!!')
    else:
        print('\t\t\033[0;31;40mError!!!\033[0m')
    

    return last_B, len(last_B)*len(last_B[0]), zero_num, ZeroNumRate_after_im2col[0]

def addrSparseGradient(image_size_h, image_size_w, inchannel, outchannel, ksize_h, ksize_w, \
                       pad_h_up, pad_h_down, pad_w_left, pad_w_right, stride_h, stride_w):
    print('\tGradient...')
    # backward paras
    backward_image_size_h = image_size_h
    backward_image_size_w = image_size_w
    backward_outputsize_h = ksize_h # GRADIENT SIZE_H
    backward_outputsize_w = ksize_w # GRADIENT SIZE_W
    backward_pad_h_up     = pad_h_up
    backward_pad_h_down   = pad_h_down
    backward_pad_w_left   = pad_w_left
    backward_pad_w_right  = pad_w_right
    backward_batchsize    = inchannel
    backward_stride_h     = 1
    backward_stride_w     = 1
    backward_ksize_h_old  = int((image_size_h+pad_h_up+pad_h_down-ksize_h)/stride_h+1) # LOSS SIZE_H
    backward_ksize_w_old  = int((image_size_w+pad_w_left+pad_w_right-ksize_w)/stride_w+1) # LOSS SIZE_W
    
    backward_ksize_h      = backward_ksize_h_old+(backward_ksize_h_old-1)*(stride_h-1)
    backward_ksize_w      = backward_ksize_w_old+(backward_ksize_w_old-1)*(stride_w-1)
    
    prefetch_length       = backward_image_size_h * backward_image_size_w
    
    backward_b_col = backward_batchsize * backward_outputsize_h * backward_outputsize_w
    backward_b_row = 1 * backward_ksize_h * backward_ksize_w
    
    pad_num = 0
    
    backward_last_B = [[0 for i in range(backward_b_col)] for j in range(backward_b_row)]
    backward_image = [[[0 for i in range(backward_image_size_w)] for j in range(backward_image_size_h)] for k in range(backward_batchsize)]
    pixel = 1
    for k in range(backward_batchsize):
        pixel = 1
        for j in range(backward_image_size_h):
            for i in range(backward_image_size_w):
                backward_image[k][j][i] = pixel
                pixel += 1
    
    # print('IMAGE BEFORE PADDING>>>>>>')
    # for i in range(backward_batchsize):
    #     for j in range(len(backward_image[i])):
    #         print(backward_image[i][j])
    #     print('--------------')
    # print('<<<<<<<<<<<<<<<<<<<<<<<<<<')

    # padding
    for i in range(backward_batchsize):
        for _ in range(backward_pad_h_up):
            backward_image[i].insert(0, [0 for j in range(len(backward_image[i][0]))])
    for i in range(backward_batchsize):
        for _ in range(backward_pad_h_down):
            backward_image[i].append([0 for j in range(len(backward_image[i][0]))])
    for i in range(backward_batchsize):
        for j in range(len(backward_image[i])):
            for _ in range(backward_pad_w_left):
                backward_image[i][j].insert(0, 0)
    for i in range(backward_batchsize):
        for j in range(len(backward_image[i])):
            for _ in range(backward_pad_w_right):
                backward_image[i][j].append(0)
    
    # print('IMAGE AFTER PADDING>>>>>>>')
    # for i in range(backward_batchsize):
    #     for j in range(len(backward_image[i])):
    #         print(backward_image[i][j])
    #     print('--------------')
    # print('<<<<<<<<<<<<<<<<<<<<<<<<<<')
    
    # img2col
    for r in range(backward_b_row):
        for c in range(backward_b_col):
            if c // (backward_outputsize_h*backward_outputsize_w) >= 1.:
                prefetch_length_t = prefetch_length
            else:
                prefetch_length_t = 0
            pixel_in_channel_of_image = c // (backward_outputsize_h*backward_outputsize_w)
            pixel_row_in_outimage = int(c // backward_outputsize_w) % backward_outputsize_h
            pixel_col_in_outimage = c % backward_outputsize_w
            pixel_row_in_window = int(r // backward_ksize_w) % backward_ksize_h
            pixel_col_in_window = r % backward_ksize_w
            pixel_row_in_image = pixel_row_in_outimage*backward_stride_h+pixel_row_in_window
            pixel_col_in_image = pixel_col_in_outimage*backward_stride_w+pixel_col_in_window
            if backward_image[pixel_in_channel_of_image][pixel_row_in_image][pixel_col_in_image] == 0:
                backward_last_B[r][c] = 0
            else:
                backward_last_B[r][c] = backward_image[pixel_in_channel_of_image][pixel_row_in_image][pixel_col_in_image] + pixel_in_channel_of_image*prefetch_length_t
     
    #print('MATRIX B>>>>>>>>>>>>>>>>>>')
    #for r in range(len(backward_last_B)):
    #    for c in range(len(backward_last_B[0])):
    #        backward_last_B[r][c] -= 1 #
    #        if c % 16 == 0:
    #            print("|%4d" % (backward_last_B[r][c]), end=',')
    #        else:
    #            print("%5d" % (backward_last_B[r][c]), end=',')
    #    print('')
    #    if (r+1) % 16 == 0:
    #        print('-'*6*len(backward_last_B[0]))
    #print('<<<<<<<<<<<<<<<<<<<<<<<<<<')
    zero_num = 0
    for r in range(len(backward_last_B)):
        for c in range(len(backward_last_B[r])):
            if backward_last_B[r][c] == 0:
                zero_num += 1
    print('\t\tzero_rate: ', zero_num/(len(backward_last_B)*len(backward_last_B[0])))

    ZeroNumRate_after_im2col = \
        calculate_sparsity(backward_image_size_h, backward_image_size_w, backward_pad_h_up, backward_pad_h_down, \
                           backward_pad_w_left, backward_pad_w_right, 0, 0, backward_batchsize, backward_ksize_h, \
                           backward_ksize_w, 1, 1, len(backward_last_B)*len(backward_last_B[0]), debug=0)
    print('\t\tzero_rate: ', ZeroNumRate_after_im2col[0])

    if zero_num == ZeroNumRate_after_im2col[1]:
        print('\t\tRight!!!')
    else:
        print('\t\t\033[0;31;40mError!!!\033[0m')
    
    return backward_last_B, len(backward_last_B)*len(backward_last_B[0]), zero_num, ZeroNumRate_after_im2col[0]


def get_sparsity(model, model_name):
    #model_name = str(model).split('(')[0]
    f = open('./Result/Sparsity_'+model_name+'.csv', 'w', encoding='utf-8', newline='')
    f_config = open('./Config/Sparsity_'+model_name+'.cfg', "w")
    f_config.writelines('type,Hi,Wi,Ho,Wo,C,D,Kh,Kw,S,P\n')
    csv_writer = csv.writer(f)

    print('\033[0;32;40m=============================================================\033[0m')
    print('\033[0;32;40mCalculating model of '+model_name+'...\033[0m')

    #csv_writer.writerow(["input_size","output_size","kernel_size","stride","padding"])
    csv_writer.writerow(["input_size","output_size","kernel_size","stride","padding",\
                         "forward_featuremap_internal_padding","forward_featuremap_external_padding",\
                         "forward_weight_internal_padding","forward_featuremap_elements_after_im2col",\
                         "forward_featuremap_zeros_after_im2col","loss_featuremap_internal_padding",\
                         "loss_featuremap_external_padding","loss_weight_internal_padding",\
                         "loss_featuremap_elements_after_im2col","loss_featuremap_zeros_after_im2col",\
                         "gradient_featuremap_internal_padding","gradient_featuremap_external_padding",\
                         "gradient_weight_internal_padding","gradient_featuremap_elements_after_im2col",\
                         "gradient_featuremap_zeros_after_im2col","gradient_weight_zeros_after_im2col",\
                         "forward_featuremap_sparsity_after_im2col","forward_weight_sparsity_after_im2col",\
                         "loss_featuremap_sparsity_after_im2col","loss_weight_sparsity_after_im2col",\
                         "gradient_featuremap_sparsity_after_im2col","gradient_weight_sparsity_after_im2col",\
                         "forward_featuremap_additional_storage","loss_featuremap_additional_storage",\
                         "gradient_kernal_additional_storage"])
    #======================
    paras_conv = []
    for i, m in enumerate(model.modules()):
        if isinstance(m, torch.nn.Conv2d):
            kernel_size = m.__dict__.get('kernel_size')
            stride = m.__dict__.get('stride')
            padding = m.__dict__.get('padding')
            if not isinstance(kernel_size, tuple):
                kernel_size = (kernel_size, kernel_size)
            if not isinstance(stride, tuple):
                stride = (stride, stride)
            if not isinstance(padding, tuple):
                padding = (padding, padding)
            paras_conv.append({'kernel_size': kernel_size, 'stride': stride, 'padding': padding})
    #======================
    index_of_layer = 0
    modelsummary = summary(model, (3, 224, 224), depth=5, col_width=20, col_names=("input_size", "output_size"), verbose=0)
    for line in str(modelsummary).split('\n'):
        if 'Conv2d' in line:
            #print(line)
            tmp1 = line.split('[')[1].split(']')[0]
            input_size = (int(tmp1.split(',')[0]), int(tmp1.split(',')[1]), int(tmp1.split(',')[2]), int(tmp1.split(',')[3]))
            tmp2 = line.split('[')[-1].split(']')[0]
            output_size = (int(tmp2.split(',')[0]), int(tmp2.split(',')[1]), int(tmp2.split(',')[2]), int(tmp2.split(',')[3]))
            paras_conv[index_of_layer]['input_size'] = input_size
            paras_conv[index_of_layer]['output_size'] = output_size
            index_of_layer += 1

    #======================
    
    for _ in range(len(paras_conv)):
        item = paras_conv[_]
        print('\033[0;32;40m  process ' + str(_+1) + '-th layer of ' + str(len(paras_conv)) + ' layers...\033[0m')
        image_size_h    = item["input_size"][2]
        image_size_w    = item["input_size"][3]
        inchannel       = item["input_size"][1]
        outchannel      = item["output_size"][1]
        ksize_h         = item["kernel_size"][0]
        ksize_w         = item["kernel_size"][1]
        pad_h_up        = item["padding"][0]
        pad_h_down      = item["padding"][0]
        pad_w_left      = item["padding"][1]
        pad_w_right     = item["padding"][1]
        stride_h        = item["stride"][0]
        stride_w        = item["stride"][1]
        h_mod = (image_size_h + pad_h_up + pad_h_down - ksize_h) % stride_h
        w_mod = (image_size_w + pad_w_left + pad_w_right - ksize_w) % stride_w
        if not h_mod == 0:
            pad_h_down += stride_h - h_mod
        if not w_mod == 0:
            pad_w_right += stride_w - w_mod

        
        

        # forward
        inchannel_bak = 1
        item["forward_featuremap_internal_padding"] = 0#(item["stride"][0]-1, item["stride"][1]-1)
        item["forward_featuremap_external_padding"] = item["padding"]#(item["kernel_size"][0]-item["padding"][0]-1, item["stride"][1]-1)
        item["forward_weight_internal_padding"] = 0
        cal = addrSparseForward(image_size_h, image_size_w, inchannel_bak, outchannel, ksize_h, ksize_w, \
                                pad_h_up, pad_h_down, pad_w_left, pad_w_right, stride_h, stride_w)
        item["forward_featuremap_elements_after_im2col"] = cal[1]*inchannel
        item["forward_featuremap_zeros_after_im2col"] = cal[2]*inchannel
        item["forward_featuremap_sparsity_after_im2col"] = cal[3]
        item["forward_weight_sparsity_after_im2col"] = 0
        # loss
        outchannel_bak = 1
        item["loss_featuremap_internal_padding"] = (item["stride"][0]-1, item["stride"][1]-1)
        item["loss_featuremap_external_padding"] = (item["kernel_size"][0]-item["padding"][0]-1, item["kernel_size"][1]-item["padding"][1]-1)
        item["loss_weight_internal_padding"] = 0
        cal = addrSparseLOSS(image_size_h, image_size_w, inchannel, outchannel_bak, ksize_h, ksize_w, \
                             pad_h_up, pad_h_down, pad_w_left, pad_w_right, stride_h, stride_w)
        item["loss_featuremap_elements_after_im2col"] = cal[1]*outchannel
        item["loss_featuremap_zeros_after_im2col"] = cal[2]*outchannel
        item["loss_featuremap_sparsity_after_im2col"] = cal[3]
        item["loss_weight_sparsity_after_im2col"] = 0
        # gradient
        inchannel_bak = 1
        item["gradient_featuremap_internal_padding"] = 0
        item["gradient_featuremap_external_padding"] = item["padding"]
        item["gradient_weight_internal_padding"] = (item["stride"][0]-1, item["stride"][1]-1)
        cal = addrSparseGradient(image_size_h, image_size_w, inchannel_bak, outchannel, ksize_h, ksize_w, \
                                 pad_h_up, pad_h_down, pad_w_left, pad_w_right, stride_h, stride_w)
        item["gradient_featuremap_elements_after_im2col"] = cal[1]*inchannel
        item["gradient_featuremap_zeros_after_im2col"] = cal[2]*inchannel
        item["gradient_featuremap_sparsity_after_im2col"] = cal[3]
        item["gradient_weight_sparsity_after_im2col"] = 1.-(item["output_size"][2]*item["output_size"][3])/\
                                                        ((item["output_size"][2]+(item["output_size"][2]-1)*(item["stride"][0]-1))*\
                                                        (item["output_size"][3]+(item["output_size"][3]-1)*(item["stride"][1]-1)))
        item["gradient_weight_zeros_after_im2col"] = (((item["output_size"][2]+(item["output_size"][2]-1)*(item["stride"][0]-1))*\
                                                     (item["output_size"][3]+(item["output_size"][3]-1)*(item["stride"][1]-1)))-\
                                                     (item["output_size"][2]*item["output_size"][3]))*outchannel
        # others
        item["forward_featuremap_additional_storage"] = ((image_size_h+pad_h_up+pad_h_down)*(image_size_w+pad_w_left+pad_w_right)-image_size_h*image_size_w)*\
                                                        inchannel*32/8/1024/1024       # FP32, MByte
        item["loss_featuremap_additional_storage"] = ((  (item["output_size"][2]+(item["output_size"][2]-1)*(item["stride"][0]-1)+(ksize_h-1-pad_h_up)+(ksize_h-1-pad_h_down)) *\
                                                     (item["output_size"][3]+(item["output_size"][3]-1)*(item["stride"][1]-1)+(ksize_w-1-pad_w_left)+(ksize_w-1-pad_w_right)) )-\
                                                     (item["output_size"][2]*item["output_size"][3]))*outchannel*\
                                                     32/8/1024/1024       # FP32, MByte
        item["gradient_kernal_additional_storage"] = item["gradient_weight_zeros_after_im2col"]*32/8/1024/1024       # FP32, MByte
    for item in paras_conv:
        csv_writer.writerow([str(item["input_size"]),str(item["output_size"]),str(item["kernel_size"]),str(item["stride"]),str(item["padding"]),\
                            str(item["forward_featuremap_internal_padding"]),str(item["forward_featuremap_external_padding"]),\
                            str(item["forward_weight_internal_padding"]),str(item["forward_featuremap_elements_after_im2col"]),\
                            str(item["forward_featuremap_zeros_after_im2col"]),str(item["loss_featuremap_internal_padding"]),\
                            str(item["loss_featuremap_external_padding"]),str(item["loss_weight_internal_padding"]),\
                            str(item["loss_featuremap_elements_after_im2col"]),str(item["loss_featuremap_zeros_after_im2col"]),\
                            str(item["gradient_featuremap_internal_padding"]),str(item["gradient_featuremap_external_padding"]),\
                            str(item["gradient_weight_internal_padding"]),str(item["gradient_featuremap_elements_after_im2col"]),\
                            str(item["gradient_featuremap_zeros_after_im2col"]),str(item["gradient_weight_zeros_after_im2col"]),\
                            str(item["forward_featuremap_sparsity_after_im2col"]),str(item["forward_weight_sparsity_after_im2col"]),\
                            str(item["loss_featuremap_sparsity_after_im2col"]),str(item["loss_weight_sparsity_after_im2col"]),\
                            str(item["gradient_featuremap_sparsity_after_im2col"]),str(item["gradient_weight_sparsity_after_im2col"]),\
                            str(item["forward_featuremap_additional_storage"]),str(item["loss_featuremap_additional_storage"]),
                            str(item["gradient_kernal_additional_storage"])])
        f_config.writelines('convolution layer'+','+str(item["input_size"][2])+','+str(item["input_size"][3])+','+str(item["output_size"][2])+','+\
                            str(item["output_size"][3])+','+str(item["input_size"][1])+','+str(item["output_size"][1])+','+str(item["kernel_size"][0])+','+\
                            str(item["kernel_size"][1])+','+str(item["stride"][0])+','+str(item["padding"][0])+'\n')

    f.close()
    f_config.close()

if __name__ == '__main__':

    ## IF you want to test only one conv layer, you can specify the 
    ## parameters of this layer and calculating mode, then run this code.
    ## You can choose mode from:  "forward" / "loss" / "gradient"
    mode = 'gradient' 

    image_size_h    = 23
    image_size_w    = 23
    inchannel       = 1
    outchannel      = 1
    ksize_h         = 11
    ksize_w         = 11
    pad_h_up        = 2
    pad_h_down      = 2
    pad_w_left      = 2
    pad_w_right     = 2
    stride_h        = 4
    stride_w        = 4

    if not (image_size_h + pad_h_up + pad_h_down - ksize_h) % stride_h == 0 or\
       not (image_size_w + pad_w_left + pad_w_right - ksize_w) % stride_w == 0:
       print("\033[31mParams can not div!\n",\
             "[(image_size_h + pad_h_up + pad_h_down - ksize_h) % stride_h == 0] or \n",\
             "[(image_size_w + pad_w_left + pad_w_right - ksize_w) % stride_w == 0]\n",\
             "Check your params!\033[0m")
             
       exit()

    if mode == 'forward':
        addrSparseForward(image_size_h, image_size_w, inchannel, outchannel, ksize_h, ksize_w, \
                          pad_h_up, pad_h_down, pad_w_left, pad_w_right, stride_h, stride_w)
    elif mode == 'loss':
        addrSparseLOSS(image_size_h, image_size_w, inchannel, outchannel, ksize_h, ksize_w, \
                       pad_h_up, pad_h_down, pad_w_left, pad_w_right, stride_h, stride_w)
    elif mode == 'gradient':
        addrSparseGradient(image_size_h, image_size_w, inchannel, outchannel, ksize_h, ksize_w, \
                           pad_h_up, pad_h_down, pad_w_left, pad_w_right, stride_h, stride_w)

    
    all_model = [{"model": tv.models.alexnet(), "model_name": "alexnet"},\
                 {"model": tv.models.vgg11(), "model_name": "vgg11"},\
                 {"model": tv.models.vgg13(), "model_name": "vgg13"},\
                 {"model": tv.models.vgg16(), "model_name": "vgg16"},\
                 {"model": tv.models.vgg19(), "model_name": "vgg19"},\
                 {"model": tv.models.resnet18(), "model_name": "resnet18"},\
                 {"model": tv.models.resnet34(), "model_name": "resnet34"},\
                 {"model": tv.models.resnet50(), "model_name": "resnet50"},\
                 {"model": tv.models.resnet101(), "model_name": "resnet101"},\
                 {"model": tv.models.resnet152(), "model_name": "resnet152"},\
                 {"model": tv.models.squeezenet1_0(), "model_name": "squeezenet1_0"},\
                 {"model": tv.models.squeezenet1_1(), "model_name": "squeezenet1_1"},\
                 {"model": tv.models.densenet121(), "model_name": "densenet121"},\
                 {"model": tv.models.densenet169(), "model_name": "densenet169"},\
                 {"model": tv.models.densenet161(), "model_name": "densenet161"},\
                 {"model": tv.models.densenet201(), "model_name": "densenet201"},\
                 {"model": tv.models.shufflenet_v2_x0_5(), "model_name": "shufflenet_v2_x0_5"},\
                 {"model": tv.models.shufflenet_v2_x1_0(), "model_name": "shufflenet_v2_x1_0"},\
                 {"model": tv.models.shufflenet_v2_x1_5(), "model_name": "shufflenet_v2_x1_5"},\
                 {"model": tv.models.shufflenet_v2_x2_0(), "model_name": "shufflenet_v2_x2_0"},\
                 {"model": tv.models.mobilenet_v2(), "model_name": "mobilenet_v2"},\
                 {"model": tv.models.mobilenet_v3_large(), "model_name": "mobilenet_v3_large"},\
                 {"model": tv.models.mobilenet_v3_small(), "model_name": "mobilenet_v3_small"}]

    for item in all_model:
        try:
            get_sparsity(item["model"], item["model_name"])
        except:
            print('\033[31mCalculating '+item["model_name"]+' falied, this code may not support inception/googlenet now...\033[0m')
        
