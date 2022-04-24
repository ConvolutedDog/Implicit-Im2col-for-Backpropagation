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
import torch
import torchvision
from utils.utils_v2 import gradient_backward_v2

# __all__ is all the models we now support.
__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2',
           'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 
           'vgg16_bn', 'vgg19', 'vgg19_bn', 'alexnet']

# You can choose the model from __all__.
model = torchvision.models.alexnet()
print(model, end='\n')

# Alexnet can use (224, 224) pictures or (228, 228) pictures;
# VGG series network models can use (224, 224) or (228, 228) pictures;
# ResNet and ResNext series network models can use (228, 228) pictures.
# Choose the input shape (BacthSize x InputChannel x Height x Width).
img = torch.rand(2,3,228,228)

# You also can edit the label. 
label = torch.Tensor([1 for i in range(2)]).long()

# g_view = True means means that the structure diagram of neural network model will be drawn.
# For details on the implementation of back propagation on each layer, refer to the source code
# in utils_v2.py.
dz_list, dw, db = gradient_backward_v2(model, img, label, num_class=1000, g_view=True)
