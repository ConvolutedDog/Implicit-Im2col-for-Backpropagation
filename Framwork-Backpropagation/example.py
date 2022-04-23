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

# We chose a small model to show how to run this code.
class ExampleNet(torch.nn.Module):
	def __init__(self):
		super(ExampleNet, self).__init__()
		self.img = torch.rand(2,3,44,44)
		self.label = torch.Tensor([1 for i in range(2)]).long()
		self.features = torch.nn.Sequential(
			torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
		)
		self.classifier = torch.nn.Sequential(
			torch.nn.Dropout(),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features=14112, out_features=1000, bias=True)
		)
 
	def forward(self, x):
		conv_out = self.features(x)
		res = conv_out.view(conv_out.size(0), -1)
		out = self.classifier(res)
		return out


# model, img and label
model = ExampleNet()
print(model, end='\n')
img = model.img
label = model.label

# Inference
result = model(img)

# Use our framework to perform Backpropagation.
dz_list, dw, db = gradient_backward_v2(model, img, label, num_class=1000, g_view=True)

# dz_list[-1] is the loss of output of inference 
# (result in this code).
# dz_list[0:-2] is the loss of the input of all layers.
# dw is the gradient of the first conv layer's kernel.
# db is the loss of the first conv layer's bias.
