import torchvision
import torch
import torch.nn as nn

from utils import *

class AlexNet(nn.Module):
	def __init__(self, num_classes: int = 1000) -> None:
		super(AlexNet, self).__init__()
		self.img = torch.rand(2,3,224,224)
		self.label = torch.Tensor([1 for i in range(2)]).long()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=12, stride=4, padding=2),
			nn.ReLU(inplace=True),
			nn.AvgPool2d(kernel_size=3, stride=2, padding=0),
			nn.Conv2d(64, 192, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(192, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.AvgPool2d(kernel_size=3, stride=2, padding=0),
		)
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(256 * 6 * 6, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x


class Net2(torch.nn.Module):
	def __init__(self):
		super(Net2, self).__init__()
		self.img = torch.rand(2,3,44,44)
		self.label = torch.Tensor([1 for i in range(2)]).long()
		self.features = torch.nn.Sequential(
			torch.nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
		)
		self.classifier = torch.nn.Sequential(
			nn.Dropout(),
			torch.nn.ReLU(),
			nn.Dropout(),
			torch.nn.Linear(in_features=14112, out_features=672, bias=True),
			torch.nn.ReLU(),
			torch.nn.Linear(in_features=672, out_features=1000, bias=True)
		)
 
	def forward(self, x):
		conv_out = self.features(x)
		res = conv_out.view(conv_out.size(0), -1)
		out = self.classifier(res)
		return out



model = AlexNet
print(model)
img = model.img
label = model.label

# 删除tmp_file下的pth文件
delete_allpths(pth_dir=None)

# 用pytorch跑一遍结果
result = model(img)

# pytorch梯度反向传播
Loss = nn.CrossEntropyLoss()
loss = Loss(result, label)
loss.backward()

# 自己算一遍梯度反向传播，dz_list是每一层的误差，从最后一层到第一层
dz_list, dw, db = gradient_backward(model, img, label)


# dw 是自己算的第一层卷积W的误差，跟pytorch算的误差对比的错误率并输出：
print(judge_tensors_equal(dw, model.features[0].weight.grad))
print(judge_tensors_equal(db, model.features[0].bias.grad))