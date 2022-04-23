import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def cross_entropy_loss(y_predict, y_true):
	"""
	交叉熵损失函数
	:param y_predict: 预测值, shape (N,d), N为批量样本数
	:param y_true: 真实值, shape (N,d)
	:return: 交叉熵损失和在y_predict上的梯度
	"""
	#print(y_predict, y_true)
	print('\n======================== Layer:'+' {0:18}'.format('cross_entropy_loss')+' Start ========================')
	print('# y_predict.shape: ', list(y_predict.shape))
	print('# y_true.shape: ', list(y_true.shape))
	y_shift = torch.sub(y_predict, torch.max(y_predict, dim=1, keepdim=True).values)
	y_exp = torch.exp(y_shift)
	y_probability = torch.div(y_exp, torch.sum(y_exp, dim=1, keepdim=True))
	ypred_loss = torch.mean(-torch.sum(torch.mul(y_true, torch.log(y_probability)), dim=1, keepdim=True))
	dLoss_dypred = y_probability - y_true
	print('# dLoss_dypred.shape: ', list(dLoss_dypred.shape))
	print('# Self calculated loss: ', ypred_loss.item())
	print('======================== Layer:'+' {0:18}'.format('cross_entropy_loss')+' End ==========================')
	return ypred_loss, dLoss_dypred

@torch.no_grad()
def fc_backward(dLoss_dnextz, z, w):
	"""
	全连接层z求误差, 同时求w和b的梯度函数
	:param dLoss_dnextz: 下一层的误差
	:param z: 当前层的输出
	:param w: 当前层的权重
	:return: 全连接层z的误差和在w和b上的梯度
	"""
	print('# next_dz.shape: ', list(dLoss_dnextz.shape))
	print('# z.shape: ', list(z.shape))
	print('# weight.shape: ', list(w.shape))
	print('# bias.shape: ', '['+str(dLoss_dnextz.shape[1])+']')
	N = z.shape[0]
	dLoss_dz = torch.matmul(dLoss_dnextz, w) #delta
	dLoss_dfcW = torch.matmul(dLoss_dnextz.t(), z)
	dLoss_dfcB = torch.sum(dLoss_dnextz, dim=0)
	print('# dz.shape: ', list(dLoss_dz.shape))
	print('# dweight.shape: ', list(dLoss_dfcW.shape))
	print('# dbias.shape: ', list(dLoss_dfcB.shape))
	return dLoss_dz, dLoss_dfcW/N, dLoss_dfcB/N

@torch.no_grad()
def relu_backward(next_dz, z):
	"""
	relu反向传播
	:param next_dz: 激活后的梯度
	:param z: 激活前的值
	:return: relu层z的误差
	"""
	print('# next_dz.shape: ', list(next_dz.shape))
	print('# z.shape: ', list(z.shape))
	zeros_tensor = torch.zeros_like(next_dz)
	dLoss_dz = torch.where(torch.gt(z, 0), next_dz, zeros_tensor)
	print('# dz.shape: ', list(dLoss_dz.shape))
	return dLoss_dz

@torch.no_grad()
def dropback_backward(next_dz, mask, p):
	"""
	dropback反向传播
	:param next_dz: 激活后的梯度
	:param mask: dropback的掩码矩阵
	:param p: dropback的置零概率
	:return: dropback层z的误差
	"""
	print('# zeros probability: ', p)
	print('# next_dz.shape: ', list(next_dz.shape))
	print('# mask.shape: ', list(mask.shape))
	zeros_tensor = torch.zeros_like(mask)
	dLoss_dz = torch.mul(torch.where(torch.eq(mask, 1.), next_dz, zeros_tensor), 1./(1.-p))
	print('# dz.shape: ', list(dLoss_dz.shape))
	return dLoss_dz

@torch.no_grad()
def max_pooling_backward(next_dz, z, pooling, strides, padding=(0, 0)):
	"""
	最大池化反向过程
	:param next_dz：损失函数关于最大池化输出的损失
	:param z: 卷积层矩阵, 形状(N,C,H,W), N为batch_size, C为通道数
	:param pooling: 池化大小(k1,k2)
	:param strides: 步长
	:param padding: 0填充
	:return:
	"""
	print('# next_dz.shape: ', list(next_dz.shape))
	print('# z.shape: ', list(z.shape))
	print('# padding: ', padding)
	print('# strides: ', strides)
	N, C, H, W = z.shape
	_, _, out_h, out_w = next_dz.shape
	# 零填充
	padding_z = F.pad(z, pad=(padding[1],padding[1],padding[0],\
					  padding[0],0,0), mode='constant', value=0)
	# 零填充后的梯度
	padding_dz = torch.zeros_like(padding_z)
	for n in torch.arange(N):
		for c in torch.arange(C):
			for i in torch.arange(out_h):
				for j in torch.arange(out_w):
					# 找到最大值的那个元素坐标，将梯度传给这个坐标
					flat_idx = torch.argmax(padding_z[n, c,
										 strides[0] * i:strides[0] * i + pooling[0],
										 strides[1] * j:strides[1] * j + pooling[1]])
					h_idx = strides[0] * i + flat_idx // pooling[1]
					w_idx = strides[1] * j + flat_idx % pooling[1]

					padding_dz[n, c, h_idx, w_idx] += next_dz[n, c, i, j]
	# 返回时剔除零填充
	dz = _remove_padding(padding_dz, padding)  # padding_z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
	print('# dz.shape: ', list(dz.shape))
	return dz

@torch.no_grad()
def average_pooling_backward(next_dz, z, pooling, strides, padding=(0, 0)):
	"""
	平均池化反向过程
	:param next_dz：损失函数关于平均池化输出的损失
	:param z: 卷积层矩阵, 形状(N,C,H,W), N为batch_size, C为通道数
	:param pooling: 池化大小(k1,k2)
	:param strides: 步长
	:param padding: 0填充
	:return:
	"""
	print('# next_dz.shape: ', list(next_dz.shape))
	print('# z.shape: ', list(z.shape))
	print('# padding: ', padding)
	print('# strides: ', strides)
	N, C, H, W = z.shape
	_, _, out_h, out_w = next_dz.shape
	# 零填充
	padding_z = F.pad(z, pad=(padding[1],padding[1],padding[0],\
					  padding[0],0,0), mode='constant', value=0)
	# 零填充后的梯度
	padding_dz = torch.zeros_like(padding_z)
	for n in torch.arange(N):
		for c in torch.arange(C):
			for i in torch.arange(out_h):
				for j in torch.arange(out_w):
					# 每个神经元均分梯度
					padding_dz[n, c,
					strides[0] * i:strides[0] * i + pooling[0],
					strides[1] * j:strides[1] * j + pooling[1]] += next_dz[n, c, i, j] / (pooling[0] * pooling[1])
	# 返回时剔除零填充
	dz = _remove_padding(padding_dz, padding)  # padding_z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
	print('# dz.shape: ', list(dz.shape))
	return dz

@torch.no_grad()
def _remove_padding(z, padding):
	"""
	移除padding
	:param z: (N,C,H,W)
	:param paddings: (p1,p2)
	:return:
	"""
	if padding[0] > 0 and padding[1] > 0:
		return z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
	elif padding[0] > 0:
		return z[:, :, padding[0]:-padding[0], :]
	elif padding[1] > 0:
		return z[:, :, :, padding[1]:-padding[1]]
	else:
		return z

@torch.no_grad()
def conv_backward(next_dz, K, z, padding=(0, 0), strides=(1, 1)):
	"""
	多通道卷积层的反向过程
	:param next_dz: 卷积输出层的梯度, (N,C,H,W), H,W为卷积输出层的高度和宽度
	:param K: 当前层卷积核, (C,D,k1,k2)
	:param z: 卷积层矩阵, 形状(N,C,H,W), N为batch_size, C为通道数
	:param padding: padding
	:param strides: 步长
	:return:
	"""
	N, C, H, W = z.shape
	D, C, k1, k2 = K.shape
	N, D, H1, W1 = next_dz.shape
	
	print('# next_dz.shape: ', list(next_dz.shape))
	print('# z.shape: ', list(z.shape))
	print('# weight.shape: ', list(K.shape))
	print('# bias.shape: ', '['+str(K.shape[0])+']')
	print('# padding: ', padding)
	print('# strides: ', strides)
	
	padding_next_dz = _insert_zeros(next_dz, strides)
	# padding_next_dz.shape: N, D, (H'-1)*(strides[0]-1) + H', (W'-1)*(strides[1]-1) + W'
	# 卷积核高度和宽度翻转180度
	flip_K = torch.flip(K, (2, 3))
	# flip_K.shape: D, C, k1, k2
	# 交换C,D为D,C；D变为输入通道数了，C变为输出通道数了
	swap_flip_K = torch.swapaxes(flip_K, 0, 1)
	# swap_flip_K.shape: C, D, k1, k2
	# 增加高度和宽度0填充
	ppadding_next_dz = F.pad(padding_next_dz, pad=(k2-1,k2-1,\
							 k1-1,k1-1,0,0), mode='constant', value=0)
	# ppadding_next_dz.shape: N, D, (H'-1)*(strides[0]-1) + H' + 2k1 - 2, (W'-1)*(strides[1]-1) + W' + 2k2 - 2
	# ppadding_next_dz.shape: N, D, (H'-1)*strides[0] + 2k1 - 1, (W'-1)*strides[1] + 2k2 - 1
	# swap_flip_K.shape: C, D, k1, k2
	dz = _conv_forward(ppadding_next_dz, swap_flip_K)
	
	# 求卷积和的梯度dK
	swap_z = torch.swapaxes(z, 0, 1)  # 变为(C,N,H,W)
	# swap_z.shape: C,N,H,W
	dK = _conv_forward(torch.swapaxes(F.pad(z, pad=(padding[1],padding[1],\
							 padding[0],padding[0],0,0), mode='constant', value=0), 0, 1), torch.swapaxes(padding_next_dz, 0, 1))
	
	# 偏置的梯度
	db = torch.sum(torch.sum(torch.sum(next_dz, axis=-1), axis=-1), axis=0)  # 在高度、宽度上相加；批量大小上相加

	# 把padding减掉
	dz = _remove_padding(dz, padding)  # dz[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
	print('# dz.shape: ', list(dz.shape))
	print('# dweight.shape: ', list(dK.transpose(0,1).shape))
	print('# dbias.shape: ', list(db.shape))
	return dz, (dK/N).transpose(0,1), db/N

@torch.no_grad()
def _conv_forward(x, weight, strides=(1,1)):
	"""
	多通道卷积前向过程的一小部分，没有bias，只是个辅助函数
	:param z: 卷积层矩阵,形状(N,D,H,W)，N为batch_size，C为通道数
	:param K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
	:return: conv_z: 卷积结果[N,D,oH,oW]
	"""
	n, c, h_in, w_in = x.shape
	d, c, k, j = weight.shape
	#print()
	#print(x.shape, weight.shape)
	#print()
	x_pad = x
	x_pad = x_pad.unfold(2, k, strides[0])
	x_pad = x_pad.unfold(3, j, strides[1])        # 按照滑动窗展开
	out = torch.einsum(                          # 按照滑动窗相乘，
		'nchwkj,dckj->ndhw',                    # 并将所有输入通道卷积结果累加
		x_pad, weight)
	return out

@torch.no_grad()
def _insert_zeros(dz, strides):
	"""
	想多维数组最后两位，每个行列之间增加指定的个数的零填充
	:param dz: (N,D,H,W),H,W为卷积输出层的高度和宽度
	:param strides: 步长
	:return:
	"""
	N, D, H, W = dz.shape
	H_last = (H-1)*(strides[0]-1) + H
	W_last = (W-1)*(strides[1]-1) + W
	pz = torch.zeros(N, D, H_last, W_last)
	for n in range(N):
		for d in range(D):
			for h in range(0, H_last, strides[0]):
				for w in range(0, W_last, strides[1]):
					pz[n,d,h,w] = dz[n,d,h//strides[0],w//strides[1]]
	return pz



@torch.no_grad()
def judge_tensors_equal(tensor_A, tensor_B):
	"""
	计算两个张量是否相等，不考虑精度差的情况下
	:param tensor_A， tensor_B: 两个张良
	:return: Error rate: 不相等的元素占总元素的比例
	"""
	if(not tensor_A.shape == tensor_B.shape):
		print('Shape of two compard tensors is not equal.')
		return None
	error = 0
	error_tolerance = 0.00001
	np_A = tensor_A.detach().numpy()
	np_B = tensor_B.detach().numpy()
	if len(tensor_A.shape) == 4:
		print('4D', end=' ')
		N, C, H, W = tensor_A.shape
		for n in range(N):
			for c in range(C):
				for h in range(H):
					for w in range(W):
						if np_A[n,c,h,w]-np_B[n,c,h,w] > error_tolerance:
							#print(np_A[n,c,h,w], np_B[n,c,h,w])
							error += 1
		#print('Error rate: ', error/(N*C*H*W))
		return error/(N*C*H*W)
	elif len(tensor_A.shape) == 1:
		print('1D', end=' ')
		C = tensor_A.shape[0]
		for c in range(C):
			if np_A[c]-np_B[c] > error_tolerance:
				#print(np_A[c], np_B[c])
				error += 1
		#print('Error rate: ', error/C)
		return error/C
	elif len(tensor_A.shape) == 2:
		print('2D', end=' ')
		N, C = tensor_A.shape
		for n in range(N):
			for c in range(C):
				if np_A[n,c]-np_B[n,c] > error_tolerance:
					#print(np_A[n,c], np_B[n,c])
					error += 1
		#print('Error rate: ', error/(C*N))
		return error/(C*N)

@torch.no_grad()
def get_featuremap(featuremap_dir=None):
	"""
	读取训练过程中的featuremap
	:return: featuremap列表
	"""
	import os
	featuremap = []
	if featuremap_dir == None:
		pth_dir = "./tmp_file/"
	else:
		pth_dir = featuremap_dir
	files = os.listdir(pth_dir)
	file_nums = []
	for i in range(len(files)):
		if '.pth' in files[i]:
			file_nums.append(int(files[i].split('.pth')[0]))
	file_nums.sort()
	for file_num in file_nums:
		tensor = torch.load(pth_dir+str(file_num)+'.pth')
		featuremap.append(tensor)
	return featuremap

@torch.no_grad()
def get_structure_parameters(model):
	"""
	获取网络模型每一层layer的参数
	:return: layer参数列表
	"""
	layers = []
	for layer in model.modules():
		if not ':' in str(layer):
			layers.append(layer)
	
	parameters = []
	fc_conv_weights = []
	for layer in layers:
		if isinstance(layer, nn.Conv2d): #layer_name == 'Conv2d':
			Conv2d_params = {}
			Conv2d_params['layer_name'] = layer_name
			# in_channel
			in_channel = layer.__dict__.get('in_channels')
			Conv2d_params['in_channel'] = in_channel
			# out_channel
			out_channel = layer.__dict__.get('out_channels')
			Conv2d_params['out_channel'] = out_channel
			# kernel_size
			kernel_size = layer.__dict__.get('kernel_size')
			if not isinstance(kernel_size, tuple):
				Conv2d_params['kernel_size'] = (kernel_size, kernel_size)
			else:
				Conv2d_params['kernel_size'] = kernel_size
			# stride
			stride = layer.__dict__.get('stride')
			if not isinstance(stride, tuple):
				Conv2d_params['stride'] = (stride, stride)
			else:
				Conv2d_params['stride'] = stride
			# padding
			padding = layer.__dict__.get('padding')
			if not isinstance(padding, tuple):
				Conv2d_params['padding'] = (padding, padding)
			else:
				Conv2d_params['padding'] = padding
			# return
			fc_conv_weights.append(layer.weight)
			parameters.append(Conv2d_params)
		elif isinstance(layer, nn.Relu): #layer_name == 'ReLU':
			pass
			parameters.append({'layer_name': layer_name})
		elif isinstance(layer, nn.MaxPool2d): #layer_name == 'MaxPool2d':
			MaxPool2d_params = {}
			MaxPool2d_params['layer_name'] = layer_name
			# kernel_size
			kernel_size = layer.__dict__.get('kernel_size')
			if not isinstance(kernel_size, tuple):
				MaxPool2d_params['kernel_size'] = (kernel_size, kernel_size)
			else:
				MaxPool2d_params['kernel_size'] = kernel_size
			# stride
			stride = layer.__dict__.get('stride')
			if not isinstance(stride, tuple):
				MaxPool2d_params['stride'] = (stride, stride)
			else:
				MaxPool2d_params['stride'] = stride
			# padding
			padding = layer.__dict__.get('padding')
			if not isinstance(padding, tuple):
				MaxPool2d_params['padding'] = (padding, padding)
			else:
				MaxPool2d_params['padding'] = padding
			# return
			parameters.append(MaxPool2d_params)
		elif isinstance(layer, AvgPool2d): #layer_name == 'AvgPool2d':
			AvgPool2d_params = {}
			AvgPool2d_params['layer_name'] = layer_name
			# kernel_size
			kernel_size = layer.__dict__.get('kernel_size')
			if not isinstance(kernel_size, tuple):
				AvgPool2d_params['kernel_size'] = (kernel_size, kernel_size)
			else:
				AvgPool2d_params['kernel_size'] = kernel_size
			# stride
			stride = layer.__dict__.get('stride')
			if not isinstance(stride, tuple):
				AvgPool2d_params['stride'] = (stride, stride)
			else:
				AvgPool2d_params['stride'] = stride
			# padding
			padding = layer.__dict__.get('padding')
			if not isinstance(padding, tuple):
				AvgPool2d_params['padding'] = (padding, padding)
			else:
				AvgPool2d_params['padding'] = padding
			# return
			parameters.append(AvgPool2d_params)
		elif isinstance(layer, nn.Dropout): #layer_name == 'Dropout':
			Dropout_params = {}
			Dropout_params['layer_name'] = layer_name
			# p
			p = layer.__dict__.get('p')
			Dropout_params['p'] = p
			# return
			parameters.append(Dropout_params)
		elif isinstance(layer, nn.Linear): #layer_name == 'Linear':
			Linear_params = {}
			Linear_params['layer_name'] = layer_name
			# in_features
			in_features = layer.__dict__.get('in_features')
			Linear_params['in_features'] = in_features
			# out_features
			out_features = layer.__dict__.get('out_features')
			Linear_params['out_features'] = out_features
			# return
			fc_conv_weights.append(layer.weight)
			parameters.append(Linear_params)
		elif isinstance(layer, nn.AdaptiveAvgPool2d): #layer_name == 'AdaptiveAvgPool2d':
			AdaptiveAvgPool2d_params = {}
			AdaptiveAvgPool2d_params['layer_name'] = layer_name
			# output_size
			output_size = layer.__dict__.get('output_size')
			if not isinstance(output_size, tuple):
				AdaptiveAvgPool2d_params['output_size'] = (output_size, output_size)
			else:
				AdaptiveAvgPool2d_params['output_size'] = output_size
			# return
			parameters.append(AdaptiveAvgPool2d_params)
		else:
			print('The layer has not been processed in get_structure_parameters!')
	return parameters, fc_conv_weights

@torch.no_grad()
def delete_allpths(pth_dir=None):
	"""
	删除所有上次运行产生的pth文件
	"""
	import os
	if pth_dir == None:
		pth_dir = "./tmp_file/"
	for root, dirs, files in os.walk(pth_dir, topdown=False):
		for name in files:
			if name.endswith('.pth',):
				os.remove(os.path.join(root, name))

@torch.no_grad()
def gradient_backward(model, img, label, num_class=1000):
	return_dz = []
	parameters, fc_conv_weights = get_structure_parameters(model)
	featuremap = get_featuremap(featuremap_dir=None)
	featuremap.insert(0, img) ###
	y_true = F.one_hot(label, num_classes=num_class).float()
	# 最后一层交叉熵计算
	loss, dLoss_dz = cross_entropy_loss(featuremap[-1], y_true)
	#print('Self calculated loss: ', loss)
	featuremap.pop()
	return_dz.append(dLoss_dz)
	# 前面n层倒序遍历
	dW_dB_fc_conv = []
	for i in range(len(parameters)-1, -1, -1):
		layer = parameters[i]
		print('\n======================== {0:3} Layer: '.format(str(i))+'{0:9}'.format(layer['layer_name'])+' Backward Start ========================')
		if layer['layer_name'] == 'Conv2d':
			
			z = featuremap[-1]
			weight_z = fc_conv_weights[-1]
			try:
				padding = layer['padding']
			except:
				padding = (0, 0)
			stride = layer['stride']
			dLoss_dz, dLoss_dW, dLoss_dB = conv_backward(dLoss_dz, weight_z, z, padding, stride)
			return_dz.append(dLoss_dz)
			fc_conv_weights.pop()
			if not len(featuremap) == 1:
				featuremap.pop()
			if not len(dLoss_dz.shape) == len(featuremap[-1].shape):
				dLoss_dz = dLoss_dz.reshape(featuremap[-1].shape)
		elif layer['layer_name'] == 'ReLU':
			z = featuremap[-1]
			dLoss_dz = relu_backward(dLoss_dz, z)
			return_dz.append(dLoss_dz)
			featuremap.pop()
			if not len(dLoss_dz.shape) == len(featuremap[-1].shape):
				dLoss_dz = dLoss_dz.reshape(featuremap[-1].shape)
		elif layer['layer_name'] == 'MaxPool2d':
			z = featuremap[-1]
			pooling = layer['kernel_size']
			stride = layer['stride']
			padding = layer['padding']
			dLoss_dz = max_pooling_backward(dLoss_dz, z, pooling, stride, padding)
			return_dz.append(dLoss_dz)
			featuremap.pop()
			if not len(dLoss_dz.shape) == len(featuremap[-1].shape):
				dLoss_dz = dLoss_dz.reshape(featuremap[-1].shape)
		elif layer['layer_name'] == 'AvgPool2d':
			z = featuremap[-1]
			pooling = layer['kernel_size']
			stride = layer['stride']
			padding = layer['padding']
			dLoss_dz = average_pooling_backward(dLoss_dz, z, pooling, stride, padding)
			return_dz.append(dLoss_dz)
			featuremap.pop()
			if not len(dLoss_dz.shape) == len(featuremap[-1].shape):
				dLoss_dz = dLoss_dz.reshape(featuremap[-1].shape)
		elif layer['layer_name'] == 'Linear':
			weight_z = fc_conv_weights[-1]
			z = featuremap[-1]
			dLoss_dz, dLoss_dW, dLoss_dB = fc_backward(dLoss_dz, z, weight_z)
			return_dz.append(dLoss_dz)
			fc_conv_weights.pop()
			featuremap.pop()
			if not len(dLoss_dz.shape) == len(featuremap[-1].shape):
				dLoss_dz = dLoss_dz.reshape(featuremap[-1].shape)
		elif layer['layer_name'] == 'Dropout':		
			p = layer['p']
			mask = featuremap[-1]
			dLoss_dz = dropback_backward(dLoss_dz, mask, p)
			return_dz.append(dLoss_dz)
			featuremap.pop()
			lastpop = featuremap.pop()
			if not len(dLoss_dz.shape) == len(lastpop.shape):
				dLoss_dz = dLoss_dz.reshape(lastpop.shape)		
		else:
			print('Not completed in gradient_backward!')
		print('======================== {0:3} Layer: '.format(str(i))+'{0:9}'.format(layer['layer_name'])+' Backward End ==========================')

	delete_allpths(pth_dir=None)
	return return_dz, dLoss_dW, dLoss_dB
		