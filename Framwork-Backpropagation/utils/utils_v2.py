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
import torch.nn as nn
import torch.nn.functional as F
from graphviz import Digraph, render
from torch.autograd import Variable

@torch.no_grad()
def cross_entropy_loss(y_predict, y_true):
	print('\n=========================== Layer:'+' {0:18}'.format('cross_entropy_loss')+' Start ===========================')
	print('# y_predict.shape: ', list(y_predict.shape))
	print('# y_true.shape: ', list(y_true.shape))
	y_shift = torch.sub(y_predict, torch.max(y_predict, dim=1, keepdim=True).values)
	y_exp = torch.exp(y_shift)
	y_probability = torch.div(y_exp, torch.sum(y_exp, dim=1, keepdim=True))
	ypred_loss = torch.mean(-torch.sum(torch.mul(y_true, torch.log(y_probability)), dim=1, keepdim=True))
	dLoss_dypred = y_probability - y_true
	print('# dLoss_dypred.shape: ', list(dLoss_dypred.shape))
	print('# Self calculated loss: ', ypred_loss.item())
	print('=========================== Layer:'+' {0:18}'.format('cross_entropy_loss')+' End =============================')
	return ypred_loss, dLoss_dypred

@torch.no_grad()
def fc_backward(dLoss_dnextz, z, w):
	print('# next_dz.shape: ', list(dLoss_dnextz.shape))
	print('# z.shape: ', list(z.shape))
	print('# weight.shape: ', list(w.shape))
	print('# bias.shape: ', '['+str(dLoss_dnextz.shape[1])+']')
	N = z.shape[0]
	if len(z.shape) == 4: 
		z = z.view(z.size(0), -1)
	dLoss_dz = torch.matmul(dLoss_dnextz, w) #delta
	dLoss_dfcW = torch.matmul(dLoss_dnextz.t(), z)
	dLoss_dfcB = torch.sum(dLoss_dnextz, dim=0)
	print('# dz.shape: ', list(dLoss_dz.shape))
	print('# dweight.shape: ', list(dLoss_dfcW.shape))
	print('# dbias.shape: ', list(dLoss_dfcB.shape))
	return dLoss_dz, dLoss_dfcW/N, dLoss_dfcB/N

@torch.no_grad()
def view_backward(dLoss_dnextz, last_z, params):
	print('# next_dz.shape: ', list(dLoss_dnextz.shape))
	print('# last_z.shape: ', list(last_z.shape))
	if params:
		pooling = params[0]
		stride = params[1]
		padding = params[2]
		output_size = (int((last_z.shape[2]-pooling[0]+2*padding[0])/stride[0]+1), \
					int((last_z.shape[3]-pooling[0]+2*padding[0])/stride[0]+1))
		dLoss_dz = dLoss_dnextz.reshape(last_z.shape[0], last_z.shape[1], output_size[0], output_size[1])
	else:
		dLoss_dz = dLoss_dnextz.reshape(last_z.shape)
	print('# dz.shape: ', list(dLoss_dz.shape))
	return dLoss_dz

def add_backward(dLoss_dnextz):
	print('# next_dz.shape: ', list(dLoss_dnextz.shape))
	dLoss_dz = dLoss_dnextz
	print('# dz.shape: ', list(dLoss_dz.shape))
	return dLoss_dz

@torch.no_grad()
def relu_backward(next_dz, z):
	print('# next_dz.shape: ', list(next_dz.shape))
	print('# z.shape: ', list(z.shape))
	zeros_tensor = torch.zeros_like(next_dz)
	dLoss_dz = torch.where(torch.gt(z, 0), next_dz, zeros_tensor)
	print('# dz.shape: ', list(dLoss_dz.shape))
	return dLoss_dz

@torch.no_grad()
def dropback_backward(next_dz, mask, p):
	print('# zeros probability: ', p)
	print('# next_dz.shape: ', list(next_dz.shape))
	print('# mask.shape: ', list(mask.shape))
	zeros_tensor = torch.zeros_like(mask)
	dLoss_dz = torch.mul(torch.where(torch.eq(mask, 1.), next_dz, zeros_tensor), 1./(1.-p))
	print('# dz.shape: ', list(dLoss_dz.shape))
	return dLoss_dz

@torch.no_grad()
def max_pooling_backward(next_dz, z, pooling, strides, padding=(0, 0)):
	print('# next_dz.shape: ', list(next_dz.shape))
	print('# z.shape: ', list(z.shape))
	print('# padding: ', padding)
	print('# strides: ', strides)
	N, C, H, W = z.shape
	_, _, out_h, out_w = next_dz.shape

	padding_z = F.pad(z, pad=(padding[1],padding[1],padding[0],\
					  padding[0],0,0), mode='constant', value=0)

	padding_dz = torch.zeros_like(padding_z)
	for n in torch.arange(N):
		for c in torch.arange(C):
			for i in torch.arange(out_h):
				for j in torch.arange(out_w):
					flat_idx = torch.argmax(padding_z[n, c,
										 strides[0] * i:strides[0] * i + pooling[0],
										 strides[1] * j:strides[1] * j + pooling[1]])
					h_idx = strides[0] * i + flat_idx // pooling[1]
					w_idx = strides[1] * j + flat_idx % pooling[1]

					padding_dz[n, c, h_idx, w_idx] += next_dz[n, c, i, j]
	dz = _remove_padding(padding_dz, padding)  # padding_z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
	print('# dz.shape: ', list(dz.shape))
	return dz

@torch.no_grad()
def batchnorm2d_backward(next_dz, z, eps, gamma=torch.Tensor([1.,1.,1.])):
	print('# next_dz.shape: ', list(next_dz.shape))
	print('# z.shape: ', list(z.shape))
	print('# eps: ', eps)
	print('# gamma.shape: ', list(gamma.shape))
	N, C, H, W = z.shape
	m = N*H*W
	shape = [N,C,H,W]
	import numpy as np
	ax = list(np.arange(len(shape)))
	shape.pop(1)
	ax.pop(1)
	axis = tuple(ax)

	dxhut = torch.zeros_like(next_dz)
	for c in range(C):
		dxhut[:,c] = next_dz[:,c]*gamma[c]
	dz1 = m*dxhut

	mu = z.mean(axis=axis, keepdim=True)
	xmu = z - mu
	xmu2 = xmu**2
	var = xmu2.sum(axis=axis, keepdim=True)/m

	ivar = 1./torch.pow(var+eps, 0.5)

	dz2 = (ivar**2)*((dxhut*xmu).sum(axis=axis, keepdim=True))*xmu
	dz3 = dxhut.sum(axis=axis, keepdim=True)
	dz = ivar/m*(dz1-dz2-dz3)
	print('# dz.shape: ', list(dz.shape))
	return dz

@torch.no_grad()
def average_pooling_backward(next_dz, z, pooling, strides, padding=(0, 0)):
	print('# next_dz.shape: ', list(next_dz.shape))
	print('# z.shape: ', list(z.shape))
	print('# padding: ', padding)
	print('# strides: ', strides)
	N, C, H, W = z.shape
	_, _, out_h, out_w = next_dz.shape

	padding_z = F.pad(z, pad=(padding[1],padding[1],padding[0],\
					  padding[0],0,0), mode='constant', value=0)

	padding_dz = torch.zeros_like(padding_z)
	for n in torch.arange(N):
		for c in torch.arange(C):
			for i in torch.arange(out_h):
				for j in torch.arange(out_w):
					padding_dz[n, c,
					strides[0] * i:strides[0] * i + pooling[0],
					strides[1] * j:strides[1] * j + pooling[1]] += next_dz[n, c, i, j] / (pooling[0] * pooling[1])

	dz = _remove_padding(padding_dz, padding)  # padding_z[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
	print('# dz.shape: ', list(dz.shape))
	return dz

@torch.no_grad()
def _remove_padding(z, padding):
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

	flip_K = torch.flip(K, (2, 3))

	swap_flip_K = torch.swapaxes(flip_K, 0, 1)

	ppadding_next_dz = F.pad(padding_next_dz, pad=(k2-1-padding[1],k2-1-padding[1],\
							 k1-1-padding[0],k1-1-padding[0],0,0), mode='constant', value=0)

	dz = _conv_forward(ppadding_next_dz, swap_flip_K)
	
	swap_z = torch.swapaxes(z, 0, 1) 

	dK = _conv_forward(torch.swapaxes(F.pad(z, pad=(padding[1],padding[1],\
							 padding[0],padding[0],0,0), mode='constant', value=0), 0, 1), torch.swapaxes(padding_next_dz, 0, 1))
	
	db = torch.sum(torch.sum(torch.sum(next_dz, axis=-1), axis=-1), axis=0)  # 在高度、宽度上相加；批量大小上相加

	print('# dz.shape: ', list(dz.shape))
	print('# dweight.shape: ', list(dK.transpose(0,1).shape))
	print('# dbias.shape: ', list(db.shape))
	return dz, (dK/N).transpose(0,1), db/N

@torch.no_grad()
def _conv_forward(x, weight, strides=(1,1)):
	n, c, h_in, w_in = x.shape
	d, c, k, j = weight.shape
	
	x_pad = x
	x_pad = x_pad.unfold(2, k, strides[0])
	x_pad = x_pad.unfold(3, j, strides[1])  
	out = torch.einsum(                     
		'nchwkj,dckj->ndhw', 
		x_pad, weight)
	return out

@torch.no_grad()
def _insert_zeros(dz, strides):
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
	if(not tensor_A.shape == tensor_B.shape):
		print('Shape of two compard tensors is not equal.')
		return None
	error = 0
	error_tolerance = 0.001
	np_A = tensor_A.detach().numpy()
	np_B = tensor_B.detach().numpy()
	if len(tensor_A.shape) == 4:
		N, C, H, W = tensor_A.shape
		for n in range(N):
			for c in range(C):
				for h in range(H):
					for w in range(W):
						if np_A[n,c,h,w]-np_B[n,c,h,w] > error_tolerance or np_B[n,c,h,w]-np_A[n,c,h,w] > error_tolerance:
							error += 1
							if error%20 == 0:
								pass
								print('error', np_A[n,c,h,w], np_B[n,c,h,w])
						else:
							if n*c*h*w % 20000000000000 == 0:
								pass
								#print('right', np_A[n,c,h,w], np_B[n,c,h,w])
							
		#print('Error rate: ', error/(N*C*H*W))
		print('4D-error-rate: ', end=' ')
		return error/(N*C*H*W)
	elif len(tensor_A.shape) == 1:
		C = tensor_A.shape[0]
		for c in range(C):
			if np_A[c]-np_B[c] > error_tolerance or np_B[c]-np_A[c] > error_tolerance:
				#print(np_A[c], np_B[c])
				error += 1
		#print('Error rate: ', error/C)
		print('1D-error-rate: ', end=' ')
		return error/C
	elif len(tensor_A.shape) == 2:
		N, C = tensor_A.shape
		for n in range(N):
			for c in range(C):
				if np_A[n,c]-np_B[n,c] > error_tolerance or np_B[n,c]-np_A[n,c] > error_tolerance:
					#print(np_A[n,c], np_B[n,c])
					error += 1
		#print('Error rate: ', error/(C*N))
		print('2D-error-rate: ', end=' ')
		return error/(C*N)

@torch.no_grad()
def get_featuremap(featuremap_dir=None):
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
	delete_allpths(pth_dir=None)
	return featuremap


@torch.no_grad()
def get_structure_parameters_v1(model):
	layers = []
	for layer in model.modules():
		if not ':' in str(layer):
			layers.append(layer)
	
	parameters = []
	fc_conv_weights = []
	for layer in layers:
		if isinstance(layer, nn.Conv2d): 
			layer_name = 'Conv2d'
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
		elif isinstance(layer, nn.ReLU): 
			layer_name = 'ReLU'
			parameters.append({'layer_name': layer_name})
		elif isinstance(layer, nn.MaxPool2d): 
			layer_name = 'MaxPool2d'
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
		elif isinstance(layer, nn.AvgPool2d): 
			layer_name = 'AvgPool2d'
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
		elif isinstance(layer, nn.Dropout): 
			layer_name = 'Dropout'
			Dropout_params = {}
			Dropout_params['layer_name'] = layer_name
			# p
			p = layer.__dict__.get('p')
			Dropout_params['p'] = p
			# return
			parameters.append(Dropout_params)
		elif isinstance(layer, nn.BatchNorm2d): 
			layer_name = 'BatchNorm2d'
			BatchNorm2d_params = {}
			BatchNorm2d_params['layer_name'] = layer_name
			# num_features
			num_features = layer.__dict__.get('num_features')
			BatchNorm2d_params['num_features'] = num_features
			# eps
			eps = layer.__dict__.get('eps')
			BatchNorm2d_params['eps'] = eps
			# return
			fc_conv_weights.append(layer.weight)
			parameters.append(BatchNorm2d_params)
		elif isinstance(layer, nn.Linear): 
			layer_name = 'Linear'
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
		elif isinstance(layer, nn.AdaptiveAvgPool2d): 
			layer_name = 'AdaptiveAvgPool2d'
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
			print('The layer has not been processed in get_structure_parameters_v1!')
	return parameters, fc_conv_weights


@torch.no_grad()
def delete_allpths(pth_dir=None):
	import os
	if pth_dir == None:
		pth_dir = "./tmp_file/"
	for root, dirs, files in os.walk(pth_dir, topdown=False):
		for name in files:
			if name.endswith('.pth',):
				os.remove(os.path.join(root, name))

@torch.no_grad()
def mul_items(tensor_size):
	x = list(tensor_size)
	mul = 1.
	for i in range(len(x)):
		mul *= x[i]
	return mul


@torch.no_grad()
def gradient_backward_v1(model, img, label, num_class=1000):
	return_dz = []
	parameters, fc_conv_weights = get_structure_parameters_v1(model)
	featuremap = get_featuremap(featuremap_dir=None)
	featuremap.insert(0, img) ###
	y_true = F.one_hot(label, num_classes=num_class).float()
	
	loss, dLoss_dz = cross_entropy_loss(featuremap[-1], y_true)
	print('Self calculated loss: ', loss)
	featuremap.pop()
	return_dz.append(dLoss_dz)
	
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
				lastpop = featuremap.pop()
				if not len(dLoss_dz.shape) == len(lastpop.shape):
					dLoss_dz = dLoss_dz.reshape(lastpop.shape)
		elif layer['layer_name'] == 'ReLU':
			z = featuremap[-1]
			dLoss_dz = relu_backward(dLoss_dz, z)
			return_dz.append(dLoss_dz)
			lastpop = featuremap.pop()
			if not len(dLoss_dz.shape) == len(lastpop.shape):
				dLoss_dz = dLoss_dz.reshape(lastpop.shape)
		elif layer['layer_name'] == 'MaxPool2d':
			z = featuremap[-1]
			pooling = layer['kernel_size']
			stride = layer['stride']
			padding = layer['padding']
			dLoss_dz = max_pooling_backward(dLoss_dz, z, pooling, stride, padding)
			return_dz.append(dLoss_dz)
			lastpop = featuremap.pop()
			if not len(dLoss_dz.shape) == len(lastpop.shape):
				dLoss_dz = dLoss_dz.reshape(lastpop.shape)
		elif layer['layer_name'] == 'AvgPool2d':
			z = featuremap[-1]
			pooling = layer['kernel_size']
			stride = layer['stride']
			padding = layer['padding']
			dLoss_dz = average_pooling_backward(dLoss_dz, z, pooling, stride, padding)
			return_dz.append(dLoss_dz)
			lastpop = featuremap.pop()
			if not len(dLoss_dz.shape) == len(lastpop.shape):
				dLoss_dz = dLoss_dz.reshape(lastpop.shape)
		elif layer['layer_name'] == 'Linear':
			weight_z = fc_conv_weights[-1]
			z = featuremap[-1]
			dLoss_dz, dLoss_dW, dLoss_dB = fc_backward(dLoss_dz, z, weight_z)
			return_dz.append(dLoss_dz)
			fc_conv_weights.pop()
			lastpop = featuremap.pop()
			if not len(dLoss_dz.shape) == len(lastpop.shape):
				dLoss_dz = dLoss_dz.reshape(lastpop.shape)
		elif layer['layer_name'] == 'Dropout':		
			p = layer['p']
			mask = featuremap[-1]
			dLoss_dz = dropback_backward(dLoss_dz, mask, p)
			return_dz.append(dLoss_dz)
			featuremap.pop()
			lastpop = featuremap.pop()
			if not len(dLoss_dz.shape) == len(lastpop.shape):
				dLoss_dz = dLoss_dz.reshape(lastpop.shape)		
		elif layer['layer_name'] == 'BatchNorm2d':		
			eps = layer['eps']
			z = featuremap[-1]
			gamma = fc_conv_weights[-1]
			dLoss_dz = batchnorm2d_backward(dLoss_dz, z, eps, gamma)
			return_dz.append(dLoss_dz)
			fc_conv_weights.pop()
			lastpop = featuremap.pop()
			if not len(dLoss_dz.shape) == len(lastpop.shape):
				dLoss_dz = dLoss_dz.reshape(lastpop.shape)
		else:
			print('Not completed in gradient_backward_v1!')
		print('======================== {0:3} Layer: '.format(str(i))+'{0:9}'.format(layer['layer_name'])+' Backward End ==========================')

	delete_allpths(pth_dir=None)
	return return_dz, dLoss_dW, dLoss_dB


@torch.no_grad()
def make_dot(var, params=None):
	""" Produces Graphviz representation of PyTorch autograd graph
	Blue nodes are the Variables that require grad, orange are Tensors
	saved for backward in torch.autograd.Function
	Args:
		var: output Variable
		params: dict of (name, Variable) to add names to node that
			require grad (TODO: make optional)
	"""
	if params is not None:
		assert isinstance(params.values()[0], Variable)
		param_map = {id(v): k for k, v in params.items()}
 
	node_attr = dict(style='filled',
					 shape='box',
					 align='left',
					 fontsize='12',
					 ranksep='0.1',
					 height='0.2')
	
	dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
	seen = set()
 
	def size_to_str(size):
		return '('+(', ').join(['%d' % v for v in size])+')'
 
	def add_nodes(var):
		if var not in seen:
			if torch.is_tensor(var):
				dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
			elif hasattr(var, 'variable'):
				u = var.variable
				name = param_map[id(u)] if params is not None else ''
				node_name = '%s\n %s' % (name, size_to_str(u.size()))
				dot.node(str(id(var)), node_name, fillcolor='lightblue')
			else:
				dot.node(str(id(var)), str(type(var).__name__))
			seen.add(var)
			if hasattr(var, 'next_functions'):
				for u in var.next_functions:
					if u[0] is not None:
						dot.edge(str(id(u[0])), str(id(var)))
						add_nodes(u[0])
			if hasattr(var, 'saved_tensors'):
				for t in var.saved_tensors:
					dot.edge(str(id(t)), str(id(var)))
					add_nodes(t)
	#print(var)
	add_nodes(var.grad_fn)
	return dot


def generate_g(model, x):
	delete_allpths(pth_dir=None)
	print('\n=========================== Store network model Results Start =========================')
	y = model(x)
	print('=========================== Store network model Results End ===========================\n')
	if 'GoogLeNet' in str(model).split('\n')[0]:
		g = make_dot(y[0])
		return g
	else:
		g = make_dot(y)
		return g

@torch.no_grad()
def exchange_name(name):
	if 'Relu' in name:
		return 'ReLU'
	elif 'AddmmBackward' in name:
		return 'Linear'
	elif 'ViewBackward' in name:
		return 'View'
	elif 'Mean' in name or 'Avg' in name:
		return 'AvgPool2d'
	elif 'BatchNorm' in name:
		return 'BatchNorm2d'
	elif 'Conv' in name:
		return 'Conv2d'
	elif 'MaxPool' in name:
		return 'MaxPool2d'
	elif 'MulBackward' in name:
		return 'Dropout_2'
	elif 'DivBackward' in name:
		return 'Dropout_1'
	elif 'AddBackward' in name:
		return 'Add'
	elif 'Cat' in name:
		return 'Cat'
	elif 'Hardtanh' in name:
		return 'ReLU6'
	else:
		return 'None'


@torch.no_grad()
def generate_connections(g):
	graph = str(g).split('\n')

	labels = {}
	connections = []
	for i in range(len(graph)):
		if 'label' in graph[i] and graph[i][-1] == '"':
			labels[(graph[i]+graph[i+1][1:]).split('\t')[1].split(' ')[0]]=\
				(graph[i]+graph[i+1][1:]).split('\t')[1].split('"')[1]
		if 'label' in graph[i] and graph[i][-1] == ']':
			labels[graph[i].split('\t')[1].split(' ')[0]]=\
				graph[i].split('\t')[1].split('=')[1].split(']')[0]
	for i in range(len(graph)):	
		if '->' in graph[i]:
			connections.append({labels[graph[i].split('\t')[1].split(' -> ')[0]]+'_'+\
				graph[i].split('\t')[1].split(' -> ')[0]:\
				labels[graph[i].split('\t')[1].split(' -> ')[1]]+'_'+\
				graph[i].split('\t')[1].split(' -> ')[1]})

	pop_index = []
	for i in range(len(connections)):
		item_key = list(connections[i].keys())[0]
		if '(' in item_key or 'TBackward' in item_key:
			pop_index.append(connections[i])
	for i in range(len(pop_index)-1, -1, -1):
		connections.remove(pop_index[i])

	new_connections = []
	for item in connections:
		key, value = list(item.items())[0]
		key1 = exchange_name(key.split('_')[0]) + '_' + key.split('_')[1]
		value1 = exchange_name(value.split('_')[0]) + '_' + value.split('_')[1]
		if 'None' in key1 or 'None' in value1:
			print('Not completed for '+key+' or '+value+'! Check exchange_name function!')
			exit()
		new_connections.append({key1: value1})

	if not len(new_connections) == len(connections):
		print('Generate connections not done! Check generate_connections function!')
		exit()

	new_connections.insert(0, {list(new_connections[0].values())[0]: None})
	new_connections.append({'None': 'None'})
	return connections, new_connections

@torch.no_grad()
def get_split_connections(connections):
	return_connections = []
	tmp_split = []
	for i in range(len(connections)):
		item = connections[i]
		if len(tmp_split) == 0:
			tmp_split.append(item)
			continue
		value = list(item.values())[0]
		last_key = list(tmp_split[-1].keys())[0]
		if value == last_key:
			tmp_split.append(item)
		else:
			return_connections.append(tmp_split)
			tmp_split = [item]
	return return_connections

@torch.no_grad()
def find_start_end(list_dic_key_value, i, j):
	key1 = list(list_dic_key_value[i].values())[0]
	key2 = list(list_dic_key_value[j].keys())[0]
	start = 0
	end = len(list_dic_key_value)-1
	for index in range(len(list_dic_key_value)):
		if key1 == list(list_dic_key_value[index].keys())[0]:
			start = index
			break
	for index in range(len(list_dic_key_value)):
		if key2 == list(list_dic_key_value[index].keys())[0]:
			end = index
			break
	return start+1, end-1

@torch.no_grad()
def merge_connections(connections):
	import copy
	last_connections = copy.deepcopy(connections)
	connections.append({'None':'None'})

	num_Throwed = 0
	notchoosed = []
	print('\n=========================== Restore network model Start ===============================')
	for i in range(len(connections)):
		print('# Restore network model: processing {}/{}'.format(i, len(connections)-1))
		item_key = list(connections[i].keys())[0]
		if not 'None' in item_key:
			if i == 0:
				pass
			else:
				last_item_key = list(connections[i-1].keys())[0]
				if not connections[i][item_key] == last_item_key:
					for j in range(i+1, len(connections)):
						if not list(connections[j].values())[0] == list(connections[j-1].keys())[0]:
							notchoosed.append(i)
							start, end = find_start_end(connections, i, j-1)
							tmp = []
							tmp.append(connections[start:end+1])
							tmp.append(connections[i:j-1])
							last_connections[start:end+1] = [tmp]
							for kk in range(end-start):
								last_connections.insert(start, 'Throwed')
								num_Throwed += 1
							break


	if not notchoosed == []:
		last_connections = last_connections[:notchoosed[0]]
	else:
		pass
	for i in range(num_Throwed):
		last_connections.remove('Throwed')
	if last_connections[-1] == {'None': 'None'}:
		last_connections.remove({'None': 'None'})
	
	print('=========================== Restore network model End =================================\n')
	return last_connections

@torch.no_grad()
def find_next_layer_by_name(layers, name, start_i):
	for i in range(start_i, len(layers)):
		layer = layers[i]
		if name in str(layer):
			return layer, i

@torch.no_grad()
def get_layers(last_connections, model):
	return_layers = []
	tmp_layers = []
	for layer in model.modules():
		if not ':' in str(layer):
			tmp_layers.append(layer)

	index_tmp_layers = 0

	for i in range(len(last_connections)-1, -1, -1):
		if not isinstance(last_connections[i], list):
			current_layer_name = list(last_connections[i].keys())[0].split('_')[0]
			if 'ReLU' in current_layer_name:
				return_layers.insert(0, torch.nn.ReLU(inplace=True))
			elif 'Add' in current_layer_name:
				return_layers.insert(0, 'Add')
			elif 'View' in current_layer_name:
				return_layers.insert(0, 'View')
			else:
				tmp = find_next_layer_by_name(tmp_layers, current_layer_name, index_tmp_layers)
				return_layers.insert(0, tmp[0])
				if isinstance(last_connections[i-1], list):
					index_tmp_layers = tmp[1] + 1
				elif not list(last_connections[i-1].keys())[0].split('_')[0] == 'Dropout':
					index_tmp_layers = tmp[1] + 1
		else:
			return_layers.insert(0, [])
			for j in range(len(last_connections[i])):
				
				return_layers[0].append([])
				
				if len(last_connections[i][j]) == 0:
					continue
				for k in range(len(last_connections[i][j])-1, -1, -1):
					current_layer_name = list(last_connections[i][j][k].keys())[0].split('_')[0]
					if 'ReLU' in current_layer_name:
						return_layers[0][j].insert(0, torch.nn.ReLU(inplace=True))
					elif 'Add' in current_layer_name:
						return_layers[0][j].insert(0, 'Add')
					elif 'View' in current_layer_name:
						return_layers.insert(0, 'View')
					else:
						tmp = find_next_layer_by_name(tmp_layers, current_layer_name, index_tmp_layers)
						return_layers[0][j].insert(0, tmp[0])
						if not list(last_connections[i][j][k-1].keys())[0].split('_')[0] == 'Dropout':
							index_tmp_layers = tmp[1] + 1

	return return_layers



@torch.no_grad()
def get_tensors(last_connections):
	tensors = get_featuremap(featuremap_dir=None)
	index_tensors = 0
	import copy
	last_tensors = copy.deepcopy(last_connections)

	for i in range(len(last_connections)-1, -1, -1):
		if not isinstance(last_connections[i], list):
			current_layer_name = list(last_connections[i].keys())[0].split('_')[0]
			if 'Add' in current_layer_name:
				last_tensors[i] = 'Add'
			elif 'View' in current_layer_name:
				last_tensors[i] = 'View'
			else:
				last_tensors[i] = tensors[index_tensors]
				index_tensors += 1

		else:
			for j in range(len(last_connections[i])):
				if len(last_connections[i][j]) == 0:
					continue
				for k in range(len(last_connections[i][j])-1, -1, -1):
					current_layer_name = list(last_connections[i][j][k].keys())[0].split('_')[0]
					if 'Add' in current_layer_name:
						last_tensors[i][j][k] = 'Add'
					elif 'View' in current_layer_name:
						last_tensors[i][j][k] = 'View'
					else:
						last_tensors[i][j][k] = tensors[index_tensors]
						index_tensors += 1

	for i in range(len(last_tensors)-1, -1, -1):
		if isinstance(last_tensors[i], str):
			# Add or View
			if last_tensors[i] == 'Add':
				last_tensors[i] = last_tensors[i+1][0][0] + last_tensors[i+1][1][0]
			if last_tensors[i] == 'View':
				last_tensors[i] = last_tensors[i+1].view(last_tensors[i+1].size(0), -1)
		elif isinstance(last_tensors[i], list):
			for j in range(len(last_tensors[i])):
				if len(last_tensors[i][j]) == 0:
					last_tensors[i][j].append(last_tensors[i+1])

	return last_tensors

@torch.no_grad()
def get_structure_parameters(return_layers):
	import copy
	parameters = copy.deepcopy(return_layers)
	fc_conv_weights = copy.deepcopy(return_layers)

	for i in range(len(return_layers)):
		layer = return_layers[i]
		if isinstance(layer, nn.Conv2d): 
			layer_name = 'Conv2d'
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
			fc_conv_weights[i] = layer.weight
			parameters[i] = Conv2d_params
		elif isinstance(layer, nn.ReLU): 
			layer_name = 'ReLU'
			parameters[i] = {'layer_name': layer_name}
		elif layer == 'Add': 
			layer_name = 'Add'
			parameters[i] = {'layer_name': layer_name}
		elif layer == 'View': 
			layer_name = 'View'
			parameters[i] = {'layer_name': layer_name}
		elif layer == 'Cat': 
			layer_name = 'Cat'
			parameters[i] = {'layer_name': layer_name}
		elif isinstance(layer, nn.MaxPool2d): 
			layer_name = 'MaxPool2d'
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
			parameters[i] = MaxPool2d_params
		elif isinstance(layer, nn.AvgPool2d): 
			layer_name = 'AvgPool2d'
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
			parameters[i] = AvgPool2d_params
		elif isinstance(layer, nn.Dropout): 
			layer_name = 'Dropout'
			Dropout_params = {}
			Dropout_params['layer_name'] = layer_name
			# p
			p = layer.__dict__.get('p')
			Dropout_params['p'] = p
			# return
			parameters[i] = Dropout_params
		elif isinstance(layer, nn.BatchNorm2d): 
			layer_name = 'BatchNorm2d'
			BatchNorm2d_params = {}
			BatchNorm2d_params['layer_name'] = layer_name
			# num_features
			num_features = layer.__dict__.get('num_features')
			BatchNorm2d_params['num_features'] = num_features
			# eps
			eps = layer.__dict__.get('eps')
			BatchNorm2d_params['eps'] = eps
			# return
			fc_conv_weights[i] = layer.weight
			parameters[i] = BatchNorm2d_params
		elif isinstance(layer, nn.Linear): 
			layer_name = 'Linear'
			Linear_params = {}
			Linear_params['layer_name'] = layer_name
			# in_features
			in_features = layer.__dict__.get('in_features')
			Linear_params['in_features'] = in_features
			# out_features
			out_features = layer.__dict__.get('out_features')
			Linear_params['out_features'] = out_features
			# return
			fc_conv_weights[i] = layer.weight
			parameters[i] = Linear_params
		elif isinstance(layer, nn.AdaptiveAvgPool2d): 
			layer_name = 'AdaptiveAvgPool2d'
			AdaptiveAvgPool2d_params = {}
			AdaptiveAvgPool2d_params['layer_name'] = layer_name
			# output_size
			output_size = layer.__dict__.get('output_size')
			if not isinstance(output_size, tuple):
				AdaptiveAvgPool2d_params['output_size'] = (output_size, output_size)
			else:
				AdaptiveAvgPool2d_params['output_size'] = output_size
			# return
			parameters[i] = AdaptiveAvgPool2d_params
		elif isinstance(layer, list):
			for j in range(len(layer)): 
				for k in range(len(layer[j])): 
					tmp_layer = layer[j][k]
					###
					if isinstance(tmp_layer, nn.Conv2d): 
						layer_name = 'Conv2d'
						Conv2d_params = {}
						Conv2d_params['layer_name'] = layer_name
						# in_channel
						in_channel = tmp_layer.__dict__.get('in_channels')
						Conv2d_params['in_channel'] = in_channel
						# out_channel
						out_channel = tmp_layer.__dict__.get('out_channels')
						Conv2d_params['out_channel'] = out_channel
						# kernel_size
						kernel_size = tmp_layer.__dict__.get('kernel_size')
						if not isinstance(kernel_size, tuple):
							Conv2d_params['kernel_size'] = (kernel_size, kernel_size)
						else:
							Conv2d_params['kernel_size'] = kernel_size
						# stride
						stride = tmp_layer.__dict__.get('stride')
						if not isinstance(stride, tuple):
							Conv2d_params['stride'] = (stride, stride)
						else:
							Conv2d_params['stride'] = stride
						# padding
						padding = tmp_layer.__dict__.get('padding')
						if not isinstance(padding, tuple):
							Conv2d_params['padding'] = (padding, padding)
						else:
							Conv2d_params['padding'] = padding
						# return
						fc_conv_weights[i][j][k] = tmp_layer.weight
						parameters[i][j][k] = Conv2d_params
					elif isinstance(tmp_layer, nn.ReLU): 
						layer_name = 'ReLU'
						parameters[i][j][k] = {'layer_name': layer_name}
					elif tmp_layer == 'Add': 
						layer_name = 'Add'
						parameters[i][j][k] = {'layer_name': layer_name}
					elif tmp_layer == 'View': 
						layer_name = 'View'
						parameters[i][j][k] = {'layer_name': layer_name}
					elif tmp_layer == 'Cat': 
						layer_name = 'Cat'
						parameters[i][j][k] = {'layer_name': layer_name}
					elif isinstance(tmp_layer, nn.MaxPool2d): 
						layer_name = 'MaxPool2d'
						MaxPool2d_params = {}
						MaxPool2d_params['layer_name'] = layer_name
						# kernel_size
						kernel_size = tmp_layer.__dict__.get('kernel_size')
						if not isinstance(kernel_size, tuple):
							MaxPool2d_params['kernel_size'] = (kernel_size, kernel_size)
						else:
							MaxPool2d_params['kernel_size'] = kernel_size
						# stride
						stride = tmp_layer.__dict__.get('stride')
						if not isinstance(stride, tuple):
							MaxPool2d_params['stride'] = (stride, stride)
						else:
							MaxPool2d_params['stride'] = stride
						# padding
						padding = tmp_layer.__dict__.get('padding')
						if not isinstance(padding, tuple):
							MaxPool2d_params['padding'] = (padding, padding)
						else:
							MaxPool2d_params['padding'] = padding
						# return
						parameters[i][j][k] = MaxPool2d_params
					elif isinstance(tmp_layer, nn.AvgPool2d): 
						layer_name = 'AvgPool2d'
						AvgPool2d_params = {}
						AvgPool2d_params['layer_name'] = layer_name
						# kernel_size
						kernel_size = tmp_layer.__dict__.get('kernel_size')
						if not isinstance(kernel_size, tuple):
							AvgPool2d_params['kernel_size'] = (kernel_size, kernel_size)
						else:
							AvgPool2d_params['kernel_size'] = kernel_size
						# stride
						stride = tmp_layer.__dict__.get('stride')
						if not isinstance(stride, tuple):
							AvgPool2d_params['stride'] = (stride, stride)
						else:
							AvgPool2d_params['stride'] = stride
						# padding
						padding = tmp_layer.__dict__.get('padding')
						if not isinstance(padding, tuple):
							AvgPool2d_params['padding'] = (padding, padding)
						else:
							AvgPool2d_params['padding'] = padding
						# return
						parameters[i][j][k] = AvgPool2d_params
					elif isinstance(tmp_layer, nn.Dropout): 
						layer_name = 'Dropout'
						Dropout_params = {}
						Dropout_params['layer_name'] = layer_name
						# p
						p = tmp_layer.__dict__.get('p')
						Dropout_params['p'] = p
						# return
						parameters[i][j][k] = Dropout_params
					elif isinstance(tmp_layer, nn.BatchNorm2d): 
						layer_name = 'BatchNorm2d'
						BatchNorm2d_params = {}
						BatchNorm2d_params['layer_name'] = layer_name
						# num_features
						num_features = tmp_layer.__dict__.get('num_features')
						BatchNorm2d_params['num_features'] = num_features
						# eps
						eps = tmp_layer.__dict__.get('eps')
						BatchNorm2d_params['eps'] = eps
						# return
						fc_conv_weights[i][j][k] = tmp_layer.weight
						parameters[i][j][k] = BatchNorm2d_params
					elif isinstance(tmp_layer, nn.Linear): 
						layer_name = 'Linear'
						Linear_params = {}
						Linear_params['layer_name'] = layer_name
						# in_features
						in_features = tmp_layer.__dict__.get('in_features')
						Linear_params['in_features'] = in_features
						# out_features
						out_features = tmp_layer.__dict__.get('out_features')
						Linear_params['out_features'] = out_features
						# return
						fc_conv_weights[i][j][k] = tmp_layer.weight
						parameters[i][j][k] = Linear_params
					elif isinstance(tmp_layer, nn.AdaptiveAvgPool2d): 
						layer_name = 'AdaptiveAvgPool2d'
						AdaptiveAvgPool2d_params = {}
						AdaptiveAvgPool2d_params['layer_name'] = layer_name
						# output_size
						output_size = tmp_layer.__dict__.get('output_size')
						if not isinstance(output_size, tuple):
							AdaptiveAvgPool2d_params['output_size'] = (output_size, output_size)
						else:
							AdaptiveAvgPool2d_params['output_size'] = output_size
						# return
						parameters[i][j][k] = AdaptiveAvgPool2d_params
					###
		else:
			print('The layer has not been processed in get_structure_parameters!')
	return parameters, fc_conv_weights



def gradient_backward_v2(model, img, label, num_class=1000, g_view=False):
	x = Variable(img)
	g = generate_g(model, x)

	if g_view:
		g.view()

	delete_allpths(pth_dir=None)
	print('\n=========================== Generate Tensors Start ====================================')
	result = model(img)
	print('=========================== Generate Tensors End ======================================\n')
	Loss = nn.CrossEntropyLoss()
	if 'GoogLeNet' in str(model).split('\n')[0]:
		loss_torch = Loss(result[0], label)
	else:
		loss_torch = Loss(result, label)

	_, connections = generate_connections(g)
	last_connections = merge_connections(connections)
	return_layers = get_layers(last_connections, model)
	return_tensors = get_tensors(last_connections)
	parameters, fc_conv_weights = get_structure_parameters(return_layers)

	'''
	print('================')
	for i in range(len(last_connections)):
		print(i, last_connections[i])
	print('================')

	print('================')
	for i in range(len(return_layers)):
		print(i, return_layers[i])
	print('================')

	print('================')
	for i in range(len(parameters)):
		print(i, parameters[i])
	print('================')
	
	print('================')
	for i in range(len(return_tensors)):
		if not isinstance(return_tensors[i], list) and not isinstance(return_tensors[i], str):
			print('=========', i, return_tensors[i].shape)
	print('================')
	'''

	import copy
	return_dz = copy.deepcopy(last_connections)
	featuremap = return_tensors
	featuremap.append(img)

	
	
	y_true = F.one_hot(label, num_classes=num_class).float()

	loss, dLoss_dz = cross_entropy_loss(featuremap[0], y_true)
	featuremap.pop(0)
	return_dz.append(dLoss_dz)
	
	#####################tensors
	'''
	for i in range(len(last_connections)):
		print(last_connections[i])
	
	for i in range(len(featuremap)):
		if not isinstance(featuremap[i], list):
			print('=========', i, featuremap[i].shape)
		else:
			for j in range(len(featuremap[i])):
				for k in range(len(featuremap[i][j])):
					print('  =========', i, j, k, featuremap[i][j][k].shape)
	'''
	#####################
	# 前面n层倒序遍历
	for i in range(len(parameters)):
		layer = parameters[i]
		if not isinstance(layer, list):
			print('\n======================== {0:3} Layer: '.format(str(len(parameters)-1-i))+'{0:11}'.format(layer['layer_name'])+' Backward Start ========================')
			if layer['layer_name'] == 'Conv2d':
				z = featuremap[i]
				weight_z = fc_conv_weights[i]
				try:
					padding = layer['padding']
				except:
					padding = (0, 0)
				stride = layer['stride']
				dLoss_dz, dLoss_dW, dLoss_dB = conv_backward(dLoss_dz, weight_z, z, padding, stride)
				return_dz[i] = dLoss_dz

			elif layer['layer_name'] == 'ReLU':
				z = featuremap[i]
				dLoss_dz = relu_backward(dLoss_dz, z)
				return_dz[i] = dLoss_dz
			elif layer['layer_name'] == 'MaxPool2d':
				z = featuremap[i]
				pooling = layer['kernel_size']
				stride = layer['stride']
				padding = layer['padding']
				dLoss_dz = max_pooling_backward(dLoss_dz, z, pooling, stride, padding)
				return_dz[i] = dLoss_dz
			elif layer['layer_name'] == 'AvgPool2d':
				z = featuremap[i]
				pooling = layer['kernel_size']
				stride = layer['stride']
				padding = layer['padding']
				dLoss_dz = average_pooling_backward(dLoss_dz, z, pooling, stride, padding)
				return_dz[i] = dLoss_dz
			elif layer['layer_name'] == 'Linear':
				weight_z = fc_conv_weights[i]
				z = featuremap[i]
				dLoss_dz, dLoss_dW, dLoss_dB = fc_backward(dLoss_dz, z, weight_z)
				return_dz[i] = dLoss_dz

			elif layer['layer_name'] == 'View':
				last_z = featuremap[i+1]
				if 'Pool' in parameters[i+1]['layer_name']:
					params = (parameters[i+1]['kernel_size'], parameters[i+1]['stride'], parameters[i+1]['padding'])
				else:
					params = None
				dLoss_dz = view_backward(dLoss_dz, last_z, params)
				return_dz[i] = dLoss_dz
			elif layer['layer_name'] == 'Add':
				dLoss_dz = add_backward(dLoss_dz)
				return_dz[i] = dLoss_dz
			elif layer['layer_name'] == 'Dropout':	
				if parameters[i-1]['layer_name'] == 'Dropout':
					return_dz[i] = dLoss_dz
					print('# Skip this layer because the layer has been calcualted!')
					print('======================== {0:3} Layer: '.format(str(len(parameters)-1-i))+'{0:11}'.\
						  format(layer['layer_name'])+' Backward End ==========================')
					continue
				p = layer['p']
				mask = featuremap[i]
				dLoss_dz = dropback_backward(dLoss_dz, mask, p)
				return_dz[i] = dLoss_dz	
			elif layer['layer_name'] == 'BatchNorm2d':		
				eps = layer['eps']
				z = featuremap[i]
				gamma = fc_conv_weights[i]
				dLoss_dz = batchnorm2d_backward(dLoss_dz, z, eps, gamma)
				return_dz[i] = dLoss_dz	
			print('======================== {0:3} Layer: '.format(str(len(parameters)-1-i))+'{0:11}'.format(layer['layer_name'])+' Backward End ==========================')
		elif isinstance(layer, list):

			import copy
			tmp_dLoss_dz = []
			for j in range(len(layer)):

				tmp_dLoss_dz.append(copy.deepcopy(dLoss_dz))
				for k in range(len(layer[j])):

					tmp_layer = layer[j][k]
					print('\n=========================== {0:3} Branch: '.format(str(len(parameters)-1-i))+'{0:11}'.format(tmp_layer['layer_name'])+' Backward Start ====================')
					if tmp_layer['layer_name'] == 'Conv2d':
						if k+1 >= len(featuremap[i-1][j]):
							z = featuremap[i]
						else:
							z = featuremap[i-1][j][k+1]
						weight_z = fc_conv_weights[i][j][k]
						try:
							padding = tmp_layer['padding']
						except:
							padding = (0, 0)
						stride = tmp_layer['stride']
						tmp_dLoss_dz[-1], dLoss_dW, dLoss_dB = conv_backward(tmp_dLoss_dz[-1], weight_z, z, padding, stride)
						return_dz[i][j][k] = tmp_dLoss_dz[-1]
					elif tmp_layer['layer_name'] == 'ReLU':
						z = featuremap[i-1][j][k+1]
						tmp_dLoss_dz[-1] = relu_backward(tmp_dLoss_dz[-1], z)
						return_dz[i][j][k] = tmp_dLoss_dz[-1]
					elif tmp_layer['layer_name'] == 'BatchNorm2d':	
						eps = tmp_layer['eps']
						z = featuremap[i-1][j][k+1]
						gamma = fc_conv_weights[i][j][k]
						tmp_dLoss_dz[-1] = batchnorm2d_backward(tmp_dLoss_dz[-1], z, eps, gamma)
						return_dz[i][j][k] = tmp_dLoss_dz[-1]	
					print('=========================== {0:3} Branch: '.format(str(len(parameters)-1-i))+'{0:11}'.format(tmp_layer['layer_name'])+' Backward End ======================')

			#print(tmp_dLoss_dz[0].shape, tmp_dLoss_dz[1].shape)
			dLoss_dz = tmp_dLoss_dz[0] + tmp_dLoss_dz[1]
		else:
			print('Not completed in gradient_backward!')
		

	print('# Torch calculated loss: ', loss_torch.detach().numpy())
	loss_torch.backward()
	
	if 'VGG' in str(model) or 'AlexNet' in str(model):
		print(judge_tensors_equal(dLoss_dW, model.features[0].weight.grad))
	elif 'ResNet' in str(model):
		print(judge_tensors_equal(dLoss_dW, model.conv1.weight.grad))


	delete_allpths(pth_dir=None)
	return return_dz, dLoss_dW, dLoss_dB
