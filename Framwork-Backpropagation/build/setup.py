import os
import sys

try:
	import torch
	import torchvision
	torch_nn_modules_path = torch.nn.modules.__path__[0]
	torchvision_models_path = torchvision.models.__path__[0]
	f = open('./config.txt', 'w')
	f.writelines(torch_nn_modules_path+'\n')
	f.writelines(torchvision_models_path)
	f.close()
except:
	f = open('./config.txt', 'r')
	config = f.readlines()
	torch_nn_modules_path = config[0].split('\n')[0]
	torchvision_models_path = config[1].split('\n')[0]
	f.close()


torchfiles = ['activation.py', 'batchnorm.py', 'dropout.py', 'linear.py', 'pooling.py', 'conv.py']
torchvisionfiles = ['alexnet.py', 'resnet.py', 'vgg.py']

def install():
	pthpath = os.path.join(os.getcwd().split('build')[0], 'tmp_file')
	if not os.path.exists(pthpath):
		print('install: mkdir ../tmp_file')
		os.mkdir(pthpath)

	f = open('./frameworkhelp.py', 'r')
	content = f.readlines()
	f.close()
	for i in range(len(content)):
		line = content[i]
		if 'pth_dir = ' in line:
			line = (line.split('pth_dir = ')[0] + 'pth_dir =' + '\"' + pthpath + '\"' + '\n').replace('\\', '\\\\')
			content[i] = line
	f = open('./frameworkhelp.py', 'w')
	for line in content:
		f.writelines(line)
	f.close()
	
	# torch install
	for file in torchfiles:
		file_path = os.path.join(torch_nn_modules_path, file)
		
		if file+'.bakoriginal' in os.listdir(torch_nn_modules_path):
			print("\033[0;31;40mERROR: You have installed our framework!\nUse 'python .\setup.py uninstall' to uninstall.\033[0m")
			exit()
		else:
			cmd = "mv {} {}" .format(file_path, file_path+'.bakoriginal')
			print('install: ', cmd)
			os.system(cmd)

			ourfile = os.path.join('./', file)
			cmd = "cp {} {}" .format(ourfile, file_path)
			print('install: ', cmd)
			os.system(cmd)

	file = 'frameworkhelp.py'
	file_path = os.path.join(torch_nn_modules_path, file)
		
	ourfile = os.path.join('./', file)
	cmd = "cp {} {}" .format(ourfile, file_path)
	print('install: ', cmd)
	os.system(cmd)
	# torchvision install
	for file in torchvisionfiles:
		file_path = os.path.join(torchvision_models_path, file)
		
		cmd = "mv {} {}" .format(file_path, file_path+'.bakoriginal')
		print('install: ', cmd)
		os.system(cmd)

		ourfile = os.path.join('./', file)
		cmd = "cp {} {}" .format(ourfile, file_path)
		print('install: ', cmd)
		os.system(cmd)

	print("\033[0;32;40mInstall Succeed!\033[0m")


def uninstall():
	# torch uninstall
	for file in torchfiles:
		file_path = os.path.join(torch_nn_modules_path, file)
		
		if file+'.bakoriginal' in os.listdir(torch_nn_modules_path):
			cmd = "mv {} {}" .format(file_path+'.bakoriginal', file_path)
			print('uninstall: ', cmd)
			os.system(cmd)
		else:
			print("\033[0;31;40mERROR: You have not installed our framework!\nUse 'python .\setup.py install' to install.\033[0m")
			exit()

	file = 'frameworkhelp.py'
	file_path = os.path.join(torch_nn_modules_path, file)
	cmd = "rm {}" .format(file_path)
	print('uninstall: ', cmd)
	# torchvision uninstall
	for file in torchvisionfiles:
		file_path = os.path.join(torchvision_models_path, file)
		
		cmd = "mv {} {}" .format(file_path+'.bakoriginal', file_path)
		print('uninstall: ', cmd)
		os.system(cmd)

	print("\033[0;32;40mUninstall Succeed!\033[0m")

if sys.argv[1] == 'install':
	install()
	
elif sys.argv[1] == 'uninstall':
	uninstall()