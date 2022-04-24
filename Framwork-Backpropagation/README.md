# A Framework For AI Backpropagation

## File Structure.

```
./tmp_file: The generated featuremaps of all layers of a CNN model. 
./utils: The source code of our framework.
./example.py: An example will show how to use the framework.
./test_utils_v2.py: For using this framework to perform BP for torchvision's models.
./build: Files required to install the framework.
```

<font color=Red>Please don't modify the file structure !!!</font>

## How to install the necessary environment?

Since our framework needs to modify the source code of torch, we recommend using Anaconda for environmental management. 

First, you should create a new torch environment, named `test`.

```shell
$ conda create -n test
$ conda activate test
```

Next, install the libraries required by our framework.

```shell
$ pip install -r requirements.txt
```

<font color=Red>Next is the most important step: install our framework.</font>

```shell
$ cd build
$ python setup.py install
```

The install process will be:

```shell
(base) ~\Implicit-Im2col-for-Backpropagation-main\Framwork-Backpropagation> conda activate test
(test) ~\Implicit-Im2col-for-Backpropagation-main\Framwork-Backpropagation> cd build
(test) ~\Implicit-Im2col-for-Backpropagation-main\Framwork-Backpropagation\build> python setup.py install
install:  mv D:\Anaconda3\envs\test\lib\site-packages\torch\nn\modules\activation.py activation.py.bakoriginal
install:  cp ./activation.py D:\Anaconda3\envs\test\lib\site-packages\torch\nn\modules\activation.py
install:  mv D:\Anaconda3\envs\test\lib\site-packages\torch\nn\modules\batchnorm.py batchnorm.py.bakoriginal
install:  cp ./batchnorm.py D:\Anaconda3\envs\test\lib\site-packages\torch\nn\modules\batchnorm.py
install:  mv D:\Anaconda3\envs\test\lib\site-packages\torch\nn\modules\dropout.py dropout.py.bakoriginal
install:  cp ./dropout.py D:\Anaconda3\envs\test\lib\site-packages\torch\nn\modules\dropout.py
install:  mv D:\Anaconda3\envs\test\lib\site-packages\torch\nn\modules\linear.py linear.py.bakoriginal
install:  cp ./linear.py D:\Anaconda3\envs\test\lib\site-packages\torch\nn\modules\linear.py
install:  mv D:\Anaconda3\envs\test\lib\site-packages\torch\nn\modules\pooling.py pooling.py.bakoriginal
install:  cp ./pooling.py D:\Anaconda3\envs\test\lib\site-packages\torch\nn\modules\pooling.py
install:  mv D:\Anaconda3\envs\test\lib\site-packages\torch\nn\modules\conv.py conv.py.bakoriginal
install:  cp ./conv.py D:\Anaconda3\envs\test\lib\site-packages\torch\nn\modules\conv.py
install:  cp ./frameworkhelp.py D:\Anaconda3\envs\test\lib\site-packages\torch\nn\modules\frameworkhelp.py
install:  mv D:\Anaconda3\envs\test\lib\site-packages\torchvision\models\alexnet.py alexnet.py.bakoriginal
install:  cp ./alexnet.py D:\Anaconda3\envs\test\lib\site-packages\torchvision\models\alexnet.py
install:  mv D:\Anaconda3\envs\test\lib\site-packages\torchvision\models\resnet.py resnet.py.bakoriginal
install:  cp ./resnet.py D:\Anaconda3\envs\test\lib\site-packages\torchvision\models\resnet.py
install:  mv D:\Anaconda3\envs\test\lib\site-packages\torchvision\models\vgg.py vgg.py.bakoriginal
install:  cp ./vgg.py D:\Anaconda3\envs\test\lib\site-packages\torchvision\models\vgg.py
Install Succeed!
```

## How to run this framework?

Take the example model in `./example.py` to illustrate:

![examplemodel](https://github.com/ConvolutedDog/Implicit-Im2col-for-Backpropagation/blob/main/Framwork-Backpropagation/pictures/examplemodel.PNG)

```shell
(test) ~\Implicit-Im2col-for-Backpropagation-main\Framwork-Backpropagation> python example.py
ExampleNet(
  (features): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): ReLU()
    (2): Linear(in_features=14112, out_features=1000, bias=True)
  )
)
torch.nn.Conv2d save 0.pth
torch.nn.ReLU save 1.pth
torch.nn.MaxPool2d save 2.pth
torch.nn.Dropout.Mask save 3.pth
torch.nn.Dropout save 4.pth
torch.nn.ReLU save 5.pth
torch.nn.Linear save 6.pth

============== Store network model Results Start ============
torch.nn.Conv2d save 0.pth
torch.nn.ReLU save 1.pth
torch.nn.MaxPool2d save 2.pth
torch.nn.Dropout.Mask save 3.pth
torch.nn.Dropout save 4.pth
torch.nn.ReLU save 5.pth
torch.nn.Linear save 6.pth
============== Store network model Results End ==============


============== Generate Tensors Start =======================
torch.nn.Conv2d save 0.pth
torch.nn.ReLU save 1.pth
torch.nn.MaxPool2d save 2.pth
torch.nn.Dropout.Mask save 3.pth
torch.nn.Dropout save 4.pth
torch.nn.ReLU save 5.pth
torch.nn.Linear save 6.pth
============== Generate Tensors End =========================


============== Restore network model Start ==================
# Restore network model: processing 0/9
# Restore network model: processing 1/9
# Restore network model: processing 2/9
# Restore network model: processing 3/9
# Restore network model: processing 4/9
# Restore network model: processing 5/9
# Restore network model: processing 6/9
# Restore network model: processing 7/9
# Restore network model: processing 8/9
# Restore network model: processing 9/9
============== Restore network model End ====================


============== Layer: cross_entropy_loss Start ==============
# y_predict.shape:  [2, 1000]
# y_true.shape:  [2, 1000]
# dLoss_dypred.shape:  [2, 1000]
# Self calculated loss:  6.658211708068848
============== Layer: cross_entropy_loss End ================

=========== 7   Layer: Linear      Backward Start ===========
# next_dz.shape:  [2, 1000]
# z.shape:  [2, 14112]
# weight.shape:  [1000, 14112]
# bias.shape:  [1000]
# dz.shape:  [2, 14112]
# dweight.shape:  [1000, 14112]
# dbias.shape:  [1000]
=========== 7   Layer: Linear      Backward End =============

=========== 6   Layer: ReLU        Backward Start ===========
# next_dz.shape:  [2, 14112]
# z.shape:  [2, 14112]
# dz.shape:  [2, 14112]
=========== 6   Layer: ReLU        Backward End =============

=========== 5   Layer: Dropout     Backward Start ===========
# zeros probability:  0.5
# next_dz.shape:  [2, 14112]
# mask.shape:  [2, 14112]
# dz.shape:  [2, 14112]
=========== 5   Layer: Dropout     Backward End =============

=========== 4   Layer: Dropout     Backward Start ===========
# Skip this layer because the layer has been calcualted!
=========== 4   Layer: Dropout     Backward End =============

=========== 3   Layer: View        Backward Start ===========
# next_dz.shape:  [2, 14112]
# last_z.shape:  [2, 32, 42, 42]
# dz.shape:  [2, 32, 21, 21]
=========== 3   Layer: View        Backward End =============

=========== 2   Layer: MaxPool2d   Backward Start ===========
# next_dz.shape:  [2, 32, 21, 21]
# z.shape:  [2, 32, 42, 42]
# padding:  (0, 0)
# strides:  (2, 2)
# dz.shape:  [2, 32, 42, 42]
=========== 2   Layer: MaxPool2d   Backward End =============

=========== 1   Layer: ReLU        Backward Start ===========
# next_dz.shape:  [2, 32, 42, 42]
# z.shape:  [2, 32, 42, 42]
# dz.shape:  [2, 32, 42, 42]
=========== 1   Layer: ReLU        Backward End =============

=========== 0   Layer: Conv2d      Backward Start ===========
# next_dz.shape:  [2, 32, 42, 42]
# z.shape:  [2, 3, 44, 44]
# weight.shape:  [32, 3, 3, 3]
# bias.shape:  [32]
# padding:  (0, 0)
# strides:  (1, 1)
# dz.shape:  [2, 3, 44, 44]
# dweight.shape:  [32, 3, 3, 3]
# dbias.shape:  [32]
=========== 0   Layer: Conv2d      Backward End =============
# Torch calculated loss:  6.6582117
```

The operation details can be seen in `./example.py`.

Also you can choose a real model in `./test_utils_v2.py` to perform back propagation. Now, the supported models of our framework are as follows:

```python
# __all__ is all the models we now support.
__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2',
           'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 
           'vgg16_bn', 'vgg19', 'vgg19_bn', 'alexnet']
```

Note that the size of input images should satisfy:

```
# Alexnet can use (224, 224) pictures or (228, 228) pictures;
# VGG series network models can use (224, 224) or (228, 228) pictures;
# ResNet and ResNext series network models can use (228, 228) pictures.
# Choose the input shape (BacthSize x InputChannel x Height x Width).
```

Taking `AlexNet` as an example, the operation effect is as follows:

```shell
(test) ~\Implicit-Im2col-for-Backpropagation-main\Framwork-Backpropagation> python test_utils_v2.py
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(12, 12), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (13): BatchNorm2d(256, eps=0.001, momentum=1.0, affine=True, track_running_stats=True)
  )
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

=========== Store network model Results Start ===============
torch.nn.Conv2d save 0.pth
torch.nn.ReLU save 1.pth
torch.nn.MaxPool2d save 2.pth
torch.nn.Conv2d save 3.pth
torch.nn.ReLU save 4.pth
torch.nn.MaxPool2d save 5.pth
torch.nn.Conv2d save 6.pth
torch.nn.ReLU save 7.pth
torch.nn.Conv2d save 8.pth
torch.nn.ReLU save 9.pth
torch.nn.Conv2d save 10.pth
torch.nn.ReLU save 11.pth
torch.nn.MaxPool2d save 12.pth
torch.nn.BatchNorm save 13.pth
torch.nn.Dropout.Mask save 14.pth
torch.nn.Dropout save 15.pth
torch.nn.Linear save 16.pth
torch.nn.ReLU save 17.pth
torch.nn.Dropout.Mask save 18.pth
torch.nn.Dropout save 19.pth
torch.nn.Linear save 20.pth
torch.nn.ReLU save 21.pth
torch.nn.Linear save 22.pth
=========== Store network model Results End =================


=========== Generate Tensors Start ==========================
torch.nn.Conv2d save 0.pth
torch.nn.ReLU save 1.pth
torch.nn.MaxPool2d save 2.pth
torch.nn.Conv2d save 3.pth
torch.nn.ReLU save 4.pth
torch.nn.MaxPool2d save 5.pth
torch.nn.Conv2d save 6.pth
torch.nn.ReLU save 7.pth
torch.nn.Conv2d save 8.pth
torch.nn.ReLU save 9.pth
torch.nn.Conv2d save 10.pth
torch.nn.ReLU save 11.pth
torch.nn.MaxPool2d save 12.pth
torch.nn.BatchNorm save 13.pth
torch.nn.Dropout.Mask save 14.pth
torch.nn.Dropout save 15.pth
torch.nn.Linear save 16.pth
torch.nn.ReLU save 17.pth
torch.nn.Dropout.Mask save 18.pth
torch.nn.Dropout save 19.pth
torch.nn.Linear save 20.pth
torch.nn.ReLU save 21.pth
torch.nn.Linear save 22.pth
=========== Generate Tensors End ============================


=========== Restore network model Start =====================
# Restore network model: processing 0/25
# Restore network model: processing 1/25
# Restore network model: processing 2/25
# Restore network model: processing 3/25
# Restore network model: processing 4/25
# Restore network model: processing 5/25
# Restore network model: processing 6/25
# Restore network model: processing 7/25
# Restore network model: processing 8/25
# Restore network model: processing 9/25
# Restore network model: processing 10/25
# Restore network model: processing 11/25
# Restore network model: processing 12/25
# Restore network model: processing 13/25
# Restore network model: processing 14/25
# Restore network model: processing 15/25
# Restore network model: processing 16/25
# Restore network model: processing 17/25
# Restore network model: processing 18/25
# Restore network model: processing 19/25
# Restore network model: processing 20/25
# Restore network model: processing 21/25
# Restore network model: processing 22/25
# Restore network model: processing 23/25
# Restore network model: processing 24/25
# Restore network model: processing 25/25
=========== Restore network model End =======================


=========== Layer: cross_entropy_loss Start =================
# y_predict.shape:  [2, 1000]
# y_true.shape:  [2, 1000]
# dLoss_dypred.shape:  [2, 1000]
# Self calculated loss:  6.896736145019531
=========== Layer: cross_entropy_loss End ===================

=========== 23  Layer: Linear      Backward Start ===========
# next_dz.shape:  [2, 1000]
# z.shape:  [2, 4096]
# weight.shape:  [1000, 4096]
# bias.shape:  [1000]
# dz.shape:  [2, 4096]
# dweight.shape:  [1000, 4096]
# dbias.shape:  [1000]
=========== 23  Layer: Linear      Backward End =============

=========== 22  Layer: ReLU        Backward Start ===========
# next_dz.shape:  [2, 4096]
# z.shape:  [2, 4096]
# dz.shape:  [2, 4096]
=========== 22  Layer: ReLU        Backward End =============

=========== 21  Layer: Linear      Backward Start ===========
# next_dz.shape:  [2, 4096]
# z.shape:  [2, 4096]
# weight.shape:  [4096, 4096]
# bias.shape:  [4096]
# dz.shape:  [2, 4096]
# dweight.shape:  [4096, 4096]
# dbias.shape:  [4096]
=========== 21  Layer: Linear      Backward End =============

=========== 20  Layer: Dropout     Backward Start ===========
# zeros probability:  0.5
# next_dz.shape:  [2, 4096]
# mask.shape:  [2, 4096]
# dz.shape:  [2, 4096]
=========== 20  Layer: Dropout     Backward End =============

=========== 19  Layer: Dropout     Backward Start ===========
# Skip this layer because the layer has been calcualted!
=========== 19  Layer: Dropout     Backward End =============

=========== 18  Layer: ReLU        Backward Start ===========
# next_dz.shape:  [2, 4096]
# z.shape:  [2, 4096]
# dz.shape:  [2, 4096]
=========== 18  Layer: ReLU        Backward End =============

=========== 17  Layer: Linear      Backward Start ===========
# next_dz.shape:  [2, 4096]
# z.shape:  [2, 9216]
# weight.shape:  [4096, 9216]
# bias.shape:  [4096]
# dz.shape:  [2, 9216]
# dweight.shape:  [4096, 9216]
# dbias.shape:  [4096]
=========== 17  Layer: Linear      Backward End =============

=========== 16  Layer: Dropout     Backward Start ===========
# zeros probability:  0.5
# next_dz.shape:  [2, 9216]
# mask.shape:  [2, 9216]
# dz.shape:  [2, 9216]
=========== 16  Layer: Dropout     Backward End =============

=========== 15  Layer: Dropout     Backward Start ===========
# Skip this layer because the layer has been calcualted!
=========== 15  Layer: Dropout     Backward End =============

=========== 14  Layer: View        Backward Start ===========
# next_dz.shape:  [2, 9216]
# last_z.shape:  [2, 256, 6, 6]
# dz.shape:  [2, 256, 6, 6]
=========== 14  Layer: View        Backward End =============

=========== 13  Layer: BatchNorm2d Backward Start ===========
# next_dz.shape:  [2, 256, 6, 6]
# z.shape:  [2, 256, 6, 6]
# eps:  0.001
# gamma.shape:  [256]
# dz.shape:  [2, 256, 6, 6]
=========== 13  Layer: BatchNorm2d Backward End =============

=========== 12  Layer: MaxPool2d   Backward Start ===========
# next_dz.shape:  [2, 256, 6, 6]
# z.shape:  [2, 256, 13, 13]
# padding:  (0, 0)
# strides:  (2, 2)
# dz.shape:  [2, 256, 13, 13]
=========== 12  Layer: MaxPool2d   Backward End =============

=========== 11  Layer: ReLU        Backward Start ===========
# next_dz.shape:  [2, 256, 13, 13]
# z.shape:  [2, 256, 13, 13]
# dz.shape:  [2, 256, 13, 13]
=========== 11  Layer: ReLU        Backward End =============

=========== 10  Layer: Conv2d      Backward Start ===========
# next_dz.shape:  [2, 256, 13, 13]
# z.shape:  [2, 256, 13, 13]
# weight.shape:  [256, 256, 3, 3]
# bias.shape:  [256]
# padding:  (1, 1)
# strides:  (1, 1)
# dz.shape:  [2, 256, 13, 13]
# dweight.shape:  [256, 256, 3, 3]
# dbias.shape:  [256]
=========== 10  Layer: Conv2d      Backward End =============

=========== 9   Layer: ReLU        Backward Start ===========
# next_dz.shape:  [2, 256, 13, 13]
# z.shape:  [2, 256, 13, 13]
# dz.shape:  [2, 256, 13, 13]
=========== 9   Layer: ReLU        Backward End =============

=========== 8   Layer: Conv2d      Backward Start ===========
# next_dz.shape:  [2, 256, 13, 13]
# z.shape:  [2, 384, 13, 13]
# weight.shape:  [256, 384, 3, 3]
# bias.shape:  [256]
# padding:  (1, 1)
# strides:  (1, 1)
# dz.shape:  [2, 384, 13, 13]
# dweight.shape:  [256, 384, 3, 3]
# dbias.shape:  [256]
=========== 8   Layer: Conv2d      Backward End =============

=========== 7   Layer: ReLU        Backward Start ===========
# next_dz.shape:  [2, 384, 13, 13]
# z.shape:  [2, 384, 13, 13]
# dz.shape:  [2, 384, 13, 13]
=========== 7   Layer: ReLU        Backward End =============

=========== 6   Layer: Conv2d      Backward Start ===========
# next_dz.shape:  [2, 384, 13, 13]
# z.shape:  [2, 192, 13, 13]
# weight.shape:  [384, 192, 3, 3]
# bias.shape:  [384]
# padding:  (1, 1)
# strides:  (1, 1)
# dz.shape:  [2, 192, 13, 13]
# dweight.shape:  [384, 192, 3, 3]
# dbias.shape:  [384]
=========== 6   Layer: Conv2d      Backward End =============

=========== 5   Layer: MaxPool2d   Backward Start ===========
# next_dz.shape:  [2, 192, 13, 13]
# z.shape:  [2, 192, 27, 27]
# padding:  (0, 0)
# strides:  (2, 2)
# dz.shape:  [2, 192, 27, 27]
=========== 5   Layer: MaxPool2d   Backward End =============

=========== 4   Layer: ReLU        Backward Start ===========
# next_dz.shape:  [2, 192, 27, 27]
# z.shape:  [2, 192, 27, 27]
# dz.shape:  [2, 192, 27, 27]
=========== 4   Layer: ReLU        Backward End =============

=========== 3   Layer: Conv2d      Backward Start ===========
# next_dz.shape:  [2, 192, 27, 27]
# z.shape:  [2, 64, 27, 27]
# weight.shape:  [192, 64, 5, 5]
# bias.shape:  [192]
# padding:  (2, 2)
# strides:  (1, 1)
# dz.shape:  [2, 64, 27, 27]
# dweight.shape:  [192, 64, 5, 5]
# dbias.shape:  [192]
=========== 3   Layer: Conv2d      Backward End =============

=========== 2   Layer: MaxPool2d   Backward Start ===========
# next_dz.shape:  [2, 64, 27, 27]
# z.shape:  [2, 64, 56, 56]
# padding:  (0, 0)
# strides:  (2, 2)
# dz.shape:  [2, 64, 56, 56]
=========== 2   Layer: MaxPool2d   Backward End =============

=========== 1   Layer: ReLU        Backward Start ===========
# next_dz.shape:  [2, 64, 56, 56]
# z.shape:  [2, 64, 56, 56]
# dz.shape:  [2, 64, 56, 56]
=========== 1   Layer: ReLU        Backward End =============

=========== 0   Layer: Conv2d      Backward Start ===========
# next_dz.shape:  [2, 64, 56, 56]
# z.shape:  [2, 3, 228, 228]
# weight.shape:  [64, 3, 12, 12]
# bias.shape:  [64]
# padding:  (2, 2)
# strides:  (4, 4)
# dz.shape:  [2, 3, 228, 228]
# dweight.shape:  [64, 3, 12, 12]
# dbias.shape:  [64]
=========== 0   Layer: Conv2d      Backward End =============
# Torch calculated loss:  6.896736
4D-error-rate:  0.0
```

## How to uninstall the framework?

You can uninstall our framework by this:

```shell
$ cd build
$ python setup.py uninstall
```

## License

This project is licensed under the Apache-2.0 License.
