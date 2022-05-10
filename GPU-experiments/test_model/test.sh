#!/bin/sh

python3 test_model.py -model=alexnet -batchsize=1
python3 test_model.py -model=alexnet -batchsize=2
python3 test_model.py -model=alexnet -batchsize=4
python3 test_model.py -model=alexnet -batchsize=8
python3 test_model.py -model=alexnet -batchsize=16

python3 test_model.py -model=densenet201 -batchsize=1
python3 test_model.py -model=densenet201 -batchsize=2
python3 test_model.py -model=densenet201 -batchsize=4
python3 test_model.py -model=densenet201 -batchsize=8
python3 test_model.py -model=densenet201 -batchsize=16

python3 test_model.py -model=mobilenet_v3_large -batchsize=1
python3 test_model.py -model=mobilenet_v3_large -batchsize=2
python3 test_model.py -model=mobilenet_v3_large -batchsize=4
python3 test_model.py -model=mobilenet_v3_large -batchsize=8
python3 test_model.py -model=mobilenet_v3_large -batchsize=16

python3 test_model.py -model=resnet152 -batchsize=1
python3 test_model.py -model=resnet152 -batchsize=2
python3 test_model.py -model=resnet152 -batchsize=4
python3 test_model.py -model=resnet152 -batchsize=8
python3 test_model.py -model=resnet152 -batchsize=16

python3 test_model.py -model=shufflenet_v2_x2_0 -batchsize=1
python3 test_model.py -model=shufflenet_v2_x2_0 -batchsize=2
python3 test_model.py -model=shufflenet_v2_x2_0 -batchsize=4
python3 test_model.py -model=shufflenet_v2_x2_0 -batchsize=8
python3 test_model.py -model=shufflenet_v2_x2_0 -batchsize=16

python3 test_model.py -model=squeezenet1_1 -batchsize=1
python3 test_model.py -model=squeezenet1_1 -batchsize=2
python3 test_model.py -model=squeezenet1_1 -batchsize=4
python3 test_model.py -model=squeezenet1_1 -batchsize=8
python3 test_model.py -model=squeezenet1_1 -batchsize=16
