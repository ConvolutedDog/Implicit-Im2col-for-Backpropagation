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

First, you should create a new torch environment, named `editedtorch`.

```shell
$ conda create -n editedtorch
$ conda activate editedtorch
```

Next, install the libraries required by our framework.

```shell
$ pip install -r requirements.txt
```

<font color=Red>Next is the most important step: modify the source code of torch.</font>

























## License

This project is licensed under the Apache-2.0 License.