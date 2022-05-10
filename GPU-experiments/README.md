# GPU-experiments of our paper.

## Environment Preparation
* Tesla A100 GPU

## Build.
```sh
cd build
cmake ../ -DCMAKE_CXX_COMPILER="/usr/local/gcc-9.3.0/bin/g++" -DCMAKE_C_COMPILER="/usr/local/gcc-9.3.0/bin/gcc"
make -j
```

## Run
```sh
cd test_model
sh test.sh
```
Or:
```sh
# Normal im2col with data reorganization of zero-spaces.
./build/GPUImplicit preprocess batchsize C Hi Wi D Kh Kw P S 1
# Our BP-im2col without data reorganization of zero-spaces.
./build/GPUImplicit nopreprocess batchsize C Hi Wi D, Kh Kw P S 1
```
The explanation of relevant variables can be found in our paper.

## Other explanations
Coming soon.

## License

This project is licensed under the Apache-2.0 License.
