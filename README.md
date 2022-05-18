# Implicit-Im2col-for-Backpropagation

## ALL codes have been updated. The contents of this repo will be add soon.

## Analysis-of-sparsity
This project aims to obtain the configuration of the convolutional layers of some models to facilitate the use of GPU experiments. At the same time, we also analyzed how many zero data will be generated by the feature maps and kernel during inference and backpropagation in the conventional systolic array implementation, after im2col is executed, and due to these zero data additional storage overhead incurred. Detailed instructions can be found in the `Analysis-of-sparsity` folder.

## Framwork-Backpropagation
A framework to perform backpropagation designed by us. You maybe kown how to map the backpropagation process on hardware from this project. 

## GPU-experiments
GPU experiments of our paper.

## License
This project is licensed under the Apache-2.0 License.
