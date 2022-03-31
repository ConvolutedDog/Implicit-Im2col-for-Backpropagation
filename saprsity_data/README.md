# Generate parameters or some data for convolutional layers.



## File Structure.

```shell
./Config: The generated configs of convolutional layers. 
./Result: The generated data of this project.
./addrSparse.py: Source script of this project.
```



## How to install the necessary environment?

```shell
$ pip install -r requirements
```



## How to run?

```shell
$ python addrSparse.py
```



## Generated content.

### 1. "./Result/*.csv"

| Header of Table                           | Meaning                                                      |
| :---------------------------------------- | :----------------------------------------------------------- |
| input_size                                | Parameters of inference.                                     |
| output_size                               | Parameters of inference.                                     |
| kernel_size                               | Parameters of inference.                                     |
| stride                                    | Parameters of inference.                                     |
| padding                                   | Parameters of inference.                                     |
| forward_featuremap_internal_padding       | Length of zero-insertions of feature map for inference.      |
| forward_featuremap_external_padding       | Length of zero-paddings of feature map for inference.        |
| forward_weight_internal_padding           | Length of zero-insertions of kernel for inference.           |
| forward_featuremap_elements_after_im2col  | Number of pixels of feature map after im2col (1 image) for inference. |
| forward_featuremap_zeros_after_im2col     | Number of zeros of feature map after im2col (1 image) for inference. |
| loss_featuremap_internal_padding          | Length of zero-insertions of feature map for loss.           |
| loss_featuremap_external_padding          | Length of zero-paddings of feature map for loss.             |
| loss_weight_internal_padding              | Length of zero-insertions of kernel for loss.                |
| loss_featuremap_elements_after_im2col     | Number of pixels of feature map after im2col (1 image) for loss. |
| loss_featuremap_zeros_after_im2col        | Number of zeros of feature map after im2col (1 image) for loss. |
| gradient_featuremap_internal_padding      | Length of zero-insertions of feature map for gradient.       |
| gradient_featuremap_external_padding      | Length of zero-paddings of feature map for gradient.         |
| gradient_weight_internal_padding          | Length of zero-insertions of kernel for gradient.            |
| gradient_featuremap_elements_after_im2col | Number of pixels of feature map after im2col (1 image) for gradient. |
| gradient_featuremap_zeros_after_im2col    | Number of zeros of feature map after im2col (1 image) for gradient. |
| gradient_weight_zeros_after_im2col        | Number of zeros of kernel after im2col for gradient.         |
| forward_featuremap_sparsity_after_im2col  | Sparsity of zeros of feature map after im2col (1 image) for inference. |
| forward_weight_sparsity_after_im2col      | Sparsity of zeros of kernel after im2col (1 image) for inference. |
| loss_featuremap_sparsity_after_im2col     | Sparsity of zeros of feature map after im2col (1 image) for loss. |
| loss_weight_sparsity_after_im2col         | Sparsity of zeros of kernel after im2col (1 image) for loss. |
| gradient_featuremap_sparsity_after_im2col | Sparsity of zeros of feature map after im2col (1 image) for gradient. |
| gradient_weight_sparsity_after_im2col     | Sparsity of zeros of kernel after im2col (1 image) for gradient. |
| forward_featuremap_additional_storage     | Additional storage of zeros of feature map after im2col (1 image) for inference. |
| loss_featuremap_additional_storage        | Additional storage of zeros of feature map after im2col (1 image) for loss. |
| gradient_kernal_additional_storage        | Additional storage of zeros of kernel after im2col (1 image) for gradient. |



### 2. "./Config/*.cfg"

The parameters will be used in the GPU experiments.



## License

This project is licensed under the Apache-2.0 License.