# HartleySpectralPooling
 
Python code to reproduce the result of the paper

H. Zhang, J. Ma. [Hartley Spectral Pooling for Deep Learning](https://doc.global-sci.org/uploads/Issue/CSIAM-AM/v1n3/13_518.pdf). CSIAM Trans. Appl. Math., 1 (2020), pp. 518-529. doi: 10.4208/csiam-am.2020-0018.

The [original repository](https://github.com/AlbertZhangHIT/Hartley-spectral-pooling) for Hartley spectral pooling is deprecated.

# Usage
Some packages are required in `Python>=3.6`:

```
python>=3.6
torch>=0.4.1
torchvision>=0.2
numpy>=1.14
scipy>=1.0
torch-dct>=0.1
```
The [torch Discrete Cosine Transform](https://github.com/zh217/torch-dct) can be installed using `pip install torch-dct`. 


To reproduce the result on MNIST and Fashion-MNIST dataset, try
```
python mnist_simplenet.py --epochs 10 --pool hartley --run-num 10
python mnist_resnet.py --epochs 15 --pool harltey --run-num 5
python fashion_mnist_resnet.py --epochs 15 --pool hartley --run-num 5
```

To reproduce the result on Cifar10 dataset, try
```
python cifar_resnet.py --epochs 160 --pool hartley --run-num 5
```

# Citation

```
@Article{CSIAM-AM-1-518,
author = {Hao Zhang , and Jianwei Ma , },
title = {Hartley Spectral Pooling for Deep Learning},
journal = {CSIAM Transactions on Applied Mathematics},
year = {2020},
volume = {1},
number = {3},
pages = {518--529},
abstract = {<p style="text-align: justify;">In most convolution neural networks (CNNs), downsampling hidden layers is adopted for increasing computation efficiency and the receptive field size. Such
operation is commonly called pooling. Maximization and averaging over sliding windows ($max/average$ $pooling$), and plain downsampling in the form of strided convolution are popular pooling methods. Since the pooling is a lossy procedure, a motivation
of our work is to design a new pooling approach for less lossy in the dimensionality
reduction. Inspired by the spectral pooling proposed by Rippel et al. [1], we present
the Hartley transform based spectral pooling method. The proposed spectral pooling avoids the use of complex arithmetic for frequency representation, in comparison
with Fourier pooling. The new approach preserves more structure features for network&#39;s discriminability than max and average pooling. We empirically show the Hartley pooling gives rise to the convergence of training CNNs on MNIST and CIFAR-10
datasets.</p>},
issn = {2708-0579},
doi = {https://doi.org/10.4208/csiam-am.2020-0018},
url = {http://global-sci.org/intro/article_detail/csiam-am/18306.html}
}


```