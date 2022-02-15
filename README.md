![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
# Super-Fast-Adversarial-Training
[![Generic badge](https://img.shields.io/badge/Library-Pytorch-green.svg)](https://pytorch.org/)
[![Generic badge](https://img.shields.io/badge/Version-alpha-red.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Under-Develop-blue.svg)](https://shields.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ByungKwanLee/Super-Fast-Adversarial-Training/blob/master/LICENSE)
---

This is a PyTorch Implementation code for developing super fast adversarial training.
This code is combined with below state-of-the-art technologies for
accelarating adversarial attacks and defenses with Deep Neural Networks
on Volta GPU architecture.

- [x] Distributed Data Parallel [[link]](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [x] Channel Last Memory Format [[link]](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#:~:text=Channels%20last%20memory%20format%20is,pixel%2Dper%2Dpixel)
- [x] Mixed Precision Training [[link]](https://openreview.net/forum?id=r1gs9JgRZ)
- [x] Mixed Precision + Adversarial Attack (based on torchattacks [[link]](https://github.com/Harry24k/adversarial-attacks-pytorch))
- [x] Faster Adversarial Training for Large Dataset [[link]](https://openreview.net/forum?id=BJx040EFvH)
- [x] Fast Forward Computer Vision (FFCV) [[link]](https://github.com/libffcv/ffcv)

---



## Citation
If you find this work helpful, please cite it as:

```
@software{SuperFastAT_ByungKwanLee_2022,
  author = {Byung-Kwan Lee},
  title = {Super-Fast-Adversarial-Training},
  url = {https://github.com/ByungKwanLee/Super-Fast-Adversarial-Training},
  version = {alpha},
  year = {2022}
}
```
---

## Library for Fast Adversarial Attacks
This library is developed based on the well-known package of torchattacks [[link]](https://github.com/Harry24k/adversarial-attacks-pytorch) due to its simple scalability.

**Under Developement (Current Available Attacks Below)**

* Fast Gradient Sign Method ([FGSM](https://arxiv.org/abs/1412.6572))
* Projected Gradient Descent ([PGD](https://arxiv.org/abs/1706.06083))

[//]: # (* Basic Iterative Method &#40;[BIM]&#40;https://arxiv.org/abs/1611.01236&#41;&#41;)
[//]: # (* Carlini & Wagner &#40;[CW]&#40;https://arxiv.org/abs/1608.04644&#41;&#41;)
[//]: # (* AutoAttack &#40;[AA]&#40;https://arxiv.org/abs/2003.01690&#41;&#41;)
[//]: # (* Fast Adaptive Boundary &#40;[FAB]&#40;http://proceedings.mlr.press/v119/croce20a.html&#41;&#41;)

---

## Available Datasets
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
* [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [SVHN](http://ufldl.stanford.edu/housenumbers/)
* [Tiny-ImageNet](https://www.kaggle.com/c/tiny-imagenet/overview)
* [ImageNet](https://www.image-net.org/)

---

## Available Baseline Models

* [VGG](https://arxiv.org/abs/1409.1556) *(model/vgg.py)*
* [ResNet](https://arxiv.org/abs/1512.03385) *(model/resnet.py)*
* [WideResNet](https://arxiv.org/abs/1605.07146) *(model/wide.py)*
* [DenseNet](https://arxiv.org/abs/1608.06993) *(model/dense.py)*
---

# How to run

---

* First, run `fast_dataset_converter.py` to generate dataset with `.betson` extension, instead of using original dataset [[FFCV]](https://github.com/libffcv/ffcv).

```python
# Future import build
from __future__ import print_function

# Import built-in module
import os
import argparse

# fetch args
parser = argparse.ArgumentParser()

# parameter
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--gpu', default='0', type=str)
args = parser.parse_args()

# GPU configurations
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

# init fast dataloader
from utils.fast_data_utils import save_data_for_beton
save_data_for_beton(dataset=args.dataset)
```

---
* Second, run `fast_pretrain_standard.py`(Standard Training) or `fast_pretrain_adv.py` (Adversarial Training)

```python
# model parameter
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--network', default='resnet', type=str)
parser.add_argument('--depth', default=50, type=int)
parser.add_argument('--gpu', default='0,1,2,3,4', type=str)

# learning parameter
parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.0002, type=float)
parser.add_argument('--batch_size', default=512, type=float)
parser.add_argument('--test_batch_size', default=128, type=float)
parser.add_argument('--epoch', default=100, type=int)
```

or

```python
# model parameter
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--network', default='resnet', type=str)
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--gpu', default='0,1,2,3,4', type=str)

# learning parameter
parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--weight_decay', default=0.0002, type=float)
parser.add_argument('--batch_size', default=1024, type=float)
parser.add_argument('--test_batch_size', default=512, type=float)
parser.add_argument('--epoch', default=60, type=int)

# attack parameter
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--eps', default=0.03, type=float)
parser.add_argument('--steps', default=10, type=int)
```
