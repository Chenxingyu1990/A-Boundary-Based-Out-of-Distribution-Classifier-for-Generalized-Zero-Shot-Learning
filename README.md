# A Boundary Based Out-of-Distribution Classifier for Generalized Zero-Shot Learning
### A Pytorch implementation 

## Overview
This library contains a Pytorch implementation of 'A Boundary Based Out-of-Distribution Classifier for Generalized Zero-Shot Learning' [1] (#citation)(https://arxiv.org/abs/2008.04872). This work is built on top of S-VAE, as presented in [[2]](#citation)(http://arxiv.org/abs/1804.00891). 


## Dependencies

* **python>=3.6**
* **pytorch>=0.4.1**: https://pytorch.org
* **scipy**: https://scipy.org
* **numpy**: https://www.numpy.org

## Installation

1. Add this project to you python path.

2. To install, run

```bash
$ python setup.py install
```

3. Training, run

```bash
$ cd src
$ python train.py ../config/awa1.yaml
```
4. Testing, run

```bash
$ python test.py ../config/awa1.yaml
```

## Usage

Please cite [[1](#citation)] in your work when using this library in your experiments.



## Citation
```
[1] Xingyu Chen, Xuguang Lan, Fuchun Sun, Nanning Zheng. A Boundary Based Out-Of-Distribution
Classifier for Generalized Zero-Shot Learning[C]//Proceedings of the European Conference on
Computer Vision (ECCV) 2020.
```
