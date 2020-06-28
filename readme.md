# Fourier Transform Image on CNN

> **Spoiler:** Don't try throw Fourier Transform Image on CNN

**Very short and naive experiment** to demonstrate usage of Fourier Transform Image on Convolutional Neural Network.

Some details :

* Pretrained EfficientNet B0
* CIFAR-10 Dataset
* Fourier Transform function from https://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html

To see converted Image, check `fourier_demo.ipynb`

## Tested on

* Hardware :
  * Ryzen 5 1600
  * 16GB RAM
  * Nvidia GTX 1060 6GB
* Software :
  * GNU/Linux distribution based on Debian 10 Buster
  * Python 3.8.3
  * Nvidia GPU Driver 440.82
  * CUDA 10.1
  * cuDNN 7.6.5
* Library :
  * Tensorflow 2.2.0
  * Tensorflow Addons 0.10.0
  * Efficientnet 1.1.0
  * Numpy 1.18.5
  * Matplotlib 3.2.1

## Benchmark Result

|                         | F1 Score               |
| ----------------------- | ---------------------- |
| Normal Image            | **0.8101679207753008** |
| Fourier Transform Image | 0.3786648010406338     |
| Hybrid                  | 0.8096627916201677     |
| Hybrid + MLP            | 0.8022311496589801     |
