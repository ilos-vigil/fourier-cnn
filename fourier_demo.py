# %% [markdown]
### Library

# %%
import platform

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# %%
print('Python version:', platform.python_version())
print('Tensorflow Version:', tf.__version__)
print('Matplotlib Version:', matplotlib.__version__)

# %% [markdown]
### Dataset

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# %% [markdown]
### Fourier Transform
# Source : https://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html

# %%
def fft(image):
    cmap = [plt.cm.Reds_r, plt.cm.Greens_r, plt.cm.Blues_r]
    color = ['Red', 'Green', 'Blue']

    plt.figure(figsize=(6*2, 6))

    plt.subplot(2, 4, 1)
    plt.imshow(image)
    plt.title('Original image')
    plt.xticks([]), plt.yticks([])

    fourier = []
    for i in range(image.shape[2]):
        img = image[:, :, i]
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        fourier.append(magnitude_spectrum)

        plt.subplot(2, 4, i + 2)
        plt.imshow(img, cmap=cmap[i])
        plt.title(f'{color[i]} channel')
        plt.xticks([]), plt.yticks([])

        plt.subplot(2, 4, 6 + i)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title(f'Fourier {color[i]} channel')
        plt.xticks([]), plt.yticks([])

    fourier = (np.dstack(fourier)).astype(np.uint8)
    plt.subplot(2, 4, 5)
    plt.imshow(fourier, cmap='gray')
    plt.title('Fourier (combined)')
    plt.xticks([]), plt.yticks([])

    plt.show()

# %%
fft(x_train[np.random.randint(0, 50000)])

# %%
fft(x_train[np.random.randint(0, 50000)])

# %%
fft(x_train[np.random.randint(0, 50000)])
