# %% [markdown]
### Library

# %%
import platform
import os
import random

from sklearn.metrics import f1_score, classification_report
import efficientnet
import efficientnet.tfkeras as efn
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import sklearn
import matplotlib
import matplotlib.pyplot as plt

# %%
SEED = 42

os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# %%
print('Python version:', platform.python_version())
print('Tensorflow Version:', tf.__version__)
print('Tensorflow Addons Version:', tfa.__version__)
print('Efficientnet Version:', efficientnet.__version__)
print('Numpy Version:', np.__version__)
print('Matplotlib Version:', matplotlib.__version__)

# %% [markdown]
### Dataset

# %%
def fft(image):
    fourier = []

    for i in range(image.shape[2]):
        img = image[:, :, i]
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))
        fourier.append(magnitude_spectrum)

    fourier = (np.dstack(fourier)).astype(np.uint8)

    return fourier

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# %%
len_x_train = len(x_train)
len_x_test = len(x_test)
x_train_fft = x_train.copy()
x_test_fft = x_test.copy()

for i in range(len_x_train):
    x_train_fft[i] = fft(x_train[i])
for i in range(len_x_test):
    x_test_fft[i] = fft(x_test[i])

# %% [markdown]
### Check fourier transform image

# %%
def show_fft(image):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title('Fourier Transform image')
    plt.xticks([]), plt.yticks([])
    plt.show()

# %%
show_fft(x_train_fft[np.random.randint(0, len_x_train)])

# %%
show_fft(x_train_fft[np.random.randint(0, len_x_train)])

# %%
show_fft(x_train_fft[np.random.randint(0, len_x_train)])

# %% [markdown]
### Configure model

# %%
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dense, Concatenate

def get_efn0(name='efn0'):
    efn0 = efn.EfficientNetB0(
        input_shape=(32, 32, 3), include_top=False,
        weights='noisy-student', pooling='max'
    )
    for layer in efn0.layers:
        layer._name = f'{layer._name}_{name}'
    return efn0

# %%
model_image = get_efn0('image')
model_fft = get_efn0('fft')

concatenate = Concatenate(name='concatenate')([model_image.output, model_fft.output])

dense_a = Dense(512)(concatenate)
bn_a = BatchNormalization()(dense_a)
act_a = Activation('relu')(bn_a)

dense_b = Dense(128)(act_a)
bn_b = BatchNormalization()(dense_b)
act_b = Activation('relu')(bn_b)

output = Dense(10, activation='softmax', name='output')(act_b)

model = Model(inputs=[model_image.input, model_fft.input], outputs=output)
model.compile(
    optimizer=tfa.optimizers.RectifiedAdam(
        lr=0.005,
        total_steps=50,
        warmup_proportion=0.1,
        min_lr=0.0005,
    ),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'])

model.summary()

# %% [markdown]
### Train model

# %%
model.fit(
    [x_train, x_train_fft], y_train,
    batch_size=500, epochs=50, verbose=1
)

# %% [markdown]
### Test model

# %%
y_pred = model.predict([x_test, x_test_fft])
y_pred = np.argmax(y_pred, axis=-1)

# %%
f1 = f1_score(y_test, y_pred, average='weighted')

print('Weighted F1 Score:', f1)

# %%
print('Classification Report:')
print(classification_report(y_test, y_pred))
