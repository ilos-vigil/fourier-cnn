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

# %% [markdown]
### Dataset

# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# %% [markdown]
### Configure model

# %%
efn0 = efn.EfficientNetB0(
    input_shape=(32, 32, 3), include_top=False,
    weights='noisy-student', pooling='max',
)
model = tf.keras.Sequential([
    efn0,
    Dense(10, activation='softmax')
])
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
    x_train, y_train,
    batch_size=500, epochs=50, verbose=1
)

# %% [markdown]
### Test model

# %%
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=-1)

# %%
f1 = f1_score(y_test, y_pred, average='weighted')

print('Weighted F1 Score:', f1)

# %%
print('Classification Report:')
print(classification_report(y_test, y_pred))
