import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

pixels = 28*28
hidden_nodes = 64
dropout = 0.3

(xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()

xtr = xtr.reshape((60000, pixels)).astype(np.float32) / 255.0
xte = xte.reshape((10000, pixels)).astype(np.float32) / 255.0

inputs = Input(shape = (pixels,), name = 'images')
z = Dense(hidden_nodes, activation='relu', name = 'hidden1')(inputs)
z = Dropout(dropout)(z)
z = Dense(10, activation='softmax')(z)

our_model = Model(inputs = inputs, outputs = z)
our_model.summary()

our_model.compile(optimizer = 'adam', loss = SparseCategoricalCrossentropy(), metrics = ['accuracy'])
results = our_model.fit(xtr, ytr, batch_size = 32, epochs = 10, validation_split = 0.2)
our_model.save('hw01_model.hdf5')

