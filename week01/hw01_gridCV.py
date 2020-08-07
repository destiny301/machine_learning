import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

pixels = 28*28
(xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()

xtr = xtr.reshape((60000, pixels)).astype(np.float32) / 255.0
xte = xte.reshape((10000, pixels)).astype(np.float32) / 255.0

def create_model(hidden_nodes, dropout_rate, reg_val):
    inputs = Input(shape = (pixels,), name = 'images')
    z = Dense(hidden_nodes, activation='relu', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val), name = 'hidden1')(inputs)
    z = Dropout(dropout_rate)(z)
    z = Dense(10, activation='softmax')(z)

    our_model = Model(inputs = inputs, outputs = z)
    our_model.summary()

    our_model.compile(optimizer = 'adam', loss = SparseCategoricalCrossentropy(), metrics = ['accuracy'])
    return our_model

dropout_rate = [0.2, 0.3, 0.4, 0.5]
hidden_nodes = [16, 32, 64]
reg_val = [1e-3, 1e-4]
epochs = [1, 10]

params = {'hidden_nodes':hidden_nodes, 'dropout_rate': dropout_rate, 'reg_val': reg_val, 'epochs': epochs}
model = KerasClassifier(build_fn=create_model, epochs = 10, batch_size = 32, verbose = 0)

gridCV = GridSearchCV(model, params, cv = 5, n_jobs = -1)
gridCV.fit(xtr, ytr)
print("best hyperparams:", gridCV.best_params_)