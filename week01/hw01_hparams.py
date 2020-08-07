import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorboard.plugins.hparams import api as hp

pixels = 28*28
(xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
xtr = xtr.reshape((60000, pixels)).astype(np.float32) / 255.0
xte = xte.reshape((10000, pixels)).astype(np.float32) / 255.0

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32, 64]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.3))

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT],
        metrics=[hp.Metric('accuracy', display_name='Accuracy')],
    )

# hidden_nodes = 64
# dropout = 0.3

def fit_model(hparams):
    inputs = Input(shape = (pixels,), name = 'images')
    z = Dense(hparams[HP_NUM_UNITS], activation='relu', name = 'hidden1')(inputs)
    z = Dropout(hparams[HP_DROPOUT])(z)
    z = Dense(10, activation='softmax')(z)

    our_model = Model(inputs = inputs, outputs = z)
    # our_model.summary()

    our_model.compile(optimizer = 'adam', loss = SparseCategoricalCrossentropy(), metrics = ['accuracy'])
    results = our_model.fit(xtr, ytr, batch_size = 32, epochs = 1, validation_split = 0.2)

    _, acc = our_model.evaluate(xte, yte)
    return acc

def run(log_dir, hparams):
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams(hparams)
        acc = fit_model(hparams)
        tf.summary.scalar('accuracy', acc, step = 1)

num = 0

for num_units in HP_NUM_UNITS.domain.values:
    for dropout in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        hparams = {
            HP_NUM_UNITS: num_units,
            HP_DROPOUT: dropout,
        }

        run_name = "run-%d" % num
        print("----start trial: %s" % run_name)
        print({h.name:hparams[h] for h in hparams})
        run('logs/hparam_tuning/'+run_name, hparams)

        num+=1


