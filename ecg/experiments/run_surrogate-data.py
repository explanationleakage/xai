import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D

import numpy as np

from datetime import datetime
import os
import sys
import pickle
import time

sys.path.insert(0, '/scratch/nj594/ecg_explain/fastshap_ecg')
from surrogate import Surrogate

# IMPORTANT: SET RANDOM SEEDS FOR REPRODUCIBILITY
os.environ['PYTHONHASHSEED'] = str(420)
import random
random.seed(420)
np.random.seed(420)
tf.random.set_seed(420)

from importlib import reload 
import surrogate
reload(surrogate)
from surrogate import Surrogate

## Load Data

data_dir = os.path.join(os.getcwd(), 'data')

X_train = np.load(os.path.join(data_dir, 'X_train.npy'), allow_pickle=True)
X_val = np.load(os.path.join(data_dir, 'X_val.npy'), allow_pickle=True)
X_test = np.load(os.path.join(data_dir, 'X_test.npy'), allow_pickle=True)

y_train = np.load(os.path.join(data_dir, 'y_train.npy'), allow_pickle=True)
y_val = np.load(os.path.join(data_dir, 'y_val.npy'), allow_pickle=True)
y_test = np.load(os.path.join(data_dir, 'y_test.npy'), allow_pickle=True)

# Train Surrogate Directly on Data

### Save Dir

save_dir = 'surrogate-data'
model_dir = os.path.join(os.getcwd(), save_dir)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

### Surrogate Model

params = {
    #NN Hyperparameters
    "input_shape": [1000, 1],
    "num_categories": 2,
    "conv_subsample_lengths": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    "conv_filter_length": 8,
    "conv_num_filters_start": 32,
    "conv_init": "he_normal",
    "conv_activation": "relu",
    "conv_dropout": 0.2,
    "conv_num_skip": 2,
    "conv_increase_channels_at": 4,
    "compile": False,
    "is_regular_conv": False,
    "is_by_time": False,
    "is_by_lead": False,
    "ecg_out_size": 64,
    "nn_layer_sizes" : None,
    "is_multiply_layer": False, 
}

#Stanford Model
sys.path.insert(0, '/scratch/nj594/ecg/models/stanford')
import network

model_input = Input(shape=(1000,1))

cnn = network.build_network(**params) 
cnn = Model(cnn.inputs, cnn.layers[-4].output)

net = cnn(model_input)
net = GlobalAveragePooling1D()(net)
out = Dense(2, activation='softmax')(net)

surrogate_model = Model(model_input, out)

### Train

superpixel_size  = 8
# if os.path.isfile(os.path.join(model_dir, 'surrogate.h5')):
#     print('Loading saved model')
#     surrogate_model = tf.keras.models.load_model(os.path.join(model_dir, 'surrogate.h5'))
    
#     surrogate = Surrogate(surrogate_model = surrogate_model,
#                                baseline = 0,
#                                width = 1000, 
#                                superpixel_size = superpixel_size)
# else:    
surrogate = Surrogate(surrogate_model = surrogate_model,
                      baseline = 0,
                      width = 1000, 
                      superpixel_size = superpixel_size)

t = time.time()
surrogate.train(original_model = None,
                train_data = (X_train, y_train),
                val_data = (X_val, y_val),
                batch_size = 32,
                max_epochs = 100,
                validation_batch_size = 32,
                loss_fn='categorical_crossentropy',
                lr=1e-3,
                min_lr=1e-5,
                lr_factor=0.9,
                lookback=10,
                gpu_device=0,
                verbose=1, 
                model_dir = model_dir)
training_time = time.time() - t

with open(os.path.join(model_dir, 'training_time.pkl'), 'wb') as f:
    pickle.dump(training_time, f)

METRICS = [ 
  tf.keras.metrics.BinaryAccuracy(name='accuracy'),
]
OPTIMIZER = tf.keras.optimizers.Adam(1e-3)

surrogate.model.compile(
    loss='categorical_crossentropy',
    optimizer=OPTIMIZER,
    metrics=METRICS,
)
surrogate.model.evaluate(x=X_test, y=y_test)