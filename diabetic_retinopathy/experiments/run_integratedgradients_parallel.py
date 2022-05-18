import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, Layer, Dense, AveragePooling2D, UpSampling2D)

import tensorflow_datasets as tfds

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121

import numpy as np
import pandas as pd

from datetime import datetime
import os
import sys
import pickle
import time
import argparse
from tqdm import tqdm
import gc

# IMPORTANT: SET RANDOM SEEDS FOR REPRODUCIBILITY
os.environ['PYTHONHASHSEED'] = str(420)
import random
random.seed(420)
np.random.seed(420)
tf.random.set_seed(420)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Command Line Arguements
parser = argparse.ArgumentParser(description='Kernal SHAP Eye Explainer')
parser.add_argument('--index', type=int, default=9999, metavar='i',
                    help='Index for Job Array')
args = parser.parse_args()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Get Index (Either from argument or from SLURM JOB ARRAY)
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    args.index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    print('SLURM_ARRAY_TASK_ID found..., using index %s' % args.index)
else:
    print('no SLURM_ARRAY_TASK_ID... using index %s' % args.index)
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Get Arguments
batch_size = 32
index = args.index

INPUT_SHAPE = (544, 544, 3)
NUM_CLASSES = 5


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Model
from tensorflow.keras.applications.densenet import DenseNet121

base_model = DenseNet121(
    include_top=False, weights='imagenet', 
    input_shape=INPUT_SHAPE, pooling='avg'
)

base_model.trainable = True
base_model.summary()

model_input = Input(shape=INPUT_SHAPE, name='input')

net = base_model(model_input)
out = Dense(NUM_CLASSES, activation='softmax')(net)

model = Model(model_input, out)

model.load_weights('model/densenet/model_weights.h5')
model.trainable = False

OPTIMIZER = tf.keras.optimizers.Adam(1e-3)
METRICS = [ 
  tf.keras.metrics.AUC(name='auroc'),
  tf.keras.metrics.AUC(curve='PR', name='auprc'),
  tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='accuracy'),
]

model.compile(
    loss='categorical_crossentropy',
    optimizer=OPTIMIZER,
    metrics=METRICS,
)
##############################################################

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Get Arguments
for split in ['test', 'val']:
    
    # Continue on if Out of Range
    if split == 'val' and index > 999:
        print('Index out of Range')
        continue
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Load and Select Images

    data_dir = os.path.join(os.getcwd(), 'data')
    images = np.load(os.path.join(data_dir, 'X_{}_processed.npy'.format(split)), allow_pickle=True)
    labels = np.load(os.path.join(data_dir, 'y_{}.npy'.format(split)), allow_pickle=True)
    if split == 'test':
        preds = np.load(os.path.join(data_dir, 'predictions.npy'), allow_pickle=True)
    else:
        preds = np.load(os.path.join(data_dir, 'predictions_val.npy'), allow_pickle=True)

    background = None
        
    IMAGE = images[index]

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # Explain IMAGE
    
    for num_samples in [2**6, 2**7, 2**8, 2**9, 2**10]:

        #Set Model Dir
        method = 'integratedgradients'
        exp_dir = os.path.join(method, split, str(num_samples), str(index))
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        print(exp_dir)
            
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        ### Explain with IntGrad
        
        ## Paramters
        alphas = np.linspace(start=0.0,
                             stop=1.0,
                             num=num_samples, dtype=np.float32)
        baseline = np.zeros_like(IMAGE)

        ## Calculate Gradients
        t = time.time()

        explanations = []
        for y_class in range(labels.shape[1]):
            ### Reshape for Interpolations
            current_input = np.expand_dims(IMAGE, axis=0)
            current_alphas = tf.reshape(alphas, (num_samples,) + \
                                        (1,) * (len(current_input.shape) - 1))

            attribution_array = []
            ### Iterate over Batches of Alphas along Interpolated Path
            for j in tqdm(range(0, num_samples, batch_size)):
                number_to_draw = min(batch_size, num_samples - j)

                batch_alphas = current_alphas[j:min(j + batch_size, num_samples)]

                reps = np.ones(len(current_input.shape)).astype(int)
                reps[0] = number_to_draw
                batch_input = tf.convert_to_tensor(np.tile(current_input, reps))

                batch_baseline = tf.convert_to_tensor(np.tile(baseline, reps))

                batch_difference = batch_input - batch_baseline
                batch_interpolated = batch_alphas * batch_input + \
                                     (1.0 - batch_alphas) * batch_baseline

                with tf.GradientTape() as tape:
                    tape.watch(batch_interpolated)

                    batch_predictions = model(batch_interpolated)
                    ### Get prediction of the predicted class
                    batch_predictions = batch_predictions[:, y_class]

                batch_gradients = tape.gradient(batch_predictions, batch_interpolated)
                batch_attributions = batch_gradients * batch_difference

                attribution_array.append(batch_attributions)

            attribution_array = np.concatenate(attribution_array, axis=0)
            exp = np.mean(attribution_array, axis=0)

            exp = np.sum(np.abs(exp), axis=-1) # Aggregate Accross Channels 
            exp = np.expand_dims(exp, -1)

            exp = np.abs(exp)
            explanations.append(exp)

        explaining_time = time.time() - t

        with open(os.path.join(exp_dir, 'explanations.pkl'), 'wb') as f:
            pickle.dump(explanations, f)

        with open(os.path.join(exp_dir, 'explaining_time.pkl'), 'wb') as f:
            pickle.dump(explaining_time, f)