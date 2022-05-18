import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, Dense, GlobalAveragePooling1D, Conv1D, 
                                     AveragePooling1D, UpSampling1D, Flatten)

import numpy as np
import pandas as pd

from datetime import datetime
import os
import sys
import pickle
import time
import argparse
from tqdm import tqdm

sys.path.insert(0, '/scratch/nj594/xai/helpers')
from evaluate import evaluate

# IMPORTANT: SET RANDOM SEEDS FOR REPRODUCIBILITY
os.environ['PYTHONHASHSEED'] = str(420)
import random
random.seed(420)
np.random.seed(420)
tf.random.set_seed(420)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Command Line Arguements
parser = argparse.ArgumentParser(description='IntGrad Eye Experiment')
parser.add_argument('--index', type=int, default=9999, metavar='i',
                    help='Index for Job Array')
parser.add_argument('--verbose', type=int, default=1, metavar='v',
                    help='Prints Outputs')
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
num_samples = args.index
batch_size = 64

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Set Model Dir
method = 'integratedgradients'
run = str(num_samples)
model_dir = os.path.join(method, run)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Data

data_dir = os.path.join(os.getcwd(), 'data')

X_train = np.load(os.path.join(data_dir, 'X_train.npy'), allow_pickle=True)
X_val = np.load(os.path.join(data_dir, 'X_val.npy'), allow_pickle=True)
X_test = np.load(os.path.join(data_dir, 'X_test.npy'), allow_pickle=True)

y_train = np.load(os.path.join(data_dir, 'y_train.npy'), allow_pickle=True)
y_val = np.load(os.path.join(data_dir, 'y_val.npy'), allow_pickle=True)
y_test = np.load(os.path.join(data_dir, 'y_test.npy'), allow_pickle=True)

preds_train = np.load(os.path.join(data_dir, 'predictions_train.npy'), allow_pickle=True)
preds_discrete_train = np.eye(2)[preds_train.argmax(1)]
preds_val = np.load(os.path.join(data_dir, 'predictions_val.npy'), allow_pickle=True)
preds_discrete_val = np.eye(2)[preds_val.argmax(1)]
preds = np.load(os.path.join(data_dir, 'predictions.npy'), allow_pickle=True)
preds_discrete = np.eye(2)[preds.argmax(1)]


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Model

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

model = Model(model_input, out)
model.summary()

# Model Checkpointing
save_dir = 'model'
model_dir = os.path.join(os.getcwd(), save_dir)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
model_weights_path = os.path.join(model_dir, 'model_weights.h5')

# Get Checkpointed Model
print(model_weights_path)
model.load_weights(model_weights_path)
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Integrated Gradients

## Paramters
alphas = np.linspace(start=0.0,
                     stop=1.0,
                     num=num_samples, dtype=np.float64)
baseline = np.zeros_like(X_test[0])

#### TEST ####

## Calculate Gradients
t = time.time()

explanations = []
for y_class in range(y_test.shape[1]):
    exp = []
    ### Iterate Over Instances
    for i, current_input in enumerate(tqdm(X_test)):
        ### Reshape for Interpolations
        current_input = np.expand_dims(current_input, axis=0)
        current_alphas = tf.reshape(alphas, (num_samples,) + \
                                    (1,) * (len(current_input.shape) - 1))

        attribution_array = []
        ### Iterate over Batches of Alphas along Interpolated Path
        for j in range(0, num_samples, batch_size):
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
            print(batch_gradients.shape)
            batch_attributions = batch_gradients * batch_difference

            attribution_array.append(batch_attributions)

        attribution_array = np.concatenate(attribution_array, axis=0)
        exp.append(np.mean(attribution_array, axis=0))
    
    explanations.append(np.array(exp))

explaining_time = time.time() - t

with open(os.path.join(method, 'explanations.pkl'), 'wb') as f:
    pickle.dump(explanations, f)

with open(os.path.join(method, 'explaining_time.pkl'), 'wb') as f:
    pickle.dump(explaining_time, f)
    
    
#### VAL ####

## Calculate Gradients
explanations_val = []
for y_class in range(y_test.shape[1]):
    exp = []
    ### Iterate Over Instances
    for i, current_input in enumerate(tqdm(X_val)):
        ### Reshape for Interpolations
        current_input = np.expand_dims(current_input, axis=0)
        current_alphas = tf.reshape(alphas, (num_samples,) + \
                                    (1,) * (len(current_input.shape) - 1))

        attribution_array = []
        ### Iterate over Batches of Alphas along Interpolated Path
        for j in range(0, num_samples, batch_size):
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
            print(batch_gradients.shape)
            batch_attributions = batch_gradients * batch_difference

            attribution_array.append(batch_attributions)

        attribution_array = np.concatenate(attribution_array, axis=0)
        exp.append(np.mean(attribution_array, axis=0))
    
    explanations_val.append(np.array(exp))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Evaluator Model

eval_dir = os.path.join(os.getcwd(), 'evaluation', 'evaluator-data')
evaluator_model = tf.keras.models.load_model(os.path.join(eval_dir, 'surrogate.h5'))

OPTIMIZER = tf.keras.optimizers.Adam(1e-3)
METRICS = [ 
  tf.keras.metrics.AUC(name='auroc'),
  tf.keras.metrics.AUC(curve='PR', name='auprc'),
  tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='accuracy'),
]

evaluator_model.compile(
    loss='categorical_crossentropy',
    optimizer=OPTIMIZER,
    metrics=METRICS,
)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#### Retrospective Evaluation ####

# Exclusion
retro_ex_val = evaluate(X_val, explanations_val, evaluator_model, y_val, y_val, 
                        mode = 'exclude', method = method)
retro_ex_test = evaluate(X_test, explanations, evaluator_model, y_test, y_test, 
                         mode = 'exclude', method = method)

# Inclusion
retro_in_val = evaluate(X_val, explanations_val, evaluator_model, y_val, y_val, 
                        mode = 'include', method = method)
retro_in_test = evaluate(X_test, explanations, evaluator_model, y_test, y_test, 
                         mode = 'include', method = method)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#### Prospective Evaluation ####

# Exclusion
pro_ex_val = evaluate(X_val, explanations_val, evaluator_model, preds_discrete_val, y_val, 
                        mode = 'exclude', method = method)
pro_ex_test = evaluate(X_test, explanations, evaluator_model, preds_discrete, y_test, 
                         mode = 'exclude', method = method)

# Inclusion
pro_in_val = evaluate(X_val, explanations_val, evaluator_model, preds_discrete_val, y_val, 
                        mode = 'include', method = method)
pro_in_test = evaluate(X_test, explanations, evaluator_model, preds_discrete, y_test, 
                         mode = 'include', method = method)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Combine Results
tags = ['retro_ex_val','retro_ex_test','retro_in_val','retro_in_test', 
        'pro_ex_val','pro_ex_test','pro_in_val','pro_in_test']
result_list = [retro_ex_val,retro_ex_test,retro_in_val,retro_in_test,
               pro_ex_val,pro_ex_test,pro_in_val,pro_in_test]

results = {}
for res, tag  in zip(result_list, tags):
    res = {k+'-'+tag:v for k,v in res.items()}
    results = {**results, **res}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Save

### Create Results Dictionary
header = ["model_dir", "num_samples", "explaining_time"]
metrics = ['AUC_acc','AUC_auroc','AUC_log_likelihood','AUC_log_odds']
for tag in tags:
    header += [x+'-'+tag for x in metrics]
    
results['num_samples'] = num_samples
results['model_dir'] = model_dir
results["explaining_time"] = explaining_time
results = {k:v for k,v in results.items() if k in header}

### Convert to DataFrame
results_df = pd.DataFrame(results, index=[0])
results_df = results_df[header]

### Append DataFrame to csv
results_path = method+'/results.csv'
if os.path.exists(results_path):
    results_df.to_csv(results_path, mode='a',  header=False)
else:
    results_df.to_csv(results_path, mode='w',  header=True)