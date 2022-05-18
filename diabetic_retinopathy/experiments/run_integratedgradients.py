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
batch_size = 32

INPUT_SHAPE = (544, 544, 3)
NUM_CLASSES = 5

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Set Model Dir
method = 'integratedgradients'
run = str(num_samples)
exp_dir = os.path.join(method, run)
if not os.path.isdir(exp_dir):
    os.makedirs(exp_dir)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Data

data_dir = os.path.join(os.getcwd(), 'data')

X_val = np.load(os.path.join(data_dir, 'X_val_processed.npy'), allow_pickle=True)
X_test = np.load(os.path.join(data_dir, 'X_test_processed.npy'), allow_pickle=True)

y_val = np.load(os.path.join(data_dir, 'y_val.npy'), allow_pickle=True)
y_test = np.load(os.path.join(data_dir, 'y_test.npy'), allow_pickle=True)

preds = np.load(os.path.join(data_dir, 'predictions.npy'), allow_pickle=True)
preds_discrete = np.eye(5)[preds.argmax(1)]

preds_val = np.load(os.path.join(data_dir, 'predictions_val.npy'), allow_pickle=True)
preds_discrete_val = np.eye(5)[preds_val.argmax(1)]

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

# Integrated Gradients

if os.path.isfile(os.path.join(exp_dir, 'explanations.pkl')):
    print('Loading Test Explanations')
    #Load Explanations + Time
    with open(os.path.join(exp_dir, 'explanations.pkl'), 'rb') as f:
        explanations = pickle.load(f)
    
    with open(os.path.join(exp_dir, 'explaining_time.pkl'), 'rb') as f:
        explaining_time = pickle.load(f)
        
else:
    ## Paramters
    alphas = np.linspace(start=0.0,
                         stop=1.0,
                         num=num_samples, dtype=np.float32)
    baseline = np.zeros_like(X_test[0])

    #### TEST ####

    ## Calculate Gradients
    t = time.time()

    explanations = []
    for y_class in range(y_test.shape[1]):
        exp = []
        ### Iterate Over Instances
        for current_input in tqdm(X_test):
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
                batch_attributions = batch_gradients * batch_difference

                attribution_array.append(batch_attributions)

            attribution_array = np.concatenate(attribution_array, axis=0)
            exp.append(np.mean(attribution_array, axis=0))

        exp = np.array(exp)
        exp = np.sum(np.abs(exp), axis=-1) # Aggregate Accross Channels 
        exp = np.expand_dims(exp, -1)
        
        exp = np.abs(exp)
        exp = AveragePooling2D(pool_size=17)(exp) * 17 #Pooling Accross Super-pixel
        exp = UpSampling2D(size=17)(exp).numpy()
    
        explanations.append(np.array(exp))

    explaining_time = time.time() - t
    

    with open(os.path.join(exp_dir, 'explanations.pkl'), 'wb') as f:
        pickle.dump(explanations, f)

    with open(os.path.join(exp_dir, 'explaining_time.pkl'), 'wb') as f:
        pickle.dump(explaining_time, f)

    ## Clean
    gc.collect()
    del (attribution_array, batch_attributions, batch_gradients, 
         batch_predictions, reps, batch_input, batch_baseline, 
         batch_difference, batch_interpolated, exp)
    gc.collect()

#### VAL ####

if os.path.isfile(os.path.join(exp_dir, 'explanations_val.pkl')):
    print('Loading Val Explanations')
    #Load Explanations + Time
    with open(os.path.join(exp_dir, 'explanations_val.pkl'), 'rb') as f:
        explanations_val = pickle.load(f)
    
    with open(os.path.join(exp_dir, 'explaining_time_val.pkl'), 'rb') as f:
        explaining_time_val = pickle.load(f)
        
else:
    ## Calculate Gradients
    t = time.time()

    explanations_val = []
    for y_class in range(y_test.shape[1]):
        exp = []
        ### Iterate Over Instances
        for current_input in tqdm(X_val):
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
                batch_attributions = batch_gradients * batch_difference

                attribution_array.append(batch_attributions)

            attribution_array = np.concatenate(attribution_array, axis=0)
            exp.append(np.mean(attribution_array, axis=0))

        exp = np.array(exp)
        exp = np.sum(np.abs(exp), axis=-1) # Aggregate Accross Channels 
        exp = np.expand_dims(exp, -1)
        
        exp = np.abs(exp)
        exp = AveragePooling2D(pool_size=17)(exp) * 17 #Pooling Accross Super-pixel
        exp = UpSampling2D(size=17)(exp).numpy()
        
        explanations_val.append(np.array(exp))

    explaining_time_val = time.time() - t
        
    with open(os.path.join(exp_dir, 'explanations_val.pkl'), 'wb') as f:
        pickle.dump(explanations_val, f)
    
    with open(os.path.join(exp_dir, 'explaining_time_val.pkl'), 'wb') as f:
        pickle.dump(explaining_time_val, f)

    ## Clean
    gc.collect()
    del (attribution_array, batch_attributions, batch_gradients, 
         batch_predictions, reps, batch_input, batch_baseline, 
         batch_difference, batch_interpolated, exp)
    gc.collect()
    
## Clean
K.clear_session()
del model
K.clear_session()

##############################################################

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
results['model_dir'] = exp_dir
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