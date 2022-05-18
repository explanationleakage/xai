import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, Layer, Dense, AveragePooling2D, UpSampling2D)

import tensorflow_datasets as tfds

from tensorflow.keras.applications.inception_v3 import InceptionV3

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
parser = argparse.ArgumentParser(description='KS Eye Compile')
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Set Model Dir
method = 'integratedgradients'
test_dir = os.path.join(method, 'test', str(num_samples))
val_dir = os.path.join(method, 'val', str(num_samples))

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

#Compile Explanations

#### TEST ####
explanations = []
explaining_times = []
for i in tqdm(range(2500)):
    run_dir = os.path.join(test_dir, str(i))

    try:
        with open(os.path.join(run_dir, 'explanations.pkl'), 'rb') as f:
            explanation = pickle.load(f)

        with open(os.path.join(run_dir, 'explaining_time.pkl'), 'rb') as f:
            explaining_time = pickle.load(f)

        explanations.append(np.array(explanation, dtype="float32").squeeze())
        explaining_times.append(explaining_time)

    except:
        print('missing:', i)

explanations = np.stack(explanations, 0)
explanations = [explanations[:,i,:,:] for i in range(5)]
explaining_total_time = np.array(explaining_times).sum()/3600

# Save
with open(os.path.join(test_dir, 'explanations.pkl'), 'wb') as f:
    pickle.dump(explanations, f)

with open(os.path.join(test_dir, 'explaining_time.pkl'), 'wb') as f:
    pickle.dump(explaining_total_time, f)

#### VAL ####
explanations_val = []
explaining_times = []
for i in tqdm(range(1000)):
    run_dir = os.path.join(val_dir, str(i))

    try:
        with open(os.path.join(run_dir, 'explanations.pkl'), 'rb') as f:
            explanation = pickle.load(f)

        with open(os.path.join(run_dir, 'explaining_time.pkl'), 'rb') as f:
            explaining_time = pickle.load(f)

        explanations_val.append(np.array(explanation, dtype="float32").squeeze())
        explaining_times.append(explaining_time)

    except:
        print('missing:', i)

explanations_val = np.stack(explanations_val, 0)
explanations_val = [explanations_val[:,i,:,:] for i in range(5)]
explaining_total_time_val = np.array(explaining_times).sum()/3600

# Save
with open(os.path.join(val_dir, 'explanations.pkl'), 'wb') as f:
    pickle.dump(explanations_val, f)

with open(os.path.join(val_dir, 'explaining_time.pkl'), 'wb') as f:
    pickle.dump(explaining_total_time_val, f)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Evaluator Model
print('Loading Evaluator')

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

gc.collect()

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
header = ["model_dir", "num_samples", "explaining_time_val", "explaining_time"]
metrics = ['AUC_acc','AUC_auroc','AUC_log_likelihood','AUC_log_odds']
for tag in tags:
    header += [x+'-'+tag for x in metrics]
    
results['num_samples'] = num_samples
results['model_dir'] = test_dir
results["explaining_time"] = explaining_total_time
results["explaining_time_val"] = explaining_total_time_val
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