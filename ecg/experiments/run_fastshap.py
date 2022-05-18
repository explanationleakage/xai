import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, Conv1D

import numpy as np
import pandas as pd

from datetime import datetime
import os
import sys
import pickle
import time
import argparse

sys.path.insert(0, '/scratch/nj594/ecg_explain/fastshap_ecg')
from surrogate import Surrogate
from fastshap_ecg import FastSHAP

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
parser = argparse.ArgumentParser(description='REAL-X Mimic Experiment')
parser.add_argument('--arg_file', type=str, default='', metavar='a',
                    help='Path to File with Grid Search Arguments')
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
print(args.arg_file)
with open(args.arg_file, "rb") as arg_file:
    args.arg_file = pickle.load(arg_file)

arg_file = args.arg_file[args.index]
print(arg_file)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Set Model Dir
method = 'fastshap'
run = args.index
model_dir = os.path.join(method, str(run))
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

preds = np.load(os.path.join(data_dir, 'predictions.npy'), allow_pickle=True)
preds_discrete = np.eye(2)[preds.argmax(1)]

preds_val = np.load(os.path.join(data_dir, 'predictions_val.npy'), allow_pickle=True)
preds_discrete_val = np.eye(2)[preds_val.argmax(1)]


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Surrogate

superpixel_size = 8
surrogate_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'surrogate', 'surrogate.h5'))    
surrogate = Surrogate(surrogate_model = surrogate_model,
                           baseline = 0,
                           width = 1000, 
                           superpixel_size = superpixel_size)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# FastSHAP

### Specify Explainer Architecture

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
num_classes = 2

#Stanford Model
sys.path.insert(0, '/scratch/nj594/ecg/models/stanford')
import network

cnn = network.build_network(**params) 
base_model = Model(cnn.inputs, cnn.layers[67].output)

model_input = Input(shape=(1000,1))
net = base_model(model_input)
out = Conv1D(num_classes,1)(net)

explainer = Model(model_input, out)

### Extract Superpixel Size

superpixel_size = int(1000/explainer.output.shape[1])

### Train FastSHAP

fastshap = FastSHAP(explainer = explainer,
                    imputer = surrogate,
                    baseline = 0,
                    normalization=arg_file['normalization'],
                    link='identity')

t = time.time()
fastshap.train(train_data = X_train,
               val_data = X_val,
               batch_size = arg_file['batch_size'],
               num_samples = arg_file['num_samples'],
               max_epochs = arg_file['epochs'],
               validation_batch_size = arg_file['batch_size'],
               lr=arg_file['lr'],
               min_lr=1e-5,
               lr_factor=0.9,
               eff_lambda=arg_file['eff_lambda'],
               paired_sampling=arg_file['paired_sampling'],
               lookback=arg_file['lookback'],
               verbose=1, 
               model_dir = model_dir)
training_time = time.time() - t

with open(os.path.join(model_dir, 'training_time.pkl'), 'wb') as f:
    pickle.dump(training_time, f)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Explain w/ FastSHAP

### Explain

t = time.time()
shap_values = fastshap.explainer.predict(X_test)
explaining_time = time.time() - t
shap_values = [shap_values[:,:,i] for i in range(num_classes)]

shap_values_val = fastshap.explainer.predict(X_val)
shap_values_val = [shap_values_val[:,:,i] for i in range(num_classes)]

### Save

with open(os.path.join(model_dir, 'explaining_time.pkl'), 'wb') as f:
    pickle.dump(explaining_time, f)
    
with open(os.path.join(model_dir, 'shap_values.pkl'), 'wb') as f:
    pickle.dump(shap_values, f)
    
    
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
retro_ex_val = evaluate(X_val, shap_values_val, evaluator_model, y_val, y_val, 
                        mode = 'exclude', method = method)
retro_ex_test = evaluate(X_test, shap_values, evaluator_model, y_test, y_test, 
                         mode = 'exclude', method = method)

# Inclusion
retro_in_val = evaluate(X_val, shap_values_val, evaluator_model, y_val, y_val, 
                        mode = 'include', method = method)
retro_in_test = evaluate(X_test, shap_values, evaluator_model, y_test, y_test, 
                         mode = 'include', method = method)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#### Prospective Evaluation ####

# Exclusion
pro_ex_val = evaluate(X_val, shap_values_val, evaluator_model, preds_discrete_val, y_val, 
                        mode = 'exclude', method = method)
pro_ex_test = evaluate(X_test, shap_values, evaluator_model, preds_discrete, y_test, 
                         mode = 'exclude', method = method)

# Inclusion
pro_in_val = evaluate(X_val, shap_values_val, evaluator_model, preds_discrete_val, y_val, 
                        mode = 'include', method = method)
pro_in_test = evaluate(X_test, shap_values, evaluator_model, preds_discrete, y_test, 
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
header = ["model_dir", "lr", "epochs", "batch_size", "lookback", 
          "num_samples", "paired_sampling", "eff_lambda", "normalization", 
          "training_time", "explaining_time"]
metrics = ['AUC_acc','AUC_auroc','AUC_log_likelihood','AUC_log_odds']
for tag in tags:
    header += [x+'-'+tag for x in metrics]
    
results = {**results, **arg_file}
results['model_dir'] = model_dir
results["explaining_time"] = explaining_time
results["training_time"] = training_time
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