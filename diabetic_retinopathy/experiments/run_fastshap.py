import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (Input, Layer, Dense, Conv2D,
                                     experimental, Concatenate, 
                                     GlobalAveragePooling2D, Lambda, Resizing)

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
import gc

sys.path.insert(0, '/vast/nj594/xai/helpers')
from evaluate import evaluate

from fastshap import ImageSurrogate as Surrogate
from fastshap import FastSHAP

# IMPORTANT: SET RANDOM SEEDS FOR REPRODUCIBILITY
os.environ['PYTHONHASHSEED'] = str(420)
import random
random.seed(420)
np.random.seed(420)
tf.random.set_seed(420)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Command Line Arguements
parser = argparse.ArgumentParser(description='FastSHAP Eye Experiment')
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
arg_file = 'fastshap/arg_file.pkl'
with open(arg_file, "rb") as arg_file:
    arg_file = pickle.load(arg_file)

arg_file = arg_file[args.index]

INPUT_SHAPE = (544, 544, 3)
NUM_CLASSES = 5

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Set Model Dir
method = 'fastshap'
run = str(args.index)
model_dir = os.path.join(method, run)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Data

(ds_train, ds_val, ds_test), ds_info = tfds.load(
    'diabetic_retinopathy_detection/btgraham-300',
    split=['train', 'validation', 'test'],
    as_supervised=False,
    with_info=True
)

def batch_data(dataset, fn, batch_size=32):
    dataset = dataset.map(fn)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

def reformat(input_dict):
    
    image = tf.cast(input_dict['image'], tf.float32)
    image = Resizing(INPUT_SHAPE[0], INPUT_SHAPE[1], interpolation='nearest')(image)
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    label = tf.one_hot(input_dict['label'], depth = NUM_CLASSES)

    return (image, label)

   
ds_train = batch_data(ds_train, reformat, arg_file['batch_size'])
ds_val = batch_data(ds_val, reformat, arg_file['batch_size'])
ds_test = batch_data(ds_test, reformat, arg_file['batch_size'])

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

# Check if Shapely Values Already Generated
if os.path.isfile(os.path.join(model_dir, 'shap_values.pkl')):
    print('Loading Explanations')
    #Load Shapley Values 
    with open(os.path.join(model_dir, 'shap_values.pkl'), 'rb') as f:
        shap_values = pickle.load(f)
    with open(os.path.join(model_dir, 'shap_values_val.pkl'), 'rb') as f:
        shap_values_val = pickle.load(f)
        
    #Load Times
    with open(os.path.join(model_dir, 'training_time.pkl'), 'rb') as f:
        training_time = pickle.load(f)
    with open(os.path.join(model_dir, 'explaining_time.pkl'), 'rb') as f:
        explaining_time = pickle.load(f)

    #Load Loss
    with open(os.path.join(model_dir, 'loss.pkl'), 'rb') as f:
        loss = pickle.load(f)
        
else:
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Load Surrogate

    superpixel_size = 32
    surrogate_model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'surrogate-data', 'surrogate.h5'))   
    surrogate_model._name = 'surrogate'
    surrogate = Surrogate(surrogate_model = surrogate_model,
                          baseline = 0,
                          width = INPUT_SHAPE[1], 
                          height = INPUT_SHAPE[0], 
                          superpixel_size = superpixel_size)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # FastSHAP

    ### Specify Explainer Architecture

    model_input = Input(shape=INPUT_SHAPE, name='input')

    base_model = DenseNet121(
        include_top=False, weights='imagenet', 
        input_shape=INPUT_SHAPE
    )

    base_model.trainable = True

    net = base_model(model_input)

    # Learn Phi 
    phi = Conv2D(NUM_CLASSES, 1)(net)

    explainer = Model(model_input, phi)
    explainer._name = 'explainer'
    explainer.summary()


    ### Train FastSHAP
    print('Training FastSHAP')
    fastshap = FastSHAP(explainer = explainer,
                        imputer = surrogate,
                        baseline = 0,
                        normalization=arg_file['normalization'],
                        link='logit')

    t = time.time()
    fastshap.train(train_data = ds_train,
                   val_data = ds_val,
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

    ## Clear Memory

    gc.collect()
    del surrogate
    del surrogate_model
    del ds_train
    del ds_val
    del ds_test
    gc.collect()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #### TEST ####
    ### Explain w/ FastSHAP
    print('Explaining Test')

    t = time.time()
    shap_values = np.vstack([fastshap.explainer(X_test[i: i+10]).numpy() for i in range(0, 2500, 10)])
    explaining_time = time.time() - t
    shap_values = [shap_values[:,:,:,i] for i in range(NUM_CLASSES)]

    ### Save

    with open(os.path.join(model_dir, 'explaining_time.pkl'), 'wb') as f:
        pickle.dump(explaining_time, f)

    with open(os.path.join(model_dir, 'shap_values.pkl'), 'wb') as f:
        pickle.dump(shap_values, f)

    ## Clear Memory
    gc.collect()

    #### VAL ####

    ### Explain w/ FastSHAP
    print('Explaining Val')

    shap_values_val = np.vstack([fastshap.explainer(X_val[i: i+10]).numpy() for i in range(0, 1000, 10)])
    shap_values_val = [shap_values_val[:,:,:,i] for i in range(NUM_CLASSES)]

    with open(os.path.join(model_dir, 'shap_values_val.pkl'), 'wb') as f:
        pickle.dump(shap_values_val, f)

    loss = np.min(fastshap.val_losses)
    with open(os.path.join(model_dir, 'loss.pkl'), 'wb') as f:
        pickle.dump(loss, f)

    del fastshap
    del explainer
    K.clear_session()
    gc.collect()

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