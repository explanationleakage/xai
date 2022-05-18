import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense

from transformers import TFBertForSequenceClassification

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
from evaluate import evaluate_mimic as evaluate

# IMPORTANT: SET RANDOM SEEDS FOR REPRODUCIBILITY
os.environ['PYTHONHASHSEED'] = str(420)
import random
random.seed(420)
np.random.seed(420)
tf.random.set_seed(420)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Command Line Arguements
parser = argparse.ArgumentParser(description='SHAP MIMIC Compile')
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
method = 'kernelshap_s'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Load Data 

data_dir = './data'
label_list = [0, 1]
max_seq_length = 128
num_classes = len(label_list)

### Initialize Tokenizer

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

mask_token = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

### Load

train_dir = os.path.join(data_dir, 'train_dataset')
val_dir = os.path.join(data_dir, 'val_dataset')
test_dir = os.path.join(data_dir, 'test_dataset')

element_spec = ({'input_ids': tf.TensorSpec(shape=(128,), dtype=tf.int32, name=None),
                 'attention_mask': tf.TensorSpec(shape=(128,), dtype=tf.int32, name=None),
                 'token_type_ids': tf.TensorSpec(shape=(128,), dtype=tf.int32, name=None)},
                tf.TensorSpec(shape=(2,), dtype=tf.int32, name=None))

train_data = tf.data.experimental.load(train_dir, element_spec)
val_data = tf.data.experimental.load(val_dir, element_spec)
test_data = tf.data.experimental.load(test_dir, element_spec)
X_test = np.vstack([x[0]['input_ids'].numpy() for x in test_data])
X_val = np.vstack([x[0]['input_ids'].numpy() for x in val_data])
y_test = np.vstack([y.numpy() for x,y in test_data])
y_val = np.vstack([y.numpy() for x,y in val_data])

### Batch

batch_size = 16
train_data = train_data.shuffle(20000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
val_data = val_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_data = test_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

### Get Predicted Class
from transformers import TFBertForSequenceClassification
bert_model='./model/model'
base_model = TFBertForSequenceClassification.from_pretrained(bert_model)

model = Sequential()
model.add(base_model)
model.add(tf.keras.layers.Lambda(lambda x: x.logits))
model.add(tf.keras.layers.Activation('softmax'))
for x in test_data:
    model(x)
    break

model.trainable = False

preds = model.predict(test_data)
preds_discrete = np.eye(2)[preds.argmax(1)]

preds_val = model.predict(val_data)
preds_discrete_val = np.eye(2)[preds_val.argmax(1)]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Compile Explanations
test_dir = os.path.join(method, 'test', str(num_samples))
val_dir = os.path.join(method, 'val', str(num_samples))

#### TEST ####
shap_values = []
explaining_times = []
for i in tqdm(range(len(X_test))):
    run_dir = os.path.join(test_dir, str(i))

    try:
        with open(os.path.join(run_dir, 'shap_values.pkl'), 'rb') as f:
            shap_value = pickle.load(f)

        with open(os.path.join(run_dir, 'explaining_time.pkl'), 'rb') as f:
            explaining_time = pickle.load(f)

        shap_values.append(np.array(shap_value, dtype="float32").squeeze())
        explaining_times.append(explaining_time)

    except:
        print('missing:', i)

shap_values = np.stack(shap_values, 0)
shap_values += np.random.random(shap_values.shape) * 1e-6 # Add Small Random Noise to prevent ties
shap_values = [shap_values[:, i, :] for i in range(num_classes)]
explaining_total_time = np.array(explaining_times).sum()/3600

# Save
with open(os.path.join(test_dir, 'shap_values.pkl'), 'wb') as f:
    pickle.dump(shap_values, f)

with open(os.path.join(test_dir, 'explaining_time.pkl'), 'wb') as f:
    pickle.dump(explaining_total_time, f)
    
#### VAL ####
shap_values_val = []
explaining_times = []
for i in tqdm(range(len(X_val))):
    run_dir = os.path.join(val_dir, str(i))

    try:
        with open(os.path.join(run_dir, 'shap_values.pkl'), 'rb') as f:
            shap_value = pickle.load(f)

        with open(os.path.join(run_dir, 'explaining_time.pkl'), 'rb') as f:
            explaining_time = pickle.load(f)

        shap_values_val.append(np.array(shap_value, dtype="float32").squeeze())
        explaining_times.append(explaining_time)

    except:
        print('missing:', i)

shap_values_val = np.stack(shap_values_val, 0)
shap_values_val += np.random.random(shap_values_val.shape) * 1e-6 # Add Small Random Noise to prevent ties
shap_values_val = [shap_values_val[:, i, :] for i in range(num_classes)]
explaining_total_time_val = np.array(explaining_times).sum()/3600

# Save
with open(os.path.join(val_dir, 'shap_values.pkl'), 'wb') as f:
    pickle.dump(shap_values_val, f)

with open(os.path.join(val_dir, 'explaining_time.pkl'), 'wb') as f:
    pickle.dump(explaining_total_time_val, f)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Evaluate

### Load DataFrame
df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))
df_val = pd.read_csv(os.path.join(data_dir, "val.csv"))

### Load Evaluator
evaluator_base = TFBertForSequenceClassification.from_pretrained('evaluation/evaluator-data/surrogate')    

evaluator_model = tf.keras.models.Sequential()
evaluator_model.add(evaluator_base)
evaluator_model.add(tf.keras.layers.Lambda(lambda x: x.logits))
evaluator_model.add(tf.keras.layers.Activation('softmax'))
for x in test_data:
    evaluator_model(x)
    break
evaluator_model.summary()

def eval_model(x):
    attention_mask = np.ones_like(x).astype(int)
    token_type_ids = np.zeros_like(x).astype(int)
    
    input_ = dict(
        input_ids = x.astype(int),
        attention_mask = attention_mask,
        token_type_ids = token_type_ids,
    )
    
    return evaluator_model.predict(input_)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#### Retrospective Evaluation ####

# Exclusion
retro_ex_val = evaluate(df_val.copy(), X_val, shap_values_val, evaluator_model, y_val, y_val, 
                        mode = 'exclude', method = method, mask_token=mask_token)
retro_ex_test = evaluate(df_test.copy(), X_test, shap_values, evaluator_model, y_test, y_test, 
                         mode = 'exclude', method = method, mask_token=mask_token)

# Inclusion
retro_in_val = evaluate(df_val.copy(), X_val, shap_values_val, evaluator_model, y_val, y_val, 
                        mode = 'include', method = method, mask_token=mask_token)
retro_in_test = evaluate(df_test.copy(), X_test, shap_values, evaluator_model, y_test, y_test, 
                         mode = 'include', method = method, mask_token=mask_token)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#### Prospective Evaluation ####

# Exclusion
pro_ex_val = evaluate(df_val.copy(), X_val, shap_values_val, evaluator_model, preds_discrete_val, y_val, 
                        mode = 'exclude', method = method, mask_token=mask_token)
pro_ex_test = evaluate(df_test.copy(), X_test, shap_values, evaluator_model, preds_discrete, y_test, 
                         mode = 'exclude', method = method, mask_token=mask_token)

# Inclusion
pro_in_val = evaluate(df_val.copy(), X_val, shap_values_val, evaluator_model, preds_discrete_val, y_val, 
                        mode = 'include', method = method, mask_token=mask_token)
pro_in_test = evaluate(df_test.copy(), X_test, shap_values, evaluator_model, preds_discrete, y_test, 
                         mode = 'include', method = method, mask_token=mask_token)

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