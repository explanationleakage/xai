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

sys.path.insert(0, '/scratch/nj594/mimic_explain/fastshap_text')
from surrogate import TextSurrogate
from fastshap_text import FastSHAP

sys.path.insert(0, '/scratch/nj594/xai/helpers')
from evaluate import evaluate_mimic as evaluate

# IMPORTANT: SET RANDOM SEEDS FOR REPRODUCIBILITY
os.environ['PYTHONHASHSEED'] = str(420)
import random
random.seed(420)
np.random.seed(420)
tf.random.set_seed(420)


#Select GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
with open(args.arg_file, "rb") as arg_file:
    args.arg_file = pickle.load(arg_file)

args.arg_file = args.arg_file[args.index]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Set Model Dir
method = 'fastshap'
run = str(args.index)
model_dir = os.path.join(method, run)
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)

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

# Load Surrogate

from transformers import TFBertForSequenceClassification
surrogate_model = TFBertForSequenceClassification.from_pretrained('surrogate/surrogate')    
surrogate = TextSurrogate(surrogate_model = surrogate_model,
                          seq_length = max_seq_length,
                          baseline = mask_token)

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

# FastSHAP

### Specify Explainer Architecture

from transformers.models.bert.modeling_tf_bert import TFBertPredictionHeadTransform
bert_model='./model/model'
base_model = TFBertForSequenceClassification.from_pretrained(bert_model)

bert_main = base_model.layers[0]

inputs = {}
for k, v in train_data.unbatch().element_spec[0].items():
    inputs[k] = tf.keras.layers.Input(shape = v.shape, name = k, dtype = v.dtype) 
input_key = [k for k in train_data.element_spec[0].keys() if 'input' in k.lower()][0] 

bert_out = bert_main(inputs)

net = TFBertPredictionHeadTransform(config = bert_main.config)(bert_out[0])
out = Dense(2)(net)

explainer = Model(inputs, out)

### Train FastSHAP
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
    print('Training FastSHAP')
    fastshap = FastSHAP(explainer = explainer,
                        imputer = surrogate,
                        baseline = mask_token,
                        normalization=args.arg_file['normalization'],
                        link='identity')

    t = time.time()
    fastshap.train(train_data = train_data,
                   val_data = val_data,
                   batch_size = args.arg_file['batch_size'],
                   num_samples = args.arg_file['num_samples'],
                   max_epochs = args.arg_file['epochs'],
                   validation_batch_size = args.arg_file['batch_size'],
                   lr=args.arg_file['lr'],
                   min_lr=1e-5,
                   lr_factor=0.9,
                   eff_lambda=args.arg_file['eff_lambda'],
                   paired_sampling=args.arg_file['paired_sampling'],
                   lookback=args.arg_file['lookback'],
                   verbose=1, 
                   model_dir = model_dir)
    training_time = time.time() - t

    with open(os.path.join(model_dir, 'training_time.pkl'), 'wb') as f:
        pickle.dump(training_time, f)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Explain w/ FastSHAP

    ## TEST
    print('Explaining Test')

    ### Explain

    t = time.time()
    shap_values = fastshap.shap_values(test_data)
    explaining_time = time.time() - t
    shap_values = [shap_values[:, i, :] for i in range(num_classes)]

    ### Save

    with open(os.path.join(model_dir, 'explaining_time.pkl'), 'wb') as f:
        pickle.dump(explaining_time, f)

    with open(os.path.join(model_dir, 'shap_values.pkl'), 'wb') as f:
        pickle.dump(shap_values, f)

    ## VAL
    print('Explaining Val')

    ### Explain

    t = time.time()
    shap_values_val = fastshap.shap_values(val_data)
    explaining_time_val = time.time() - t
    shap_values_val = [shap_values_val[:, i, :] for i in range(num_classes)]

    ### Save

    with open(os.path.join(model_dir, 'explaining_time_val.pkl'), 'wb') as f:
        pickle.dump(explaining_time_val, f)

    with open(os.path.join(model_dir, 'shap_values_val.pkl'), 'wb') as f:
        pickle.dump(shap_values_val, f)

    loss = np.min(fastshap.val_losses)
    with open(os.path.join(model_dir, 'loss.pkl'), 'wb') as f:
        pickle.dump(loss, f)
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### Add Small Random Noise to prevent ties
#### TEST
shap_values = [sv + (np.random.random(sv.shape) * 1e-6) for sv in shap_values]
#### VAL
shap_values_val = [sv + (np.random.random(sv.shape) * 1e-6) for sv in shap_values_val]

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
header = ["model_dir", "lr", "epochs", "batch_size", "lookback", 
          "num_samples", "paired_sampling", "eff_lambda", "normalization", 
          "training_time", "explaining_time"]
metrics = ['AUC_acc','AUC_auroc','AUC_log_likelihood','AUC_log_odds']
for tag in tags:
    header += [x+'-'+tag for x in metrics]
    
results = {**results, **args.arg_file}
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