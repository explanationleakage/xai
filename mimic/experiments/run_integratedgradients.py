import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model, Sequential

import numpy as np
import pandas as pd
import sys, os
import time
import pickle
from datetime import datetime
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
batch_size = 16
index = args.index
NUM_CLASSES = 2  
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
# Load Data

data_dir = './data'
label_list = [0, 1]
max_seq_length = 128

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
n_test = len(test_data)
n_val = len(val_data)

### Batch

train_data = train_data.shuffle(20000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
val_data = val_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_data = test_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Model

from transformers import TFBertForSequenceClassification
bert_model='./model/model'
model = TFBertForSequenceClassification.from_pretrained(bert_model)

### Define Functions that Split the Model

def embedding_model(inputs):
    
    input_shape = list(inputs["input_ids"].shape)

    embedding_output = model.bert.embeddings(input_ids = inputs['input_ids'], 
                                             position_ids = None,
                                             token_type_ids = inputs['token_type_ids'])
    
    return embedding_output

def prediction_model(embedding_output):
    attention_mask = tf.ones(embedding_output.shape[:2])
    attention_mask = tf.cast(attention_mask, dtype=tf.float32)
    
    input_shape = attention_mask.shape
    
    extended_attention_mask = tf.reshape(attention_mask, (input_shape[0], 1, 1, input_shape[1]))
    extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)
    one_cst = tf.constant(1.0, dtype=embedding_output.dtype)
    ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)
    extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)
        
    head_mask = [None] * model.bert.config.num_hidden_layers

    encoder_outputs = model.bert.encoder(hidden_states = embedding_output, 
                                     attention_mask = extended_attention_mask, 
                                     head_mask = head_mask, 
                                     encoder_hidden_states=None,
                                     encoder_attention_mask=None,
                                     past_key_values=None,
                                     use_cache=False,
                                     output_attentions=False,
                                     output_hidden_states=False,
                                     return_dict=False,
                                     training=False)
    
    sequence_output = encoder_outputs[0]
    pooled_output = model.bert.pooler(sequence_output)
    logits = model.classifier(pooled_output)
    return logits

##############################################################

X_test = tf.constant(np.vstack([embedding_model(x).numpy() for x, y in test_data]))
X_val = tf.constant(np.vstack([embedding_model(x).numpy() for x, y in val_data]))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

### Get [MASK] Baseline Values
for x, y in test_data:
    break
baseline_ids = {k:tf.cast(tf.ones((1, max_seq_length)) * mask_token if k=='input_ids' 
                          else tf.expand_dims(v[0], 0), tf.int32) 
                for k, v in x.items()}
baseline_embedding = embedding_model(baseline_ids)

# Get Arguments
for split in ['test', 'val']:
    
    # Continue on if Out of Range
    if ((split == 'val' and index > n_val) or 
        (split == 'test' and index > n_test)):
        print('Index out of Range')
        continue
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Select Text
    if split == 'test':
        TEXT = X_test[index]
    else:
        TEXT = X_val[index]
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # Explain TEXT
    
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

        ## Calculate Gradients
        t = time.time()

        explanations = []
        for y_class in range(NUM_CLASSES):
            ### Reshape for Interpolations
            current_input = np.expand_dims(TEXT, axis=0)
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

                batch_baseline = tf.convert_to_tensor(np.tile(baseline_embedding, reps))

                batch_difference = batch_input - batch_baseline
                batch_interpolated = batch_alphas * batch_input + \
                                     (1.0 - batch_alphas) * batch_baseline

                with tf.GradientTape() as tape:
                    tape.watch(batch_interpolated)

                    batch_predictions = prediction_model(batch_interpolated)
                    ### Get prediction of the selected class
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