import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import shap
import sys, os
import time
from tqdm.notebook import tqdm

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model, Sequential

from scipy.stats import entropy

import argparse
import pickle
import math

# IMPORTANT: SET RANDOM SEEDS FOR REPRODUCIBILITY
os.environ['PYTHONHASHSEED'] = str(420)
import random
random.seed(420)
np.random.seed(420)
tf.random.set_seed(420)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Command Line Arguements
parser = argparse.ArgumentParser(description='Imagenette Kernal SHAP Explainer')
parser.add_argument('--index', type=int, default=9999, metavar='i',
                    help='Index for Job Array')
args = parser.parse_args()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Get Index (Either from argument or from SLURM JOB ARRAY)
if 'SLURM_ARRAY_TASK_ID' in os.environ:
    args.index = int(os.environ['SLURM_ARRAY_TASK_ID'])
#     args.index += 3000
    print('SLURM_ARRAY_TASK_ID found..., using index %s' % args.index)
else:
    print('no SLURM_ARRAY_TASK_ID... using index %s' % args.index)
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Data

data_dir = './data'
label_list = [0, 1]
max_seq_length = 128

test_dir = os.path.join(data_dir, 'test_dataset')
val_dir = os.path.join(data_dir, 'val_dataset')

element_spec = ({'input_ids': tf.TensorSpec(shape=(128,), dtype=tf.int32, name=None),
                 'attention_mask': tf.TensorSpec(shape=(128,), dtype=tf.int32, name=None),
                 'token_type_ids': tf.TensorSpec(shape=(128,), dtype=tf.int32, name=None)},
                tf.TensorSpec(shape=(2,), dtype=tf.int32, name=None))

test_data = tf.data.experimental.load(test_dir, element_spec)
test_data = [x[0]['input_ids'].numpy() for x in test_data]

val_data = tf.data.experimental.load(val_dir, element_spec)
val_data = [x[0]['input_ids'].numpy() for x in val_data]

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

mask_token = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Surrogate Model

from transformers import TFBertForSequenceClassification
model = TFBertForSequenceClassification.from_pretrained('surrogate/surrogate')    

### Edit Model to Only Require input_ids
def prediction_model(x):
    attention_mask = np.ones_like(x).astype(int)
    token_type_ids = np.zeros_like(x).astype(int)
    
    input_ = dict(
        input_ids = x.astype(int),
        attention_mask = attention_mask,
        token_type_ids = token_type_ids,
    )
    
    logits = model(input_).logits
    probs = tf.keras.layers.Softmax()(logits)
    
    return probs

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Get Arguments
for split in ['test', 'val']:
    
    # Continue on if Out of Range
    if ((split == 'val' and args.index > len(val_data)) or 
        (split == 'test' and args.index > len(test_data))):
        print('Index out of Range')
        continue
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Select Text
    if split == 'test':
        TEXT = test_data[args.index]
    else:
        TEXT = val_data[args.index]
        
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Explain Text

    ### Generate Masked Text Prediction Function

    # Mask Function, Takes test, mask, mask_token
    def mask_text(masks, text, mask_token=None):

        # Mask text 
        if mask_token is not None:
            masked_text = np.vstack([np.expand_dims(
                    (mask * text) + ((1-mask)*mask_token), 0
                ) for mask in masks])     
        else:     
            masked_text = np.vstack([np.expand_dims(mask * text, 0) for mask in masks])

        return masked_text #masks, ecg

    # Function to Make Predictions from Masked Text
    def f_mask(z):
        y_full = prediction_model(np.expand_dims(TEXT, 0)).numpy()
        y_p = []
        if z.shape[0] == 1:
            masked_text = mask_text(z, TEXT, mask_token)
            y_p = prediction_model(masked_text).numpy()
            kl = -entropy(y_full, qk=y_p, base=None, axis=1)
            return kl
        else:
            for i in tqdm(range(int(math.ceil(z.shape[0]/100)))):
                m = z[i*100:(i+1)*100]
                masked_text = mask_text(m, TEXT, mask_token)
                y_p.append(prediction_model(masked_text).numpy())
            y_p = np.vstack(y_p)
            y_full = np.tile(y_full, (len(y_p),1))
            kl = -entropy(y_full, qk=y_p, base=None, axis=1)
            return kl

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for num_samples in [2**9, 2**10, 2**11, 2**12, 2**13]:

        #Set Model Dir
        exp = 'kernelshap_s-dkl'
        exp_dir = os.path.join(exp, split, str(num_samples), str(args.index))
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        print(exp_dir)
            
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        ### Explain with Kernel SHAP  
        explainer = shap.KernelExplainer(f_mask, np.zeros((1,128), dtype=int))
        t = time.time()
        shap_values = explainer.shap_values(np.ones((1,128), dtype=int), 
                                            nsamples=num_samples, l1_reg=False)
        explaining_time = time.time() - t
        
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        ### Save
        with open(os.path.join(exp_dir, 'explaining_time.pkl'), 'wb') as f:
            pickle.dump(explaining_time, f)

        with open(os.path.join(exp_dir, 'shap_values.pkl'), 'wb') as f:
            pickle.dump(shap_values, f)