import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import shap
import sys, os
import time
from tqdm.notebook import tqdm

from tensorflow.keras.layers import (Input, Layer, Dense, GlobalAveragePooling1D, UpSampling1D)
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K

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
parser = argparse.ArgumentParser(description='Kernal SHAP ECG Explainer')
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

# Load Model

#Stanford Model
sys.path.insert(0, '/scratch/nj594/ecg/models/stanford')
import network

model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'surrogate', 'surrogate.h5'))    

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Get Arguments
for split in ['test', 'val']:
    
    # Continue on if Out of Range
    if split == 'val' and args.index > 2155:
        print('Index out of Range')
        continue
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Load and Select ECG

    data_dir = os.path.join(os.getcwd(), 'data')
    ecgs = np.load(os.path.join(data_dir, 'X_{}.npy'.format(split)), allow_pickle=True)
    labels = np.load(os.path.join(data_dir, 'y_{}.npy'.format(split)), allow_pickle=True)
    if split == 'test':
        preds = np.load(os.path.join(data_dir, 'predictions.npy'), allow_pickle=True)
    else:
        preds = np.load(os.path.join(data_dir, 'predictions_val.npy'), allow_pickle=True)

    background = None
        
    ECG = ecgs[args.index]

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # Explain ECG

    ### Generate Masked ECG Prediction Function

    # Mask Function, Takes ecg, mask, background dataset 
    # --> Resizes Mask from flat 125 --> 1000
    # background=None
    def mask_ecg(masks, ecg, background=None):
        # Reshape/size Mask
        mask_len = masks.shape[1]
        masks = np.expand_dims(masks, -1)
        resize_aspect = ecg.shape[0]/mask_len
        masks = np.repeat(masks, resize_aspect, axis =1)

        # Mask ECG 
        if background is not None:
            if len(background.shape) == 2:
                masked_ecgs = np.vstack([np.expand_dims(
                    (mask * ecg) + ((1-mask)*background[0]), 0
                ) for mask in masks])
            else:
                # Fill with Background
                masked_ecgs = []
                for mask in masks:
                    bg = [im * (1-mask) for im in background]
                    masked_ecgs.append(np.vstack([np.expand_dims((mask*ecg) + fill, 0) for fill in bg]))     
        else:     
            masked_ecgs = np.vstack([np.expand_dims(mask * ecg, 0) for mask in masks])

        return masked_ecgs #masks, ecg

    # Function to Make Predictions from Masked Images
    def f_mask(z):
        if background is None or len(background.shape)==2:
            y_p = []
            if z.shape[0] == 1:
                masked_ecgs = mask_ecg(z, ECG, background)
                return(model(masked_ecgs).numpy())
            else:
                for i in tqdm(range(int(math.ceil(z.shape[0]/100)))):
                    m = z[i*100:(i+1)*100]
                    masked_ecgs = mask_ecg(m, ECG, background)
                    y_p.append(model(masked_ecgs).numpy())
                print (np.vstack(y_p).shape)
                return np.vstack(y_p)
        else:
            y_p = []
            if z.shape[0] == 1:
                masked_ecgs = mask_ecg(z, ECG, background)
                for masked_ecg in masked_ecgs:
                    y_p.append(np.mean(model(masked_ecg), 0))
            else:
                for i in tqdm(range(int(math.ceil(z.shape[0]/100)))):
                    m = z[i*100:(i+1)*100]
                    masked_ecgs = mask_ecg(m, ECG, background)
                    for masked_ecg in masked_ecgs:
                        y_p.append(np.mean(model(masked_ecg), 0))
            return np.vstack(y_p)

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for num_samples in [2**9, 2**10, 2**11, 2**12, 2**13]:

        #Set Model Dir
        exp = 'kernelshap_s'
        exp_dir = os.path.join(exp, split, str(num_samples), str(args.index))
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
        print(exp_dir)
            
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        ### Explain with Kernel SHAP    

        explainer = shap.KernelExplainer(f_mask, np.zeros((1,125)))
        t = time.time()
        shap_values = explainer.shap_values(np.ones((1,125)), nsamples=num_samples, l1_reg=False)
        explaining_time = time.time() - t

        def resize_mask(masks, ecg):
            # Reshape/size Mask
            mask_len = masks.shape[1]
            resize_aspect = ecg.shape[0]/mask_len
            masks = np.repeat(masks, resize_aspect, axis =1)

            return masks

        shap_values = [resize_mask(sv, ECG)  for sv in shap_values]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        ### Save

        with open(os.path.join(exp_dir, 'explaining_time.pkl'), 'wb') as f:
            pickle.dump(explaining_time, f)

        with open(os.path.join(exp_dir, 'shap_values.pkl'), 'wb') as f:
            pickle.dump(shap_values, f)