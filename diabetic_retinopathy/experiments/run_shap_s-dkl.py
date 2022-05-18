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
parser = argparse.ArgumentParser(description='Kernal SHAP Eye Explainer')
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
    
INPUT_SHAPE = (544, 544, 3)
NUM_CLASSES = 5
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load Surrogate Model
from tensorflow.keras.applications.densenet import DenseNet121

model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'surrogate/densenet', 'surrogate.h5'))

##############################################################

# Get Arguments
for split in ['test', 'val']:
    
    # Continue on if Out of Range
    if split == 'val' and args.index > 999:
        print('Index out of Range')
        continue
        
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Load and Select Images

    data_dir = os.path.join(os.getcwd(), 'data')
    images = np.load(os.path.join(data_dir, 'X_{}_processed.npy'.format(split)), allow_pickle=True)
    labels = np.load(os.path.join(data_dir, 'y_{}.npy'.format(split)), allow_pickle=True)
    if split == 'test':
        preds = np.load(os.path.join(data_dir, 'predictions.npy'), allow_pickle=True)
    else:
        preds = np.load(os.path.join(data_dir, 'predictions_val.npy'), allow_pickle=True)

    background = None
        
    IMAGE = images[args.index]

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # Explain IMAGE

    ### Generate Masked IMAGE Prediction Function

    # Mask Function, Takes ecg, mask, background dataset 
    # --> Resizes Mask from flat 125 --> 1000
    # background=None
    def mask_image(masks, image, background=None):
        # Reshape/size Mask 
        mask_shape = int(masks.shape[1]**.5)
        masks = np.reshape(masks, (masks.shape[0], mask_shape, mask_shape, 1))
        resize_aspect = image.shape[0]/mask_shape
        masks = np.repeat(masks, resize_aspect, axis =1)
        masks = np.repeat(masks, resize_aspect, axis =2)

        # Mask Image 
        if background is not None:
            if len(background.shape) == 3:
                masked_images = np.vstack([np.expand_dims(
                    (mask * image) + ((1-mask)*background[0]), 0
                ) for mask in masks])
            else:
                # Fill with Background
                masked_images = []
                for mask in masks:
                    bg = [im * (1-mask) for im in background]
                    masked_images.append(np.vstack([np.expand_dims((mask*image) + fill, 0) for fill in bg]))     
        else:     
            masked_images = np.vstack([np.expand_dims(mask * image, 0) for mask in masks])

        return masked_images #masks, image

    # Function to Make Predictions from Masked Images
    def f_mask(z):
        y_full = model(np.expand_dims(IMAGE, 0)).numpy()
        if background is None or len(background.shape)==3:
            y_p = []
            if z.shape[0] == 1:
                masked_images = mask_image(z, IMAGE, background)
                y_p = model(masked_images).numpy()
                kl = -entropy(y_full, qk=y_p, base=None, axis=1)
                return kl
            else:
                for i in tqdm(range(int(math.ceil(z.shape[0]/100)))):
                    m = z[i*100:(i+1)*100]
                    masked_images = mask_image(m, IMAGE, background)
                    y_p.append(model(masked_images).numpy())
                y_p = np.vstack(y_p)
                y_full = np.tile(y_full, (len(y_p),1))
                kl = -entropy(y_full, qk=y_p, base=None, axis=1)
                return kl
        else:
            y_p = []
            if z.shape[0] == 1:
                masked_images = mask_image(z, IMAGE, background)
                for masked_image in masked_images:
                    y_p.append(np.mean(model(masked_image), 0))
            else:
                for i in tqdm(range(int(math.ceil(z.shape[0]/100)))):
                    m = z[i*100:(i+1)*100]
                    masked_images = mask_image(m, IMAGE, background)
                    for masked_image in masked_images:
                        y_p.append(np.mean(model(masked_image), 0))
            return np.vstack(y_p)

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
        
        explainer = shap.KernelExplainer(f_mask, np.zeros((1,17*17)))
        t = time.time()
        shap_values = explainer.shap_values(np.ones((1,17*17)), nsamples=num_samples, l1_reg=False)
        explaining_time = time.time() - t

        def resize_mask(masks, image):
            mask_shape = int(masks.shape[1]**.5)
            masks = np.reshape(masks, (masks.shape[0], mask_shape, mask_shape, 1))
            resize_aspect = image.shape[0]/mask_shape
            masks = np.repeat(masks, resize_aspect, axis =1)
            masks = np.repeat(masks, resize_aspect, axis =2)

            return masks
        
        shap_values = resize_mask(shap_values, IMAGE)

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        ### Save

        with open(os.path.join(exp_dir, 'explaining_time.pkl'), 'wb') as f:
            pickle.dump(explaining_time, f)

        with open(os.path.join(exp_dir, 'shap_values.pkl'), 'wb') as f:
            pickle.dump(shap_values, f)