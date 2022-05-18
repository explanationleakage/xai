import tensorflow as tf
import numpy as np
import gc
from sklearn.metrics import roc_auc_score,accuracy_score,log_loss,auc,roc_curve,precision_recall_curve

def log_odds(y_true, y_pred):
    if len(y_true.shape) == 1:
        p = y_true * y_pred
    else:
        p = (y_true * y_pred).sum(1)
    p = np.clip(p, 0.0001, 0.9999)
    return np.log(p/(1-p)).mean()

def mask_eval(inputs, labels, explanations, model, p, mode = 'exclude'):
    
    if mode == 'exclude':
        if p == 100:
            masks = 0.
        elif p == 0:
            masks = 1.
        else:
            explanations_flat = explanations.reshape(explanations.shape[0], -1)
            thresholds = np.percentile(explanations_flat, 100-p, axis=1)
            masks = np.array([e < tr for e, tr in zip(explanations, thresholds)]).astype(int)  
            if len(masks.shape) < len(inputs.shape):
                masks = np.expand_dims(masks, -1)
    if mode == 'include':
        if p == 100:
            masks = 1.
        elif p == 0:
            masks = 0.
        else:
            explanations_flat = explanations.reshape(explanations.shape[0], -1)
            thresholds = np.percentile(explanations_flat, 100-p, axis=1)
            masks = np.array([e >= tr for e, tr in zip(explanations, thresholds)]).astype(int)  
            if len(masks.shape) < len(inputs.shape):
                masks = np.expand_dims(masks, -1)
            
    #################### Mask Images ####################
    masked_inputs = inputs * masks
    del masks
    gc.collect()
    
    #################### Evaluate Masked Images ####################
    y_score = model.predict(masked_inputs)
    y_pred = y_score.argmax(1)
    y_true = labels.argmax(1)
    auroc = roc_auc_score(labels, y_score)
    acc = accuracy_score(y_true, y_pred)
    ll = -(log_loss(labels, y_score))
    lo = log_odds(labels, y_score)
    
    return auroc, acc, ll, lo, y_score


def AUCs(results):
    aucs = {}
    for metric, r in results.items():
        r_ = np.array([[float(p), n] for p,n in r.items()])
        aucs['AUC_'+metric] = auc(r_[:,0], r_[:,1])/100
    results = {**results, **aucs}
        
    return results
    

def evaluate(inputs, explanations, evaluation_model, selection_labels, evaluation_labels, 
             mode = 'exclude', aucs_only = True, method = None):
    
    if not ('realx' in method) and not ('dkl' in method):
        ################### Select Explanation ###################
        explanations_select = []
        for i, yp in enumerate(selection_labels):
            yp = yp.argmax()
            explanations_select.append(explanations[yp][i])
        explanations = np.array(explanations_select)
    
    explanations_flat = explanations.reshape(explanations.shape[0], -1)
    
    #################### Generate Results ####################
    results = {}
    results['acc'] = {}
    results['auroc'] = {}
    results['preds'] = {}
    results['log_likelihood'] = {}
    results['log_odds'] = {}
    
    # Metric at Each Mask Percentage
    for p in [100, 99, 95, 90, 85, 75, 50, 25, 15, 10, 5, 1, 0]:
        print(p)
        auroc, acc, ll, lo, y_score = mask_eval(inputs, evaluation_labels, explanations, evaluation_model, p, mode = mode)
        results['acc'][p] = acc
        results['auroc'][p] = auroc
        results['preds'][p] = y_score
        results['log_likelihood'][p] = ll
        results['log_odds'][p] = lo
    
    if aucs_only:
        #AUCs
        results = AUCs(results)

    return results
        

def metrics(y_score, labels):
    y_pred = y_score.argmax(1)
    y_true = labels.argmax(1)
    auroc = roc_auc_score(labels, y_score)
    acc = accuracy_score(y_true, y_pred)
    ll = -(log_loss(labels, y_score))
    lo = log_odds(labels, y_score)
    
    return auroc, acc, ll, lo
    

def vote_metrics(df, score, y_true):
    df = df.copy()
    df['pred_score'] = score
    df['y_true'] = y_true
    df_sort = df.sort_values(by=['ID'])
    #score 
    temp = (df_sort.groupby(['ID'])['pred_score'].agg(max)+df_sort.groupby(['ID'])['pred_score'].agg(sum)/2)/(1+df_sort.groupby(['ID'])['pred_score'].agg(len)/2)
    temp_true = (df_sort.groupby(['ID'])['y_true'].agg(max)+df_sort.groupby(['ID'])['y_true'].agg(sum)/2)/(1+df_sort.groupby(['ID'])['y_true'].agg(len)/2)
    y = (temp_true > 0.5).astype(int)
    
    #Accuracy
    acc = accuracy_score(y, (temp> .5).astype(int))
    
    #AUROC
    fpr, tpr, thresholds = roc_curve(y, temp)
    auroc = auc(fpr, tpr)
    
    #AUPRC
    precision, recall, thres = precision_recall_curve(y, temp)
    auprc = auc(fpr, tpr)
    
    #Log-odds + Log_likelihood
    ll = -(log_loss(y, temp))
    lo = log_odds(y, temp)
    
    return auroc, acc, ll, lo

def mask_eval_mimic(df, inputs, labels, explanations, model, p, mode = 'exclude', mask_token=103):
    
    if mode == 'exclude':
        if p == 100:
            masks = 0
        elif p == 0:
            masks = 1
        else:
            explanations_flat = explanations.reshape(explanations.shape[0], -1)
            thresholds = np.percentile(explanations_flat, 100-p, axis=1)
            masks = np.array([e < tr for e, tr in zip(explanations, thresholds)]).astype(int)  
            if len(masks.shape) < len(inputs.shape):
                masks = np.expand_dims(masks, -1)
    if mode == 'include':
        if p == 100:
            masks = 1
        elif p == 0:
            masks = 0
        else:
            explanations_flat = explanations.reshape(explanations.shape[0], -1)
            thresholds = np.percentile(explanations_flat, 100-p, axis=1)
            masks = np.array([e >= tr for e, tr in zip(explanations, thresholds)]).astype(int)  
            if len(masks.shape) < len(inputs.shape):
                masks = np.expand_dims(masks, -1)
            
    #################### Mask Images ####################
    masked_inputs = (masks * inputs) + ((1-masks)*mask_token)
    print(masked_inputs.shape)
    del masks
    gc.collect()
    
    #################### Evaluate Masked Images ####################
    y_score = model(masked_inputs)
    auroc, acc, ll, lo = vote_metrics(df, y_score[:,1], labels[:,1])
    
    return auroc, acc, ll, lo, y_score


def evaluate_mimic(df, inputs, explanations, evaluation_model, selection_labels, evaluation_labels, mode = 'exclude', method = None, mask_token=103):
    
    if not ('realx' in method) and not ('dkl' in method):
        ################### Select Explanation ###################
        explanations_select = []
        for i, yp in enumerate(selection_labels):
            yp = yp.argmax()
            explanations_select.append(explanations[yp][i])
        explanations = np.array(explanations_select)
    
    explanations_flat = explanations.reshape(explanations.shape[0], -1)
    
    #################### Generate Results ####################
    results = {}
    results['acc'] = {}
    results['auroc'] = {}
    results['preds'] = {}
    results['log_likelihood'] = {}
    results['log_odds'] = {}
    
    # Metric at Each Mask Percentage
    for p in [100, 99, 95, 90, 85, 75, 50, 25, 15, 10, 5, 1, 0]:
        print(p)
        auroc, acc, ll, lo, y_score = mask_eval_mimic(df, inputs, evaluation_labels, 
                                                      explanations, evaluation_model, p, mode = mode, mask_token=mask_token)
        results['acc'][p] = acc
        results['auroc'][p] = auroc
        results['preds'][p] = y_score
        results['log_likelihood'][p] = ll
        results['log_odds'][p] = lo
        
    #AUCs
    results = AUCs(results)
        
    return results   