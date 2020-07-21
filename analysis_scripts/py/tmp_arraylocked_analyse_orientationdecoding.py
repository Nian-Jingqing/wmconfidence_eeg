#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 12:01:03 2020

@author: sammirc
"""
import numpy as np
import scipy as sp
import pandas as pd
import mne
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
from scipy import ndimage

#stuff for the machine learning

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import sklearn as skl

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri/analysis_scripts')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/glm')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
#%%

def smooth(signal, twin = 10, method = 'boxcar'):
    #twin is number of samples (for 500Hz, 10 samples is 20ms)
    filt = sp.signal.windows.boxcar(twin)
    smoothed = np.convolve(filt/filt.sum(), signal, mode = 'same')
    return smoothed

#if the orientation decoding has already been run and you want to look at binned analyses, use this.
#there's a separate script for running glms based on the trialwise decoding

nconds = 2 #2 conditions, neutral vs cued; if orientations at encoding, this is still 2 (left and right orientations)
nbins  = 12 #orientation decoding uses 12 angular bins
n_oris = 2
srate  = 500
ntimes = srate*2

nbins_behav            = 3
subject_accuracies     = np.zeros(shape = (subs.size, nbins_behav, ntimes)) * np.nan
subject_predict_probas = np.zeros(shape = (subs.size, nbins_behav, ntimes)) * np.nan

metric_to_use = 'predict proba'
print('using trialwise target class prediction probabilities')


for i in range(len(subs)):
    isub = subs[i]
    
    sub = dict(loc = 'workstation', id = isub)
    param = get_subject_info_wmConfidence(sub)
    print('getting decoding and running subject %d/%d'%(i+1,subs.size))

    
    predictors     = np.load(op.join(wd, 'data', 'decoding', 'arraylocked', 'orientation', 's%02d_orientationdecoding_preds_data.npy'%(isub)))
    label_preds    = np.load(op.join(wd, 'data', 'decoding', 'arraylocked', 'orientation', 's%02d_orientationdecoding_labelpred_data.npy'%(isub)))
    predict_probas = np.load(op.join(wd, 'data', 'decoding', 'arraylocked', 'orientation', 's%02d_orientationdecoding_predictproba_data.npy'%(isub)))
    
    #predictors     -- the orientations for the left and the right items (orientation classes)
    #label_preds    -- the predicted orientation class at each time point on each trial
    #predict_probas -- on each trial, the probability of each class being predicted for each orientation, at every time point
    
    bdata = mne.epochs.read_epochs(fname = param['arraylocked'].replace('arraylocked', 'arraylocked_cleaned'), verbose=False).metadata
    if 'times' not in globals().keys():
        times = mne.epochs.read_epochs(fname = param['arraylocked'].replace('arraylocked', 'arraylocked_cleaned'), verbose=False).resample(srate).times
        
    ntrls = bdata.shape[0]
    
    if metric_to_use == 'predict proba':
        #we're going to pass forward the predicted probability of the target class rather than accuracy
        
        binvar = 'confidence'
        if binvar == 'confidence':
            behvar = bdata.confwidth.to_numpy()
            behvar = np.multiply(behvar, -1)
        elif binvar == 'error':
            behvar = bdata.error.to_numpy()
            
        if nbins_behav == 3:
            #split into thirds, so we need the 33rd and 66th percentiles
            percentiles_behav = np.quantile(behvar, [.33, .66])
            #get indices for relevant trials
            bottom_third = behvar <= percentiles_behav[0]
            middle_third = np.logical_and(behvar > percentiles_behav[0], behvar <= percentiles_behav[1])
            top_third    = behvar > percentiles_behav[1]
            
            #we need to squash predict_probas down to on dimension which is just the target class predict_proba
            
            trl_target_proba = np.zeros(shape = (predict_probas.shape[0], n_oris, ntimes)) * np.nan
            
            for trl in range(ntrls):
                for ori in range(n_oris):
                    trl_target_proba[trl, ori,:] = predict_probas[trl,predictors[ori,trl],ori,:]
                    
            #trl_target_proba: the probability of predicting the target class at each time point
            trl_target_proba = np.mean(trl_target_proba, axis = 1) #average across items in the array
            
            #smooth the single trial traces a little bit
            smooth_type = 'gaussian'
            if smooth_type == 'boxcar':
                smoothing = 20 #smoothing window in ms
                for trl in range(ntrls):
                    trl_target_proba[trl] = smooth(trl_target_proba[trl], twin = int(smoothing/(1000/srate)))
            elif smooth_type == 'gaussian':
                smoothing = 4 #standard deviation of the gaussian filter
                trl_target_proba = sp.ndimage.gaussian_filter1d(trl_target_proba, sigma = smoothing)
                
            
            bottom_beh_predict = trl_target_proba[bottom_third]
            middle_beh_predict = trl_target_proba[middle_third]
            top_beh_predict    = trl_target_proba[top_third]
            
            subject_predict_probas[i,0,:] = trl_target_proba[bottom_third].mean(0)
            subject_predict_probas[i,1,:] = trl_target_proba[middle_third].mean(0)
            subject_predict_probas[i,2,:] = trl_target_proba[top_third].mean(0)
        
    elif metric_to_use =='accuracy':
        #here we will calculate the accuracy across trials for each bin of the behavioural data
        
        binvar = 'confidence'
        if binvar == 'confidence':
            behvar = bdata.confwidth.to_numpy()
            behvar = np.multiply(behvar, -1)
        elif binvar == 'error':
            behvar = bdata.error.to_numpy()
            
        if nbins_behav == 3:
            #split into thirds, so we need the 33rd and 66th percentiles
            percentiles_behav = np.quantile(behvar, [.33, .66])
            #get indices for relevant trials
            bottom_third = behvar <= percentiles_behav[0]
            middle_third = np.logical_and(behvar > percentiles_behav[0], behvar <= percentiles_behav[1])
            top_third    = behvar > percentiles_behav[1]
            
            #accuracy constructed separately for each item, then averaged across items
            
            low_conf_acc  = np.zeros(shape = (n_oris, ntimes)) * np.nan
            mid_conf_acc  = np.zeros(shape = (n_oris, ntimes)) * np.nan
            high_conf_acc = np.zeros(shape = (n_oris, ntimes)) * np.nan
        
            for ori in range(n_oris):
                for tp in range(ntimes):
                    low_conf_acc[ori,tp] = skl.metrics.accuracy_score(predictors[ori, bottom_third], label_preds[bottom_third,ori,tp])
                    mid_conf_acc[ori,tp] = skl.metrics.accuracy_score(predictors[ori, middle_third], label_preds[middle_third,ori,tp])
                    high_conf_acc[ori,tp] = skl.metrics.accuracy_score(predictors[ori, top_third],   label_preds[top_third,ori,tp])
            
            #average across items in the array now
            
            low_conf_acc = np.squeeze(np.nanmean(low_conf_acc,0))
            mid_conf_acc = np.squeeze(np.nanmean(mid_conf_acc,0))
            high_conf_acc = np.squeeze(np.nanmean(high_conf_acc,0))
            
            
            #and now finally we will smooth the subject trace
            smooth_type = 'gaussian'
            if smooth_type == 'boxcar':
                smoothing = 20 #smoothing window in ms
                low_conf_acc = smooth( low_conf_acc, twin = int(smoothing/(1000/srate) ) )
                mid_conf_acc = smooth( mid_conf_acc, twin = int(smoothing/(1000/srate) ) )
                high_conf_acc = smooth( high_conf_acc, twin = int(smoothing/(1000/srate) ))
            elif smooth_type == 'gaussian':
                smoothing = 4 #standard deviation of the gaussian filter
                low_conf_acc = sp.ndimage.gaussian_filter1d(low_conf_acc, sigma = smoothing)
                mid_conf_acc = sp.ndimage.gaussian_filter1d(mid_conf_acc, sigma = smoothing)
                high_conf_acc = sp.ndimage.gaussian_filter1d(high_conf_acc, sigma = smoothing)
        subject_accuracies[i,0,:] = low_conf_acc
        subject_accuracies[i,1,:] = mid_conf_acc
        subject_accuracies[i,2,:] = high_conf_acc
#%%

#just standardise the output variable we're dealing with now
if metric_to_use == 'predict proba':
    subject_output = deepcopy(subject_predict_probas)
elif metric_to_use == 'accuracy':
    subject_output = deepcopy(subject_accuracies)

plotmean = np.nanmean(subject_output, axis = 0)
plotsem  = sp.stats.sem(subject_output, axis = 0)
labels = ['low', 'medium', 'high']
labels = [x + ' %s'%binvar for x in labels]

colors = ['#bdbdbd', '#b2df8a', '#1f78b4']

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(labels)):
    ax.plot(times, plotmean[i], lw = .75, label = labels[i], color = colors[i])
    ax.fill_between(times, plotmean[i]-plotsem[i], plotmean[i]+plotsem[i], color = colors[i], alpha = .2)
ax.set_ylabel('orientation decoding accuracy')
ax.set_xlabel('time relative to array onset (s)')
ax.axhline(1/nbins, lw = .5, ls = '--', color = 'k')
ax.set_xlim([-0.3,1])
fig.legend()















