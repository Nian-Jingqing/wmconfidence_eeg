#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:11:09 2020

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

tmin, tmax = -0.5, 2.0
nconds = 2 #2 conditions, neutral vs cued
srate = 500
ntimes = np.arange(tmin, tmax, 1/srate).size

nbins_behav            = 2
subject_accuracies     = np.zeros(shape = (subs.size, nbins_behav, ntimes)) * np.nan
subject_predictprobas  = np.zeros(shape = (subs.size, ntimes)) * np.nan
subject_trialwise_predictprobas = dict()

#metric_to_use = 'predict proba'
#print('using trialwise target class prediction probabilities')
for i in range(len(subs)):
    isub = subs[i]
    
    sub = dict(loc = 'workstation', id = isub)
    param = get_subject_info_wmConfidence(sub)
    print('getting decoding and running subject %d/%d'%(i+1,subs.size))


    predict_probas = np.load(op.join(wd, 'data', 'decoding', 'cuelocked', 'target_side', 's%02d_sidedecoding_predictprobas_data.npy'%(isub)))
    label_preds    = np.load(op.join(wd, 'data', 'decoding', 'cuelocked', 'target_side', 's%02d_sidedecoding_labelpreds_data.npy'%(isub)))
    predictor      = np.load(op.join(wd, 'data', 'decoding', 'cuelocked', 'target_side', 's%02d_sidedecoding_predictors_data.npy'%(isub)))
    bdata          = pd.read_csv(op.join(wd,'data','decoding','cuelocked','target_side','s%02d_sidedecoding_bdata.csv'%(isub)))
    
    if 'times' not in globals().keys():
        times = np.load(op.join(wd, 'data', 'decoding', 'cuelocked', 'target_side', 'times_sidedecoding.npy') )
        
    
    #get ur behavioural var of interest
    behav_varname = 'error'
    if behav_varname == 'error':
        behvar = bdata.absrdif.to_numpy()
    elif behav_varname == 'confidence':
        behvar = bdata.confwidth.to_numpy()
        behvar = np.multiply(behvar, -1) #flip around zero so higher values (less -ve) indicate higher confidence, lower values are lower confidence
        
    if nbins_behav == 3:
        percentiles_behav = np.quantile(behvar, [0.33,0.66])
        #get indices for relevant trials
        bottom_third = behvar <= percentiles_behav[0]
        middle_third = np.logical_and(behvar > percentiles_behav[0], behvar <= percentiles_behav[1])
        top_third    = behvar > percentiles_behav[1]
    
    
        low_third_acc  = np.zeros(ntimes) * np.nan
        mid_third_acc  = np.zeros(ntimes) * np.nan
        high_third_acc = np.zeros(ntimes) * np.nan
            
        for tp in range(ntimes):
            low_third_acc[tp] = skl.metrics.accuracy_score(predictor[bottom_third], label_preds[bottom_third,tp])
            mid_third_acc[tp] = skl.metrics.accuracy_score(predictor[middle_third], label_preds[middle_third,tp])
            high_third_acc[tp] = skl.metrics.accuracy_score(predictor[top_third],   label_preds[top_third,tp])
                
        #smooth the subject accuracy trace
        smooth_type = 'gaussian'
        if smooth_type == 'boxcar':
            smoothing = 20 #smoothing window in ms
            low_third_acc = smooth( low_third_acc, twin = int(smoothing/(1000/srate) ) )
            mid_third_acc = smooth( mid_third_acc, twin = int(smoothing/(1000/srate) ) )
            high_third_acc = smooth( high_third_acc, twin = int(smoothing/(1000/srate) ))
        elif smooth_type == 'gaussian':
            smoothing = 4 #standard deviation of the gaussian filter
            low_third_acc = sp.ndimage.gaussian_filter1d(low_third_acc, sigma = smoothing)
            mid_third_acc = sp.ndimage.gaussian_filter1d(mid_third_acc, sigma = smoothing)
            high_third_acc = sp.ndimage.gaussian_filter1d(high_third_acc, sigma = smoothing)
        
        subject_accuracies[i,0,:] = low_third_acc
        subject_accuracies[i,1,:] = mid_third_acc
        subject_accuracies[i,2,:] = high_third_acc
    elif nbins_behav == 2: #median split if this is the case
        median_behav = np.nanmedian(behvar)
        bottom_half = behvar <= median_behav
        top_half    = behvar > median_behav
        
        bottom_half_acc = np.zeros(ntimes) * np.nan
        top_half_acc    = np.zeros(ntimes) * np.nan
        
        for tp in range(ntimes):
            
        
        
    #get probabilities of the target class on a trial at each time point
    
    trl_target_proba = np.zeros(shape = (predict_probas.shape[0], ntimes)) * np.nan
            
    for trl in range(predict_probas.shape[0]):
        trl_target_proba[trl, :] = predict_probas[trl,predictor[trl],:]
    
    subject_trialwise_predictprobas[str(i)] = trl_target_proba
    subject_trialwise_predictprobas[str(i)+'_errors'] = bdata.absrdif.to_numpy()
    subject_trialwise_predictprobas[str(i)+'_conf']   = bdata.confwidth.to_numpy()
    
    subject_predictprobas[i,:] = trl_target_proba.mean(0) #assign average predict probability to matrix
    
    
    
#%%    
    

plotmean = np.mean(subject_accuracies, axis = 0)
plotsem  = sp.stats.sem(subject_accuracies, axis = 0)

labels = ['low', 'medium', 'high']
labels = [x + ' %s'%behav_varname for x in labels]

colors = ['#bdbdbd', '#4daf4a', '#377eb8']
colors = ['#4daf4a', '#bdbdbd', '#377eb8']

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(labels)):
    ax.plot(times,plotmean[i], lw = 1, label = labels[i], color = colors[i])
    ax.fill_between(times, plotmean[i]-plotsem[i], plotmean[i]+plotsem[i], color = colors[i], alpha = .1)
ax.set_ylabel('target side decoding accuracy')
ax.set_xlabel('time relative to array onset (s)')
ax.axhline(.5, lw = .5, ls = '--', color = 'k')
ax.set_xlim([-0.3,2.0])
fig.legend()

#%%

smooth_trialwise = True
if smooth_trialwise:
    for i in range(len(subs)):
        for trl in range(subject_trialwise_predictprobas[str(i)].shape[0]):
            subject_trialwise_predictprobas[str(i)][trl,:] = sp.ndimage.gaussian_filter1d(subject_trialwise_predictprobas[str(i)][trl,:], sigma = 5)


#look at the target class predict probabilities, these are unsmoothed tho
error_correlations = np.zeros((len(subs), ntimes)) * np.nan

for i in range(len(subs)):
    errcorr = np.zeros((ntimes)) * np.nan
    for tp in range(ntimes):
        errcorr[tp] = sp.stats.pearsonr(subject_trialwise_predictprobas[str(i)][:,tp],
                                        subject_trialwise_predictprobas[str(i)+'_errors'])[0]
    error_correlations[i] = errcorr

#plot unsmoothed average correlations

plt.figure()
plt.plot(times, error_correlations.mean(0).T)
plt.axhline(0, ls = '--', color = 'k', lw = .5)

#smooth and then plot
smoothed_errcorrs = deepcopy(error_correlations)
for i in range(subs.size):
    smoothed_errcorrs[i,:] = sp.ndimage.gaussian_filter1d(smoothed_errcorrs[i], sigma = 5)
    
plotmean = np.nanmean(smoothed_errcorrs, axis = 0)
plotsem  = sp.stats.sem(smoothed_errcorrs, axis = 0)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times, plotmean, lw = 1, color = '#377eb8')
ax.fill_between(times, plotmean-plotsem, plotmean+plotsem, color = '#377eb8', alpha = .1)
ax.axhline(0, ls = '--', lw = .5, color = '#000000')

plt.plot(times, smoothed_errcorrs.mean(0).T)
plt.axhline(0, ls = '--', color = 'k', lw = .5)
#

#%%

fig = plt.figure()
ax  = fig.add_subplot(111)
for smooth in [1,3,5,6]:
    ax.plot(times, sp.ndimage.gaussian_filter1d(subject_trialwise_predictprobas[str(0)][0], sigma = smooth), label = str(smooth))
ax.axhline(0.5, lw=.5,ls='--',color='k')
fig.legend()



    
    
    
    
    