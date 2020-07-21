#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:35:55 2020

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



#stuff for the machine learning

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from mne.decoding import (GeneralizingEstimator, SlidingEstimator, Vectorizer, TimeFrequency, cross_val_multiscore,
                          UnsupervisedSpatialFilter, Scaler, LinearModel, get_coef)
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
import progressbar
progressbar.streams.flush()

#set verbosity for all mne functions to only be if there is an error, just stops it being cluttered in the console
mne.set_log_level(verbose = 'Error') #only prints things to console if there's an error. only use this if you are sure it works

tmin, tmax = -0.5, 2.0
nconds = 2 #2 conditions, neutral vs cued
srate = 500
ntimes = np.arange(tmin, tmax, 1/srate).size
subject_accuracies = np.zeros((len(subs), nconds, ntimes)) * np.nan
coefficients = dict()

for i in range(len(subs)):
    isub = subs[i]
    
    print('\nworking on subject ' + str(isub))
    sub = dict(loc = 'workstation', id = isub)
    param = get_subject_info_wmConfidence(sub)
    
    epochs = mne.epochs.read_epochs(fname = param['cuelocked'].replace('cuelocked', 'cuelocked_cleaned'), verbose=False,preload = True)
    epochs.crop(tmin = -0.5, tmax = 2)
    epochs = mne.add_reference_channels(epochs, ref_channels = 'LM') #adds a channel of zeros for the left mastoid (as ref to itself in theory)
    #    deepcopy(epochs['cued_left']).apply_baseline((-.25,0)).average().plot_joint(title = 'cued left, no re-ref', times = np.arange(.1,.7,.1))
    #    deepcopy(epochs['cued_left']).set_eeg_reference(['RM']).apply_baseline((-.25,0)).average().plot_joint(title='cued_left, RM re-ref', times = np.arange(.1,.7,.1))
    #    deepcopy(epochs['cued_left']).set_eeg_reference(['LM', 'RM']).apply_baseline((-.25,0)).average().plot_joint(title = 'cued left, ave mast ref', times = np.arange(.1,.7,.1))
        
    epochs.set_eeg_reference(ref_channels = ['LM','RM'], verbose = False) #re-reference average of the left and right mastoid now
    #    epochs.set_eeg_reference('average')
    epochs.apply_baseline((-.25, 0)) #baseline 250ms prior to feedback
    epochs.resample(srate) #resample to 500Hz
    if 'VEOG' in epochs.ch_names:
        epochs = epochs.drop_channels(['VEOG', 'HEOG'])
    
    #subset only cued trials    
    epochs = epochs[['cued_left','cued_right']]
    ntrials = len(epochs)
    
    #print('%d trials in the data\n'%ntrials)
    
    epochs = epochs.equalize_event_counts(event_ids = ['cued_left', 'cued_right'])[0]
    ds = deepcopy(epochs)
    data = ds.get_data() #shape is trials x channels x time
    
    predictvar = 'pside'
    if predictvar == 'pside':
        predictor = ds.metadata.pside.to_numpy()
        
#    classifier = 'logistic'
    classifier = 'LDA'
    #classifier = 'Ridge'
    
    splittype = 'rskf' #can be rskf 
    #splittype = 'leaveoneout'
    
    ntrials, nchannels, ntimes = data.shape
    nbins = 2 #how many classes are you trying to decode? only two here, left or right
    
    prediction = np.zeros(([ntrials, nbins, ntimes])) * np.nan
    label_pred = np.zeros((ntrials, ntimes)) * np.nan
    coefs      = np.zeros([ntrials, nchannels, ntimes]) * np.nan
    
    
    
    #we need to reshape the data to channels x time, and demean each channel, then reconcatenate into the trial structure
    data = np.transpose(data, (1,0,2)) #now its channels x trials x times
    data = data.reshape((nchannels, ntrials*ntimes))
    data = np.subtract(data, data.mean(1).reshape(-1,1)) #demean
    
    data = data.reshape(nchannels, ntrials, ntimes)
    data = np.transpose(data, (1,0,2))
    
#    print('running decoding across time points now')
    progressbar.streams.wrap_stderr()
    bar = progressbar.ProgressBar(max_value = ntimes).start()
    for tp in range(ntimes): #loop over timepoints
        bar.update(tp)
        
        dat = data[:,:,tp] #get data for this time point
        
        if splittype == 'rskf':
            cv = RepeatedStratifiedKFold(n_splits =10, n_repeats = 1, random_state = 4)
        elif splittype == 'leaveoneout':
            cv  = skl.model_selection.LeaveOneOut()
            
        #run the iteration over train/test splits
        for train_index, test_index in cv.split(data, predictor):
            x_train, x_test = dat[train_index, :], dat[test_index, :] #get training and test data
            y_train, y_test = predictor[train_index], predictor[test_index] #get training and test predict data
    
            
            if splittype =='leaveoneout':
                x_test = x_test.reshape(1,-1) #just needed if leave one out so the scaler works
            
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test  = scaler.transform(x_test)
            
            if classifier == 'logistic':
                clf = LogisticRegression(random_state = 42, solver = 'lbfgs', multi_class ='ovr') #ovr cos only doing binary choice of classes
            if classifier == 'LDA':
                clf = LinearDiscriminantAnalysis(shrinkage='auto', solver = 'lsqr')
    #        if classifier == 'Ridge':
    #            clf = skl.linear_model.RidgeClassifier()
                
            clf.fit(x_train, y_train)
    
            prediction[test_index,:,tp] = clf.predict_proba(x_test) #probability of predicting the two classes on these trials, at this time point
            label_pred[test_index,tp]   = clf.predict(x_test)       #the winning prediction for a trial, at this time point
            coefs[test_index,:,tp] = np.squeeze(clf.coef_)
    bar.finish()
#    nconds = 2 #neutral and cued
#    conds = np.arange(nconds)
#    cuecond = ds.metadata.cue.to_numpy()
#    accuracy = np.zeros((nconds, ntimes)) * np.nan
#
#    #split accuracy by confidence!
#    #for low confidence
#    for tp in range(ntimes):
#        accuracy[0,tp]  = skl.metrics.accuracy_score(predictor[lowconf], label_pred[lowconf,tp])
#        accuracy[1,tp] = skl.metrics.accuracy_score(predictor[highconf], label_pred[highconf,tp]) 

    #save out these files per subject
    np.save(op.join(wd, 'data', 'decoding', 'cuelocked', 'target_side', 's%02d_sidedecoding_predictprobas_data.npy'%(isub)), prediction)
    np.save(op.join(wd, 'data', 'decoding', 'cuelocked', 'target_side', 's%02d_sidedecoding_labelpreds_data.npy'%(isub)), label_pred)
    np.save(op.join(wd, 'data', 'decoding', 'cuelocked', 'target_side', 's%02d_sidedecoding_predictors_data.npy'%(isub)), predictor)
    epochs.metadata.to_csv(op.join(wd,'data','decoding','cuelocked','target_side','s%02d_sidedecoding_bdata.csv'%(isub)))
    
    if i == 0:
        np.save(op.join(wd, 'data', 'decoding', 'cuelocked', 'target_side', 'times_sidedecoding.npy'), ds.times)
#    for x in conds:
#        for tp in range(ntimes):
#            accuracy[x,tp] = skl.metrics.accuracy_score(predictor[cuecond==x], label_pred[cuecond==x,tp])
#    coefficients[str(i)] = coefs #add sensor weights for each trial and time point into dictionary. each key is an individual

#    subject_accuracies[i, :, :] = accuracy

#%%

plt.figure()
for i in range(len(subs)):
    plt.plot(ds.times, subject_accuracies[i,1,:])
plt.title('neutral trial decoding accuracies for all subjects')
plt.ylim([0.3, 0.7])
plt.xlim([-0.1, 1.5])
plt.axhline(0.5, lw = .75, ls = 'dashed', color = 'k')


plt.figure()
#plt.plot(ds.times, np.mean(subject_accuracies,0)[0,:], lw = 1, label = 'gave decoding accuracy - neutral')
plt.plot(ds.times, np.mean(subject_accuracies,0)[1,:], lw = 1, label = 'gave decoding accuracy - cued')
plt.ylim([0.4, 0.6])
plt.xlim([-0.1, 1.5])
plt.axhline(0.5, lw = .75, ls = 'dashed', color = 'k')





def smooth(signal, twin = 10, method = 'boxcar'):
    #twin is number of samples (for 500Hz, 10 samples is 20ms)
    filt = sp.signal.windows.boxcar(twin)
    smoothed = np.convolve(filt/filt.sum(), signal, mode = 'same')
    return smoothed

decoding = deepcopy(subject_accuracies)
times = ds.times
smoothing = 20 #smoothing in ms
for i in range(decoding.shape[0]):
    for cond in range(decoding.shape[1]):
        decoding[i,cond,:] = smooth(decoding[i,cond,:], twin = int(smoothing/(1000/srate)))

plotmean = np.mean(decoding,0)
plotsem  = sp.stats.sem(decoding,0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times, plotmean[0], lw = 1, color = '#636363', label = 'decoding accuracy - low confidence')
ax.fill_between(times, plotmean[0]-plotsem[0], plotmean[0]+plotsem[0], alpha = .3, color = '#636363')

ax.plot(times, plotmean[1], lw = 1, color = '#3182bd', label = 'decoding accuracy - high confidence')
ax.fill_between(times, plotmean[1]-plotsem[1], plotmean[1]+plotsem[1], alpha = .3, color = '#3182bd')
fig.legend()
ax.set_xlim([-0.3,1])
ax.set_ylim([0.45,0.6])                
                

####
#### note that until the gaussian smoothing (or other filtering) is sorted, boxcar is actually better (and you know what it is doing)
#### so just use the single smooth function
####
def gsmooth(signal, stddev = 10, method = 'hanning'):
    #signal to smooth
    #std deviations of the gaussian to use in the window)
    
    
    if method == 'hanning':
        filt = np.hanning(stddev) #constructs hanning window of width ur std dev (ms of smoothing you want)
        filt = filt / filt.sum() #normalise the window area
    
    if method == 'gaussian':
        filt = sp.signal.gaussian(100, stddev)
        filt = filt / filt.sum()
        
    smoothed = np.convolve(signal, filt, mode = 'same')
    return smoothed


plotmean = np.mean(decoding,0)
plotsem  = sp.stats.sem(decoding,0)
times    = ds.times

fig = plt.figure(figsize = [12,6])
ax = fig.add_subplot(111)
ax.plot(times, plotmean, lw = 1, color = '#3182bd', label = 'mean cuelocked target side decoding')
ax.fill_between(times, plotmean-plotsem, plotmean+plotsem, alpha = .3, color = '#3182bd')
ax.set_xlim([-0.3,1.4])
ax.set_ylim([0.45, 0.6])
ax.hlines(0.5, xmin=-0.3,xmax=1.5, lw = .75, color = '#000000', linestyles = 'dashed')
ax.vlines(0, ymin=0.47, ymax=.57, color ='#000000', linestyles ='dashed', lw = .75)
ax.set_xlabel('time from cue onset (s)')
ax.set_ylabel('target side decoding accuracy')
fig.suptitle('decoding of target side across subjects in cued trials')
          
fig.savefig('/home/sammirc/Desktop/DPhil/wmConfidence/figures/eeg_figs/cuelocked/decoding/cuedside_decoding_cuedonly.eps', dpi=300, frameon = None)
fig.savefig('/home/sammirc/Desktop/DPhil/wmConfidence/figures/eeg_figs/cuelocked/decoding/cuedside_decoding_cuedonly.pdf', dpi = 300, frameon = None)


#%%
