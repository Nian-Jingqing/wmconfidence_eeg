#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:00:54 2020

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
from wmConfidence_funcs import gesd, plot_AR, nanzscore

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/glm')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm
from glmtools.regressors import CategoricalRegressor, ParametricRegressor
from glmtools.design import Contrast

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


def smooth(signal, twin = 10, method = 'boxcar'):
    #twin is number of samples (for 500Hz, 10 samples is 20ms)
    filt = sp.signal.windows.boxcar(twin)
    smoothed = np.convolve(filt/filt.sum(), signal, mode = 'same')
    return smoothed


subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
#%%

#import progressbar
#progressbar.streams.flush()

n_oris = 2 #2 items in the array
nbins  = 12 #orientation decoding uses 12 angular bins

srate = 500
ntimes = srate*2
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
    times = mne.epochs.read_epochs(fname = param['arraylocked'].replace('arraylocked', 'arraylocked_cleaned'), verbose=False).resample(srate).times
    nobs = bdata.shape[0]
    
    trl_target_proba = np.zeros(shape = (predict_probas.shape[0], n_oris, ntimes)) * np.nan
            
    for trl in range(nobs):
        for ori in range(n_oris):
            trl_target_proba[trl, ori,:] = predict_probas[trl,predictors[ori,trl],ori,:]
    trl_target_proba = np.mean(trl_target_proba, axis = 1) #average decoding across the two items of the array    
    
    smooth_type = 'gaussian'
    if smooth_type == 'boxcar':
        smoothing = 20 #smoothing window in ms
        for trl in range(nobs):
            trl_target_proba[trl] = smooth(trl_target_proba[trl], twin = int(smoothing/(1000/srate)))
    elif smooth_type == 'gaussian':
        smoothing = 4 #standard deviation of the gaussian filter
        trl_target_proba = sp.ndimage.gaussian_filter1d(trl_target_proba, sigma = smoothing)
    
    
#    tmin, tmax = -.3, 1
#    timeinds = [np.abs(times-tmin).argmin(), np.abs(times-tmax).argmin()]
#    times = times[timeinds[0]:timeinds[1]]
#    decoding = decoding[:, timeinds[0]:timeinds[1]]
    
    #one problem is that decoding is always above 0 for these things, because it's % of a binary classification
    #here, chance is 1/12, not 0
    #if we subtract chance, we 0 centre things so the resulting betas and tstats are informative around 0.
#    decoding = np.subtract(decoding, 1/nbins)
    
    
    
    #lets construct the glm now
    glmdata     = glm.data.TrialGLMData(data = trl_target_proba, time_dim = 1, sample_rate = 500)
    
    nobs        = glmdata.num_observations
    alltrials   = np.ones(nobs)
    
    error       = bdata.absrdif.to_numpy()
    confwidth   = bdata.confwidth.to_numpy()
    conf        = np.radians(np.multiply(confwidth, -1))
    error       = sp.stats.zscore(error)
    conf        = sp.stats.zscore(conf)
    
    regressors = list()
    regressors.append(ParametricRegressor(name = 'grandmean',  values = alltrials, preproc = None, num_observations = nobs))
    regressors.append(ParametricRegressor(name = 'error',      values = error,     preproc = None, num_observations = nobs))
#    regressors.append(ParametricRegressor(name = 'confidence', values = conf,      preproc = None, num_observations = nobs))
    
    contrasts = list()
    contrasts.append(Contrast([1, 0], 'grandmean'))
    contrasts.append(Contrast([0, 1], 'error'))
#    contrasts.append(Contrast([0, 0, 1], 'confidence'))
    
    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
#    glmdes.plot_summary()


    print('\nrunning glm\n')
    model = glm.fit.OLSModel( glmdes, glmdata)
    
    
    for iname in range(len(glmdes.contrast_names)):
        name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name
        
        beta = np.squeeze(model.copes[iname,:])
        plotfig = False
        if plotfig:
            plt.figure()
            plt.plot(times, beta, label = name)
#            plt.plot(times, trl_target_proba.mean(0).T, label = 'average decoding')
            plt.legend()
            if iname == 0:
                plt.axhline(1/nbins, ls = 'dashed', color = '#000000', lw = .75)
            else:
                plt.axhline(0, ls = 'dashed', color = '#000000', lw = .75)
                                
    
        tstat = np.squeeze(model.get_tstats()[iname,:])
        plott = False
        if plott:
            plt.figure()
            plt.plot(times, tstat, label = 'tstat for '+name)
            plt.legend()
            plt.axhline(0, ls = 'dashed', color = '#000000', lw = .75)
                        
        #now we'll just save out these numpy arrays so we can load in all subjects later on
        np.save(op.join(wd, 'data/decoding/arraylocked/glm2', 's%02d_orientation_decodingglm_%s_beta.npy'%(isub, name)), beta)
        np.save(op.join(wd, 'data/decoding/arraylocked/glm2', 's%02d_orientation_decodingglm_%s_tstat.npy'%(isub, name)), tstat)
        
        if i == 0:
            #just once, output the times we're looking at
            np.save(op.join(wd, 'data/decoding/arraylocked/glm2/decoding_data_times.npy'), times)

#%%