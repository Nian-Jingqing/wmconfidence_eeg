#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:12:59 2020

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


subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
#%%

import progressbar
progressbar.streams.flush()


nconds = 2 #2 conditions, neutral vs cued; if orientations at encoding, this is still 2 (left and right orientations)

srate = 500
if srate == 500:
    ntimes = 1000 #specify the number of timepoints we're going to be looking at
elif srate == 250:
    ntimes = 500
elif srate == 200:
    ntimes = 400
    
ntimes = srate*2
subject_accuracies = np.zeros((len(subs), nconds, ntimes)) * np.nan


for i in range(len(subs)):
    isub = subs[i]
    
    if op.exists(op.join(wd, 'data', 'decoding', 'arraylocked', 'orientation', 's%02d_orientationdecoding_preds_data.npy'%(isub))):
        print('decoding already run for subject %d'%isub)
    if not op.exists(op.join(wd, 'data', 'decoding', 'arraylocked', 'orientation', 's%02d_orientationdecoding_preds_data.npy'%(isub))):
        print('\n\nworking on subject ' + str(isub) + '  (%d/%d)\n\n'%(i+1, len(subs)))
        sub = dict(loc = 'workstation', id = isub)
        param = get_subject_info_wmConfidence(sub)
        
        epochs = mne.epochs.read_epochs(fname = param['arraylocked'].replace('arraylocked', 'arraylocked_cleaned'), preload=True) #read raw data
        epochs.set_channel_types(dict(RM='eeg'))
        if 'VEOG' in epochs.ch_names: #drop eog channels if present
            epochs.pick_types(eeg = True)
        epochs = mne.add_reference_channels(epochs, ref_channels = 'LM')
        epochs.set_eeg_reference(ref_channels = ['LM','RM']) #re-reference average of the left and right mastoid now
        
        epochs.apply_baseline((-.25, 0)) #baseline 250ms prior to array onset
        epochs.resample(srate) #resample 
        epochs.drop_channels(['RM', 'LM']) #drop these reference channels now, we don't need them
        
        
        ntrials = len(epochs)
        print('\n\n %d trials in the data\n\n'%ntrials)
        
        #can't easily equalise counts of the orientation categories because they arent event_ids...
        #epochs = epochs.equalize_event_counts(event_ids = ['neutral_left', 'neutral_right', 'cued_left', 'cued_right'])[0]
        
        #angles are between 0 and 179 degrees, so lets generate some 15 degree bins to categorise these orientations
        
        #set up some class boundaries as we're going to discretise orientation into equally sized bins and make it a classification problem
        #this lets us treat it more simply in the first instance, as it's easier to implement
        
        category_lowerbounds = np.array([-7, 8, 23, 38, 53, 68, 83, 98, 113, 128, 143, 158,]) #-7 is 173 degrees because of wrapping after 180
        category_upperbounds = np.add(category_lowerbounds, 15)
        
        cats = np.stack([category_lowerbounds, category_upperbounds]).T
        category_angles = np.zeros((cats.shape[0],15)) * np.nan
        for x in range(category_angles.shape[0]):
            category_angles[x,:] = np.arange(cats[x,0], cats[x,1])
        category_angles = category_angles.astype(int)
        category_angles = np.where(category_angles <0, category_angles+180, category_angles)
        
        df = pd.DataFrame()
        df['ori1'] = 0
        df['category_label1'] = 0
        #df['ori2'] = 0
        #df['category_label2'] = 0
        cat_labels = np.arange(12)
        count = 0
        for y in cat_labels:
            for x in category_angles[y,:]:
                df.loc[count,'ori1'] = x
                #df.loc[count, 'ori2'] = x
                df.loc[count,'category_label1'] = y
                #df.loc[count, 'category_label2'] = i
                count += 1
        df = df.astype(int)
        
        df2 = pd.merge(epochs.metadata, df, on = ['ori1']) #merge category labels for left orientation
        df = df.rename(index = str, columns = {'ori1':'ori2', 'category_label1':'category_label2'}) #rename so we can do the right orientation now
        
        df2 = pd.merge(df2, df, on = 'ori2') #merge to assign labels for the right hand side orientation
        epochs.metadata = df2 #reassign this, and we now have category labels for the left and right presented items
        
        
        
        ds = deepcopy(epochs)
        data = ds.get_data() #shape is trials x channels x time
        
        predictvar  = 'category_label1'
        predictvar2 = 'category_label2' 
        predictor   = ds.metadata[predictvar].to_numpy()
        predictor2  = ds.metadata[predictvar2].to_numpy()
        
            
            
        #    classifier = 'logistic'
        classifier = 'LDA'
        #classifier = 'Ridge'
        
        splittype = 'rskf' #can be rskf 
    #    splittype = 'leaveoneout'
        
        ntrials, nchannels, ntimes = data.shape
        nbins = np.unique(predictor).size #how many classes are you trying to decode? only two here, left or right
        
        prediction = np.zeros(([ntrials, nbins, ntimes])) * np.nan
        label_pred = np.zeros((ntrials, ntimes)) * np.nan
        coefs      = np.zeros([ntrials,nbins, nchannels, ntimes]) * np.nan
        
        
        decode_orientation = True
        n_oris = 2
        if decode_orientation:
            prediction = np.zeros(([ntrials, nbins, n_oris, ntimes])) * np.nan
            label_pred = np.zeros(([ntrials, n_oris, ntimes])) * np.nan
        
        
        
        #we need to reshape the data to channels x time, and demean each channel, then reconcatenate into the trial structure
        data = np.transpose(data, (1,0,2)) #now its channels x trials x times
        data = data.reshape((nchannels, ntrials*ntimes))
        data = np.subtract(data, data.mean(1).reshape(-1,1)) #demean
        
        data = data.reshape(nchannels, ntrials, ntimes)
        data = np.transpose(data, (1,0,2))
        
        print('\n\nrunning decoding across time points now\n\n')
    
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
                    y2_train, y2_test = predictor2[train_index], predictor2[test_index] #get training and test for second orientation
                    
                    if splittype =='leaveoneout':
                        x_test = x_test.reshape(1,-1) #just needed if leave one out so the scaler works
                    
                    scaler = StandardScaler().fit(x_train)
                    x_train = scaler.transform(x_train)
                    x_test  = scaler.transform(x_test)
                    
                    if classifier == 'logistic':
                        clf = LogisticRegression(random_state = 42, solver = 'lbfgs', multi_class ='ovr') #ovr cos only doing binary choice of classes
                    if classifier == 'LDA':
                        clf = LinearDiscriminantAnalysis(shrinkage='auto', solver = 'lsqr')
                        if decode_orientation:
                            clf2 = LinearDiscriminantAnalysis(shrinkage = 'auto', solver = 'lsqr')
            #        if classifier == 'Ridge':
            #            clf = skl.linear_model.RidgeClassifier()
                        
                    clf.fit(x_train, y_train)
                    clf2.fit(x_train, y2_train)
            
                    if decode_orientation:
                        prediction[test_index,:,0,tp] = clf.predict_proba(x_test)
                        prediction[test_index,:,1,tp] = clf2.predict_proba(x_test)
                        
                        label_pred[test_index,0,tp]   = clf.predict(x_test)
                        label_pred[test_index,1,tp]   = clf2.predict(x_test)
                    else:
                        prediction[test_index,:,tp] = clf.predict_proba(x_test) #probability of predicting the two classes on these trials, at this time point
                        label_pred[test_index,tp]   = clf.predict(x_test)       #the winning prediction for a trial, at this time point
                        coefs[test_index,:,:,tp] = np.squeeze(clf.coef_) #we have a prediction probability for each class per trial and time point, and a coef for each sensor here
        bar.finish()
        
        
        if decode_orientation:
            accuracy = np.zeros((ntimes, n_oris)) * np.nan #accuracy can only be computed across trials, and results in an average
            
            preds = np.stack([predictor, predictor2])
            for ori in range(n_oris):
                for time in range(ntimes):
                    accuracy[time,ori] = skl.metrics.accuracy_score(preds[ori], label_pred[:,ori,time])    
        
        #write out the accuracy across trials for each subject into a new file
        #we need this to be able to then take it and look at its relation to behaviour
        if decode_orientation:
            np.save(op.join(wd, 'data', 'decoding', 'arraylocked', 'orientation', 's%02d_orientationdecoding_preds_data.npy'%(isub)),preds)
            np.save(op.join(wd, 'data', 'decoding', 'arraylocked', 'orientation', 's%02d_orientationdecoding_labelpred_data.npy'%(isub)),label_pred)
            np.save(op.join(wd, 'data', 'decoding', 'arraylocked', 'orientation', 's%02d_orientationdecoding_predictproba_data.npy'%(isub)),prediction)
    
    #    if decode_orientation:
    #        accuracy = np.nanmean(accuracy, 1) #average across decoding of the two items in the array
    
    #    subject_accuracies[i,:,:] = np.nanmean(accuracy, 0)
#%%
    
def smooth(signal, twin = 10, method = 'boxcar'):
    #twin is number of samples (for 500Hz, 10 samples is 20ms)
    filt = sp.signal.windows.boxcar(twin)
    smoothed = np.convolve(filt/filt.sum(), signal, mode = 'same')
    return smoothed

#if the orientation decoding has already been run and you want to look at binned analyses, use this.
#there's a separate script for running glms based on the trialwise decoding

from scipy import ndimage

nconds = 2 #2 conditions, neutral vs cued; if orientations at encoding, this is still 2 (left and right orientations)
nbins  = 12 #orientation decoding uses 12 angular bins
n_oris = 2
srate  = 500
ntimes = srate*2

nbins_behav        = 3
subject_accuracies = np.zeros(shape = (subs.size, nbins_behav, ntimes)) * np.nan

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

    
    #get the predict probability for one trial for the relevant class
#    tnum = 0
#    iori = 0
#    tmp_predictproba = predict_probas[tnum, predictors[tnum,iori],iori,:]
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(times, tmp_predictproba,label ='predict probability for class', lw = .75)
#    ax.plot(times, sp.ndimage.gaussian_filter1d(tmp_predictproba,sigma=3), label='gsmooth sigma = 3', lw = .75)
#    ax.plot(times, sp.ndimage.gaussian_filter1d(tmp_predictproba,sigma=5), label='gsmooth sigma = 5', lw = .75)
#    ax.axhline(1/12, color = '#bdbdbd', lw =.5, ls='--')
#    fig.legend()
#    
#    plt.close('all')
    
    #so if we want to use the trialwise prediction probabilities (decoding certainties) we can smooth trialwise
    
    
    #if we want to look at accuracy of decoding in different groups of trials, skl.metrics.accuracy_score does it across trials, so we would smooth the resulting trace
    #because we don't get trialwise accuracy i guess (not relative to chance at least)
    
    #so get confidence for these trials and we can do some binning
    confidence = bdata.confwidth.to_numpy()
    confidence = np.multiply(confidence, -1) #flip confidence so higher values now indicate higher confidence
    if nbins_behav == 3:
        #split into thirds, so we need the 33rd and 66th percentiles
        percentiles_behav = np.quantile(confidence, [.33, .66])
        #get indices for relevant trials
        bottom_third = confidence <= percentiles_behav[0]
        middle_third = np.logical_and(confidence > percentiles_behav[0], confidence <= percentiles_behav[1])
        top_third    = confidence > percentiles_behav[1]
        
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
            
        
#        fig = plt.figure(figsize = (12,12))
#        count = 0
#        for smoothing in [0,3,5]:
#            count +=1
#            if smoothing == 0:
#                ax = fig.add_subplot(3,1,count)
#                ax.set_title('decoding accuracy - unsmoothed')
#                ax.plot(times, low_conf_acc, label = 'low confidence', color = '#386cb0', lw = 1)
#                ax.plot(times, low_conf_acc, label = 'mid confidence', color = '#beaed4', lw = 1)
#                ax.plot(times, low_conf_acc, label = 'high confidence', color = '#7fc97f', lw = 1)
#                ax.set_xlabel('time relative to array onset (s)')
#                ax.set_ylabel('decoding accuracy (AU)')
#                ax.axhline(1/12, lw = .5, ls = '--', color = '#bdbdbd')
#            else:
#                ax = fig.add_subplot(3,1,count)
#                ax.set_title('decoding accuracy - smoothed (sigma = %d)'%smoothing)
#                ax.plot(times, sp.ndimage.gaussian_filter1d(low_conf_acc, sigma = smoothing), label = 'low confidence', color = '#386cb0', lw = 1)
#                ax.plot(times, sp.ndimage.gaussian_filter1d(mid_conf_acc, sigma = smoothing), label = 'mid confidence', color = '#beaed4', lw = 1)
#                ax.plot(times, sp.ndimage.gaussian_filter1d(high_conf_acc, sigma = smoothing), label = 'high confidence', color = '#7fc97f', lw = 1)
#                ax.set_xlabel('time relative to array onset (s)')
#                ax.set_ylabel('decoding accuracy (AU)')
#                ax.axhline(1/12, lw = .5, ls = '--', color = '#bdbdbd')
#           fig.legend()
        
        subject_accuracies[i,0,:] = low_conf_acc
        subject_accuracies[i,1,:] = mid_conf_acc
        subject_accuracies[i,2,:] = high_conf_acc

#%%
    

#plot these bins now
        
plotmean  = np.mean(subject_accuracies,axis = 0)
plotsem   = sp.stats.sem(subject_accuracies, axis = 0)

labels = ['low confidence', 'middle confidence', 'high confidence']
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










#%%
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, runclustertest_epochs




for twin in [2,4, 5, 7, 10]:
    subaccs_smoothed = np.zeros((subject_accuracies.shape)) * np.nan
    
    for i in range(len(subs)):
        for x in range(n_oris):
            subaccs_smoothed[i,x,:] = smooth(subject_accuracies[i,x,:], twin=twin)
            #subaccs_smoothed[i,x,:] = smooth(subject_accuracies[i,x,:], twin = 7)
    
    subaccs_aveitems_smoothed = np.nanmean(subaccs_smoothed, 1)
    
    plotmean = np.nanmean(subaccs_aveitems_smoothed, 0)
    plotsem  = sp.stats.sem(subaccs_aveitems_smoothed, 0)
    
    
    fig = plt.figure(figsize = (12,6))
    ax = fig.add_subplot(111)
    ax.plot(ds.times, plotmean, color = '#636363', label = 'orientation decoding at encoding')
    ax.fill_between(ds.times, plotmean-plotsem, plotmean+plotsem, color = '#bdbdbd', alpha = .3)
    ax.set_ylim([0.07, None])
    ax.set_xlim([-0.3, 1.2])
    ax.hlines(1/12,xmin=-0.3, xmax=1.2, linestyles = 'dashed', color = '#000000', lw = .75)
    ax.set_ylabel('decoding accuracy (AU)')
    ax.set_xlabel('time relative to array onset (s)')
    ax.set_title('smoothing: %dms'%(twin*1000/srate))
    ax.legend()
    


#for cluster test we need to check difference from chance, not difference from 0
#to do this, we will subtract chance from the time series so that it zeros at chance

subaccs_smoothed = np.zeros((subject_accuracies.shape)) * np.nan

smoothing = 20 #smoothing in ms
for i in range(len(subs)):
    for x in range(n_oris):
        subaccs_smoothed[i,x,:] = smooth(subject_accuracies[i,x,:], twin=int(smoothing/(1000/srate)))
        #subaccs_smoothed[i,x,:] = smooth(subject_accuracies[i,x,:], twin = 7)

subaccs_aveitems_smoothed = np.nanmean(subaccs_smoothed, 1) #average decoding across items in the array

cludat = deepcopy(subaccs_aveitems_smoothed)


checkplot=True
if checkplot:
    plt.figure()
    plt.plot(ds.times, cludat.mean(0).T)
    plt.xlim([-0.3,1])
    plt.hlines(nbins**-1, xmin=-0.3,xmax=1)

cludat = np.subtract(cludat, 1/12)  #subtract chance level so that this is zero centred for the permutation testing
cludat = cludat.reshape((len(subs), 1, -1))


tmin, tmax = -.3, 1
timeinds = [np.abs(ds.times-tmin).argmin(), np.abs(ds.times-tmax).argmin()]

clustering_times = ds.times[timeinds[0]:timeinds[1]]

cludat = deepcopy(cludat)[:,:,timeinds[0]:timeinds[1]] #get down to just the timepoints we want to consider



t, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(cludat,
                                                                       out_type='indices',
                                                                       n_permutations = 5000)
masks = np.asarray(clusters)[cluster_pv<= 0.05]


plotmean = np.mean(subaccs_aveitems_smoothed,0)
plotsem  = sp.stats.sem(subaccs_aveitems_smoothed,0)

fig = plt.figure(figsize = (12,6))
ax = fig.add_subplot(111)
ax.plot(ds.times, plotmean, color = '#636363')
ax.fill_between(ds.times, plotmean-plotsem, plotmean+plotsem, color = '#636363', alpha = .3)
ax.set_xlim([-0.3, 1])
ax.set_ylim([0.07, None])
ax.hlines(nbins**-1, xmin=-0.3, xmax=1, linestyles='dashed', color = '#000000', lw = .75)
ax.vlines(0, ymin=0.075, ymax = 0.095, linestyles = 'dashed', color = '#000000', lw = .75)
ax.set_xlabel('Time relative to array onset (s)')
ax.set_ylabel('orientation decoding accuracy (AU)')
for mask in masks:
        ax.hlines(y = 0.075,
                  xmin = np.min(clustering_times[mask[1]]),
                  xmax = np.max(clustering_times[mask[1]]),
                  lw=5, color = '#636363', alpha = .5) #plot significance timepoints for difference effect


#%%
