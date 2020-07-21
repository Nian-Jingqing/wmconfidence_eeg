#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:43:02 2019

@author: sammirc
"""


import numpy as np
import scipy as sp
import pandas as pd
import mne
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy import stats
from scipy import ndimage


sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, runclustertest_epochs

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

figpath = op.join(wd, 'figures', 'eeg_figs', 'feedbacklocked', 'epochs_glm4')

subs  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
subs  = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm
nsubs = subs.size
data = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    epochs = mne.read_epochs(fname = param['fblocked'], preload = True) #this is loaded in with the metadata
    epochs.set_eeg_reference(['RM'])
    epochs.drop_channels(['RM']) #referenced to the right mastoid already for symmetry
    epochs.set_eeg_reference(ref_channels='average') #re-reference to the common average now
    epochs.apply_baseline((-.25, 0)) #baseline 250ms prior to feedback
    epochs.resample(500) #resample to 500Hz
    ntrials = len(epochs)
    
    #before we reject trials, lets just get a couple of things into the metadata that we'll use in the glm
    bdata = epochs.metadata
    bdata['nxttrlcw'] = bdata.shift(-1).confwidth #get the confidence width of the next trial
    bdata['nxttrlcwadj'] = bdata.nxttrlcw - bdata.confwidth
    epochs.metadata = bdata
    
    
    #will do an automated process of looking for trials with heightened variance (noise) and output which trials to keep
    _, keeps = plot_AR(epochs, method = 'gesd', zthreshold = 1.5, p_out=.1, alpha = .05, outlier_side = 1)
    plt.close()
    keeps = keeps.flatten()

    discards = np.ones(len(epochs), dtype = 'bool')
    discards[keeps] = False
    epochs = epochs.drop(discards) #first we'll drop trials with excessive noise in the EEG
    
    epochs = epochs['DTcheck == 0 and clickresp == 1 and trialnum != 256'] #the last trial of the session doesn't have a following trial!
    print('a total of %d trials have been dropped for this subjects'%(ntrials-len(epochs)))
    
    data.append(epochs)
#%%
alltrls = []
incorr = []
corr   = []
for i in range(subs.size):
    incorr.append(deepcopy(data[i])['confdiff > 0'])
    corr.append(deepcopy(data[i])['confdiff <= 0'])
    alltrls.append(deepcopy(data[i].average()))

for i in range(subs.size):
    incorr[i] = incorr[i].average()
    corr[i]   = corr[i].average()
    
ivsc = []
for i in range(subs.size):
    ivsc.append(mne.combine_evoked([incorr[i], -corr[i]], weights = 'equal'))
    
mne.viz.plot_compare_evokeds(
        evokeds = dict(
                incorr = incorr,
                corr = corr,
                difference = ivsc),
        colors = dict(
                incorr = '#252525',
                corr = '#4292c6',
                difference = '#2ca25f'),
        show_legend = 'upper right', picks = ['FCZ', 'C1', 'C2', 'CZ', 'CPZ'],# combine='mean',
        ci = .68, show_sensors = False,
        truncate_xaxis = False,
        )
plt.title('grand mean and difference wave between incorrect and incorrect trials at FCz')
plt.axvline(ls='dashed', x = 0.025, ymin = -3, ymax = 16)


incorrvscorr = np.empty(shape = (subs.size, ivsc[0].times.size))
for i in range(subs.size):
    incorrvscorr[i,:] = np.squeeze(np.nanmean(deepcopy(ivsc[i]).pick_channels(['FCZ', 'C1', 'C2', 'CZ', 'CPZ']).data,0))

groupt_ivsc = sp.stats.ttest_1samp(incorrvscorr, popmean=0,axis=0)[0]
t, clu, clu_pv, _ = mne.stats.permutation_cluster_1samp_test(incorrvscorr)
masks = np.asarray(clu)[clu_pv <0.05]

mne.grand_average(ivsc).plot_joint(picks='eeg')


plotsub = True
isub = 2
if plotsub:
    deepcopy(ivsc[isub]).plot_joint(ts_args = dict(spatial_colors=True), picks = 'eeg')

#%%
    
from mne.decoding import (GeneralizingEstimator, SlidingEstimator, Vectorizer, TimeFrequency, cross_val_multiscore,
                          UnsupervisedSpatialFilter, Scaler, LinearModel, get_coef)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import sklearn as skl

est = make_pipeline(StandardScaler(),
                    LinearModel(LogisticRegression(class_weight = 'balanced', solver = 'lbfgs')))
sl = SlidingEstimator(est, scoring = 'roc_auc')

def get_patterns(epochs):
    epochs.set_eeg_reference(ref_channels='average')
    sl.fit(epochs.get_data(), epochs.metadata.confdiff.to_numpy() <= 0)
    coef = mne.decoding.get_coef(sl, 'patterns_', inverse_transform=False)
    return mne.EvokedArray(-coef, epochs.info, tmin=epochs.times[0])

allpatterns = []
for ii, epochs in enumerate(data):
    print(ii, end = ',')
    allpatterns.append(get_patterns(epochs))

scaling_dict = dict(scalings = dict(eeg=1e-6))
fig = mne.grand_average(allpatterns).plot_joint(show=False,ts_args = scaling_dict, topomap_args=scaling_dict, times=[.15, .2, .25, .3])


#%%

#lets take one subject to start with

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def matrix_vector_shift(matrix, vector, n_bins):
    r,c = matrix.shape
    matrix_shift = np.zeros((r,c))
    for row in range(0,r):
        matrix_shift[row,:] = np.roll(matrix[row,:], round(n_bins/2) - vector[row])
    return matrix_shift

ds = deepcopy(data[-1])
ds = ds.drop_channels(['VEOG', 'HEOG']) #data here is trials x channels x time
[n_trials, n_chans, n_time] = ds.get_data().shape
n_bins = 12
size_window = 5
n_folds = 5
classifier = 'LDA'
pca = False
temporal_dynamics = False
time = ds.times

prediction = np.zeros([n_trials, n_bins, n_time]) * np.nan
label_pred = np.zeros([n_trials, n_time]) *np.nan
centred_prediction = np.zeros(([n_bins, n_time])) * np.nan
accuracy = np.zeros(n_time) *np.nan

X_all = deepcopy(ds).get_data()
y = ds.metadata.confwidth.to_numpy()

pred = np.zeros((n_trials,n_time)) *np.nan
score = np.zeros(n_time)*np.nan
coeffs = np.zeros((n_chans, n_time)) * np.nan

X_demeaned = np.zeros((n_trials, n_chans, size_window)) * np.nan

for tp in range((size_window-1),n_time):
    if temporal_dynamics == True:
        for s in range((1-size_window),1):
            X_demeaned[:,:,s] = X_all[:,:,tp+s] = X_all[:,:,(tp-(size_window-1)):(tp+1)].mean(2)
        X = X_demeaned.reshape(X_demeaned.shape[0], X_demeaned.shape[1]*X_demeaned.shape[2])
        
    if pca == True:
        pca = skl.decomposition.PCA(n_components = 0.90)
        X = pca.fit(X).transform(X)
    
    if not pca and not temporal_dynamics:
        X = X_all[:,:,tp]
        
    #train test set
    rskf = RepeatedStratifiedKFold(n_splits = n_folds, n_repeats = 1, random_state = 42)
    for train_index, test_index in rskf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #standardisation
        scaler = StandardScaler().fit(X_train)
        X_train =  scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        
        #define classifier
        if classifier == 'LDA':
            clf = LinearDiscriminantAnalysis()
        if classifier == 'Ridge':
            clf = skl.linear_model.Ridge(alpha=.8)
        
        #train
        clf.fit(X_train, y_train)
        
        #test
#        prediction[test_index,:,tp] = clf.predict_proba(X_test)
        label_pred[test_index, tp] = clf.predict(X_test)
        score[tp] = clf.score(X_train,y_train)
        coeffs[:,tp] = clf.coef_

fig = plt.figure()
ax =  fig.add_subplot(111)
ax.plot(ds.times,score, lw = 1.5)
ax.hlines([0], xmin=-0.5, xmax=1.5, linestyles = 'dashed', lw = .75)
    

        
        
#%%
ds = deepcopy(data[-1])
ds = ds.drop_channels(['VEOG', 'HEOG']) #data here is trials x channels x time
time = ds.times
X_all = deepcopy(ds).get_data()
y = ds.metadata.confwidth.to_numpy()

        
clf = skl.linear_model.Ridge(alpha=.5)        
scaler = StandardScaler()
model = mne.decoding.LinearModel(clf)

labels = ds.events[:,-1]
x_data = ds.get_data().reshape(len(labels),-1)

x = scaler.fit_transform(x_data)

model.fit(x, y)

for name, coef in (('patterns', model.patterns_), ('filters', model.filters_)):
    coef = scaler.inverse_transform([coef])[0]
    coef = coef.reshape(len(ds.ch_names), -1)
    evoked = mne.EvokedArray(coef, ds.info, tmin = ds.tmin)
    evoked.plot_topomap(title = 'EEG %s' %name, time_unit = 's')

#%%
ds = deepcopy(data[-2])
ds = ds.drop_channels(['VEOG', 'HEOG']) #data here is trials x channels x time
time = ds.times
X_all = deepcopy(ds).get_data()

var = 'targinconf' #or pside if binary
continuous = None
if var == 'confwidth' or var == 'error':
    continuous = True
    if var == 'confwidth':
        y =  ds.metadata.confwidth.to_numpy()
    elif var == 'error':
        y = ds.metadata.absrdif.to_numpy()
elif var == 'pside':
    continuous = False
    y = ds.metadata.pside.to_numpy()
elif var == 'targinconf': #looks like this is also decodable
    continuous = False
    y = ds.metadata.confdiff.to_numpy()
    y = np.where(y<=0, 1, 0)

nbins = 12        
if continuous:
    #here we need to bin the data
    bins = np.arange(nbins) 
    ybinned = np.digitize(y, bins)
plotbinnum = False
if plotbinnum:
    for i in np.unique(ybinned):
        print(i, np.array(ybinned==i).sum())
        
#for tp in range(ds.times.size):
#    tpdat = X_all[:,:,tp]

if not continuous: #if just decoding binary probed side
    clf = skl.pipeline.make_pipeline(StandardScaler(),
                                     LogisticRegression(solver='lbfgs'))
#    nsplits = 5
#    cv = skl.model_selection.StratifiedKFold(n_splits=nsplits, shuffle=True)
    
    td = mne.decoding.SlidingEstimator(clf, scoring = 'roc_auc')
    
    td.fit(X_all, y)
    td.score(X_all, y)
    scores = mne.decoding.cross_val_multiscore(td, X_all, y, cv=5)
    
    fig,ax = plt.subplots()
    ax.plot(ds.times, np.mean(scores,0), label = 'score')
    ax.axhline(.5, color = 'k', ls = '--', label = 'chance')
    ax.set_ylim([0.4,1])
















