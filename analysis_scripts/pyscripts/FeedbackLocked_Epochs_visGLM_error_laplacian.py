#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:48:43 2020

@author: sammirc
"""
import numpy as np
import scipy as sp
from scipy import signal
import pandas as pd
import mne
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy import stats

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, smooth, runclustertest_epochs

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)
figpath = op.join(wd,'figures', 'eeg_figs', 'fblocked', 'epochs_glm3')


subs = np.array([4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22,24, 25, 26])

contrasts = ['correct', 'incorrect', 'errorcorrect', 'errorincorrect', 'confcorrect', 'confincorrect', 'pside',
             'incorrvscorr', 'errorincorrvscorr', 'confincorrvscorr',
             'grandmean', 'error', 'conf']

laplacian = True
if laplacian:
    lapstr = 'laplacian_'
else:
    lapstr = ''


data = dict()
data_t = dict()
for i in contrasts:
    data[i] = []
    data_t[i] = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    for name in contrasts:
        data[name].append(   mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_' + lapstr + name + '_betas-ave.fif'))[0])        
        data_t[name].append( mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_' + lapstr + name + '_tstats-ave.fif'))[0])        

#%%
gave = mne.grand_average(data['grandmean']); times = gave.times;
#%%

#firstly, just look at the scalp topographies (joint plot) for the evoked response in the relevant conditions
mne.grand_average(deepcopy(data['errorcorrect'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'error: correct trials')
mne.grand_average(deepcopy(data['errorincorrect'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'error regressor: incorrect trials')
mne.grand_average(deepcopy(data['errorincorrvscorr'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'error contrast: incorrect vs correct trials')
#%%
 
# now for channels FCz, Fz, Cz, we want to plot the ERPs in the different trial types

for channel in ['Fz', 'FCz', 'Cz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    correct    = data['errorcorrect'],
                    incorrect  = data['errorincorrect'],
                    difference = data['errorincorrvscorr']),
            colors = dict(
                    correct = '#4daf4a',
                    incorrect = '#e41a1c',
                    difference = '#000000'),
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'error regressor: electrode '+channel, truncate_xaxis = False)

#%%
#cool now that basic stuff is done, we want to take a look at the difference wave and see if this difference is significant at any point
#we will focus on Cz and FCz based on previous literature looking at feedback evoked responses and the feedback ERN
            
tmin, tmax = 0,1 #set time window for the clustering

#gonna store things in a dictionary so we dont need to constantly rerun
t_ern, clu_ern, clupv_ern, h0_ern = dict(), dict(), dict(), dict()
masks_ern = dict()
clutimes = deepcopy(data['grandmean'][0].crop(tmin = tmin, tmax = tmax).times)
for channel in ['FCz', 'Cz']:
    t_ern[channel], clu_ern[channel], clupv_ern[channel], h0_ern[channel] = runclustertest_epochs(data = data,
                                                                                                  contrast_name = 'errorincorrvscorr',
                                                                                                  channels = [channel],
                                                                                                  tmin = tmin, tmax = tmax,
                                                                                                  gauss_smoothing = None,
                                                                                                  out_type = 'indices', n_permutations = 5000)
    masks_ern[channel] = np.asarray(clu_ern[channel])[clupv_ern[channel] < 0.05]


#given the clusters that are formed above, plot the ERPs and highlight significant clusters in time
for channel in ['FCz', 'Cz']:
    fig = plt.figure()
    ax = fig.subplots(1)
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    correct    = data['errorcorrect'],
                    incorrect  = data['errorincorrect'],
                    difference = data['errorincorrvscorr']),
            colors = dict(
                    correct = '#4daf4a',
                    incorrect = '#e41a1c',
                    difference = '#000000'),
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'error regressor: electrode '+channel, ylim = dict(eeg=[-3, 4]),
            axes = ax, truncate_xaxis = False, vlines = [0, 0.5])
    ax.set_title('Error regressor on the feedback ERP at electrode '+channel)
    ax.set_ylabel('beta (AU)')
    for mask in range(len(masks_ern[channel])):
        ax.hlines(y = -4,
                  xmin = np.min( clutimes[ masks_ern[channel][mask][1] ] ),
                  xmax = np.max( clutimes[ masks_ern[channel][mask][1] ] ),
                  lw = 4, color = '#636363', alpha = .5)
    
#given these clusters, let's plot the joint_plot for the difference wave, and highlight these clusters
fig = mne.grand_average(deepcopy(data['errorincorrvscorr'])).plot_joint(topomap_args = dict(contours = 0, vmin = -3, vmax = 3),
                      times = np.arange(0.05, 0.7, 0.15))
ax = fig.axes[0]
chan2use = 'Cz' #this will change the significant cluster lines depending on what channel you're taking clusters from
for mask in range(len(masks_ern[chan2use])):
    ax.hlines(y = -3.5,
              xmin = np.min( clutimes[masks_ern[chan2use][mask][1]] ),
              xmax = np.max( clutimes[masks_ern[chan2use][mask][1]] ),
              lw = 5, color = '#bdbdbd', alpha = .5)

# the final bit to do for the basic main effect contrasts is to plot the topographies of significant clusters for these channels
              
for channel in ['FCz', 'Cz']:
    tmins, tmaxs = [], [] #we're going to get the start and end points in time of these clusters
    for mask in masks_ern[channel]: #loop over clusters
        tmins.append( clutimes[mask[1]].min() )
        tmaxs.append( clutimes[mask[1]].max() )
    
    for imask in range(len(tmins)):
        itmin = tmins[imask] #get start time of the cluster
        itmax = tmaxs[imask] #get   end time of the cluster
        
        tcentre = np.add(itmin, np.divide(np.subtract(itmax,itmin),2)) #get the halfway point
        twidth  = np.subtract(itmax,itmin) #get the width of the cluster (creates timerange for the topomap)
        fig = mne.grand_average(data['errorincorrvscorr']).plot_topomap(times = tcentre,
                                                                   average = twidth,
                                                                   vmin = -3, vmax = 3, contours = 0,
                                                                   extrapolate = 'head', res = 300, #dpi
                                                                   title = 'error contrast: incorr vs corr, %s to %sms\nclu from channel: %s'%(str(np.round(itmin,2)), str(np.round(itmax,2)), channel))
#%% for the error regressor in each trial type separately:
        
tmin, tmax = 0,1 #set time window for the clustering

#gonna store things in a dictionary so we dont need to constantly rerun
t_cope, clu_cope, clupv_cope, h0_cope = dict(), dict(), dict(), dict()
masks_cope = dict()
for contrast in ['errorcorrect', 'errorincorrect']:
    for dictobj in [t_cope, clu_cope, clupv_cope, h0_cope, masks_cope]:
        dictobj[contrast] = dict()
        for channel in ['FCz', 'Cz']:
            dictobj[contrast][channel] = []

clutimes = deepcopy(data['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
for contrast in ['errorcorrect', 'errorincorrect']:
    for channel in ['FCz', 'Cz']:
        t_cope[contrast][channel], clu_cope[contrast][channel], clupv_cope[contrast][channel], h0_cope[contrast][channel] = runclustertest_epochs(
                    data = data,
                    contrast_name = contrast,
                    channels = [channel],
                    tmin = tmin, tmax = tmax,
                    gauss_smoothing = None,
                    out_type = 'indices', n_permutations = 5000)
        masks_cope[contrast][channel] = np.asarray(clu_cope[contrast][channel])[clupv_cope[contrast][channel] < 0.05]
        

for channel in ['FCz', 'Cz']:
    fig = plt.figure(figsize = (12,8))
    ax = fig.subplots(1)
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    correct    = data['errorcorrect'],
                    incorrect  = data['errorincorrect'],
                    difference = data['errorincorrvscorr']),
            colors = dict(
                    correct = '#4daf4a',
                    incorrect = '#e41a1c',
                    difference = '#525252'),
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'error regressor: electrode '+channel, ylim = dict(eeg=[-3, 4]),
            axes = ax, truncate_xaxis = False, vlines = [0, 0.5])
    ax.set_title('Error regressor on the feedback ERP at electrode '+channel)
    ax.set_ylabel('beta (AU)')
    for contrast in ['errorcorrect', 'errorincorrect']:
        if contrast == 'errorcorrect':
            col = '#4daf4a'
            liney = -1.2
        elif contrast == 'errorincorrect':
            col = '#e41a1c'
            liney = -1.4
        for imask in range(len(masks_cope[contrast][channel])):
            ax.hlines(y = liney,
                      xmin = np.min( clutimes[masks_cope[contrast][channel][imask][1]] ),
                      xmax = np.max( clutimes[masks_cope[contrast][channel][imask][1]] ),
                      lw = 4, color = col, alpha = .5)
    fig.savefig(fname = op.join(figpath, 'feedbackERP_errorRegressor_clustertimesFrom_%s_betas_laplacian.eps'%(channel)), format = 'eps', dpi = 300 )
    fig.savefig(fname = op.join(figpath, 'feedbackERP_errorRegressor_clustertimesFrom_%s_betas_laplacian.pdf'%(channel)), format = 'pdf', dpi = 300 )


#and now just plot the topographies of any significant clusters
contrast = 'errorcorrect'
for channel in ['FCz', 'Cz']:
    tmins, tmaxs = [], [] #we're going to get the start and end points in time of these clusters
    for mask in masks_cope[contrast][channel]: #loop over clusters
        tmins.append( clutimes[mask[1]].min() )
        tmaxs.append( clutimes[mask[1]].max() )
    
    for imask in range(len(tmins)):
        itmin = tmins[imask] #get start time of the cluster
        itmax = tmaxs[imask] #get   end time of the cluster
        
        tcentre = np.add(itmin, np.divide(np.subtract(itmax,itmin),2)) #get the halfway point
        twidth  = np.subtract(itmax,itmin) #get the width of the cluster (creates timerange for the topomap)
        fig = mne.grand_average(data[contrast]).plot_topomap(times = tcentre,
                                                               average = twidth,
                                                               vmin = -.6, vmax = .6, contours = 0,
                                                               extrapolate = 'head', res = 300, #dpi
                                                               title = 'error: correct trials, %s to %sms\nclu from channel: %s'%(str(np.round(itmin,2)), str(np.round(itmax,2)), channel))
        fig.savefig(fname = op.join(figpath, 'errorCorrect_cluster_%sms_to_%sms_betas_fromChannel_%s_laplacian.eps'%(str(np.round(itmin,2)), str(np.round(itmax,2)), channel)), format = 'eps', dpi = 300 )
        fig.savefig(fname = op.join(figpath, 'errorCorrect_cluster_%sms_to_%sms_betas_fromChannel_%s_laplacian.pdf'%(str(np.round(itmin,2)), str(np.round(itmax,2)), channel)), format = 'pdf', dpi = 300 )
