#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:52:03 2019

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

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, smooth, runclustertest_epochs

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

figpath = op.join(wd,'figures', 'eeg_figs', 'feedbacklocked', 'epochs_glm3')


subs = np.array([4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22,24, 25, 26])

contrasts = ['correct', 'incorrect', 'errorcorrect', 'errorincorrect', 'confcorrect', 'confincorrect', 'pside',
             'incorrvscorr', 'errorincorrvscorr', 'confincorrvscorr',
             'grandmean', 'error', 'conf']
             
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
        data[name].append( mne.read_evokeds( fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_' + name + '_betas-ave.fif'))[0])        
        data_t[name].append( mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_' + name + '_tstats-ave.fif'))[0])        

for name in contrasts:
    for i in range(subs.size):
        data[name][i].drop_channels(['RM'])
        data_t[name][i].drop_channels(['RM'])

    
#%%
#ERN difference
gave = mne.grand_average(data['grandmean']); times = gave.times; del(gave)      

for channel in ['FCZ', 'CZ', 'CPZ', 'PZ']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    corr   = data['correct'],
                    incorr = data['incorrect'],
                    diff   = data['incorrvscorr']
                    ),
            colors = dict(
                    corr   = '#66c2a5',
                    incorr = '#fc8d62',
                    diff   = '#8da0cb'),
            show_legend = 'upper right', picks = channel,
            ci = .68, show_sensors = False, title = 'electrode ' + channel,
            truncate_xaxis=False
            )
gave = mne.grand_average(data['incorrvscorr'])

deepcopy(gave).plot_joint(times = np.arange(0.1,0.7,.1),
                title = 'incorr vs corr - RM reference',
                picks='eeg')
        

tmin, tmax = 0, 1
for channel in ['FCZ', 'CZ', 'CPZ', 'PZ']:
    t_ern, clu_ern, clupv_ern, h0_ern = runclustertest_epochs(data = data,
                                                              contrast_name = 'incorrvscorr',
                                                              channels = [channel],
                                                              tmin = tmin,
                                                              tmax = tmax,
                                                              gauss_smoothing = None,
                                                              out_type = 'indices', n_permutations = 'Default'
                                                              )
    clutimes = deepcopy(data['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_ern = np.asarray(clu_ern)[clupv_ern < 0.05]
            
    
    fig = plt.figure()
    ax = plt.axes()
    mne.viz.plot_compare_evokeds(
        evokeds = dict(
                corr   = data['correct'],
                incorr = data['incorrect'],
                diff   = data['incorrvscorr']
                ),
        colors = dict(
                corr   = '#66c2a5',
                incorr = '#fc8d62',
                diff   = '#8da0cb'),
        show_legend = 'upper right', picks = channel,
        ci = .68, show_sensors = False, title = 'electrode ' + channel,
        truncate_xaxis=False, axes = ax,
        vlines = [0, 0.5]
        )
            
    ax.set_title('feedback evoked response at electrode '+ channel)
    ax.set_ylabel('average beta (AU)')
    for mask in range(len(masks_ern)):
        ax.hlines(y = -2.5,
                  xmin = np.min(clutimes[masks_ern[mask][1]]),
                  xmax = np.max(clutimes[masks_ern[mask][1]]),
                  lw=5, color = '#4292c6', alpha = .5) #plot significance timepoints for difference effect
              
              
#this will plot these significant times as an overlay onto the plot_joint image
#requires finding the right axis within the subplot thats drawn in order to draw them on properly              
fig1 = gave.plot_joint(picks = 'eeg', topomap_args = dict(contours=0, outlines='head', vmin = -4, vmax = 4),
                       times = np.arange(0.1, 0.7, .1))
ax1 = fig1.axes[0]
for mask in range(len(masks_ern)):
    ax1.hlines(y = -3.5,
              xmin = np.min(clutimes[masks_ern[mask][1]]),
              xmax = np.max(clutimes[masks_ern[mask][1]]),
              lw=5, color = '#4292c6', alpha = .5) #plot significance timepoints for difference effect

#%%
gave_errorcorr = mne.grand_average(data['errorcorrect']); gave_errorcorr.data = toverparam(data['errorcorrect'])
gave_errorcorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0), times = np.arange(0.1,0.7,0.05), ts_args=dict(scalings = dict(eeg = 1)),
                          title = 'error regressor correct trials')

gave_errorincorr = mne.grand_average(data['errorincorrect']);  gave_errorincorr.data = toverparam(data['errorincorrect'])
gave_errorincorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0), times = np.arange(0.1,0.7,0.05), ts_args=dict(scalings = dict(eeg = 1)),
                            title = 'error regressor incorrect trials')

gave_errorivsc = mne.grand_average(data['errorincorrvscorr']); gave_errorivsc.data = toverparam(data['errorincorrvscorr'])
gave_errorivsc.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0), times = np.arange(0.1,0.7,0.05), ts_args=dict(scalings = dict(eeg = 1)),
                            title = 'error regressor incorrect vs correct trials')



#%%
gave_confcorr   = mne.grand_average(data['confcorrect']);   gave_confcorr.data = toverparam(data['confcorrect'])
gave_confcorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0), times = np.arange(0.1,0.7,0.05), ts_args=dict(scalings = dict(eeg = 1)),
                          title = 'confidence regressor correct trials')


gave_confincorr = mne.grand_average(data['confincorrect']); gave_confincorr.data = toverparam(data['confincorrect'])
gave_confincorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0), times = np.arange(0.1,0.7,0.05), ts_args=dict(scalings = dict(eeg = 1)),
                          title = 'confidence regressor incorrect trials')


gave_confincorrvscorr = mne.grand_average(data['confincorrvscorr']); gave_confincorrvscorr.data = toverparam(data['confincorrvscorr'])
gave_confincorrvscorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0), times = np.arange(0.1,0.7,0.05), ts_args=dict(scalings = dict(eeg = 1)),
                          title = 'confidence regressor incorrect vs correct trials')

#%%

np.random.seed(seed=1)
tmin, tmax = 0, 1.0
smooth_sigma = None
betas = True
alltimes = mne.grand_average(data['grandmean']).times
for channel in ['FCZ', 'CZ', 'CPZ', 'PZ']:
    if betas:
        dat2use = deepcopy(data)
        errcorr = mne.grand_average(dat2use['errorcorrect']); errcorr.data = toverparam(dat2use['errorcorrect'])
        confcorr = mne.grand_average(dat2use['confcorrect']); confcorr.data = toverparam(dat2use['confcorrect'])
        
        errincorr = mne.grand_average(dat2use['errorincorrect']); errincorr.data = toverparam(dat2use['errorincorrect'])
        confincorr = mne.grand_average(dat2use['confincorrect']); confincorr.data = toverparam(dat2use['confincorrect'])
        
        errivsc = mne.grand_average(dat2use['errorincorrvscorr']); errivsc.data = toverparam(dat2use['errorincorrvscorr'])
        confivsc = mne.grand_average(dat2use['confincorrvscorr']); confivsc.data = toverparam(dat2use['confincorrvscorr'])
        
        err = mne.grand_average(dat2use['error']); err.data = toverparam(dat2use['error'])
        conf = mne.grand_average(dat2use['conf']); conf.data = toverparam(dat2use['conf'])
    else:
        dat2use = deepcopy(data)
        errcorr = mne.grand_average(dat2use['errorcorrect']); errcorr.data = toverparam(dat2use['errorcorrect'])
        confcorr = mne.grand_average(dat2use['confcorrect']); confcorr.data = toverparam(dat2use['confcorrect'])
        
        errincorr = mne.grand_average(dat2use['errorincorrect']); errincorr.data = toverparam(dat2use['errorincorrect'])
        confincorr = mne.grand_average(dat2use['confincorrect']); confincorr.data = toverparam(dat2use['confincorrect'])
        
        errivsc = mne.grand_average(dat2use['errorincorrvscorr']); errivsc.data = toverparam(dat2use['errorincorrvscorr'])
        confivsc = mne.grand_average(dat2use['confincorrvscorr']); confivsc.data = toverparam(dat2use['confincorrvscorr'])
        
        err = mne.grand_average(dat2use['error']); err.data = toverparam(dat2use['error'])
        conf = mne.grand_average(dat2use['conf']); conf.data = toverparam(dat2use['conf'])
    
    t_errcorr, clusters_errcorr, clusters_pv_errcorr, _          = runclustertest_epochs(data = dat2use, contrast_name = 'errorcorrect', channels = [channel], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
    masks_errcorr = np.asarray(clusters_errcorr)[clusters_pv_errcorr < 0.05]
    
    t_confcorr, clusters_confcorr, clusters_pv_confcorr, _       = runclustertest_epochs(data = dat2use, contrast_name = 'confcorrect', channels = [channel], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
    masks_confcorr = np.asarray(clusters_confcorr)[clusters_pv_confcorr < 0.05]
    
    
    t_errincorr, clusters_errincorr, clusters_pv_errincorr, _    = runclustertest_epochs(data = dat2use, contrast_name = 'errorincorrect', channels = [channel], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
    masks_errincorr = np.asarray(clusters_errincorr)[clusters_pv_errincorr < 0.05]
    
    t_confincorr, clusters_confincorr, clusters_pv_confincorr, _ = runclustertest_epochs(data = dat2use, contrast_name = 'confincorrect', channels = [channel], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
    masks_confincorr = np.asarray(clusters_confincorr)[clusters_pv_confincorr < 0.05]
    
    t_errivsc, clusters_errivsc, clusters_pv_errivsc, _          = runclustertest_epochs(data = dat2use, contrast_name = 'errorincorrvscorr', channels = [channel], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
    masks_errivsc = np.asarray(clusters_errivsc)[clusters_pv_errivsc < 0.05]
    
    t_confivsc, clusters_confivsc, clusters_pv_confivsc, _       = runclustertest_epochs(data = dat2use, contrast_name = 'confincorrvscorr', channels = [channel], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
    masks_confivsc = np.asarray(clusters_confivsc)[clusters_pv_confivsc < 0.05]
    
    t_err, clusters_err, clusters_pv_err, _                 = runclustertest_epochs(data = dat2use, contrast_name = 'error', channels = [channel], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
    masks_err = np.asarray(clusters_err)[clusters_pv_err < 0.05]
    
    t_conf, clusters_conf, clusters_pv_conf, _              = runclustertest_epochs(data = dat2use, contrast_name = 'conf', channels = [channel], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
    masks_conf = np.asarray(clusters_conf)[clusters_pv_conf < 0.05]
    
    clutimes = deepcopy(data['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    
    fig = plt.figure(figsize = (10,7))
    fig.suptitle('channel '+channel)
    ax  = fig.add_subplot(411)
    ax.set_title('correct trials', loc = 'left')
    ax.plot(alltimes, deepcopy(errcorr).pick_channels([channel]).data[0], label = 'error', color = '#d7191c', lw = 1)
    ax.plot(alltimes, deepcopy(confcorr).pick_channels([channel]).data[0],  label = 'confidence', color = '#2c7bb6', lw = 1)
    ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
    ax.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
    ax.set_ylabel('t-value')
    ax.set_ylim([-6,6])
    for mask in masks_errcorr:
        start, stop = clutimes[mask[1]].min(), clutimes[mask[1]].max()
        ax.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#d7191c')
    for mask in masks_confcorr:
        start, stop = clutimes[mask[1]].min(), clutimes[mask[1]].max()
        ax.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')
    
    ax2 = fig.add_subplot(412)
    ax2.set_title('incorrect trials', loc = 'left')
    ax2.plot(alltimes, deepcopy(errincorr).pick_channels([channel]).data[0], color = '#d7191c', lw = 1)
    ax2.plot(alltimes, deepcopy(confincorr).pick_channels([channel]).data[0], color = '#2c7bb6', lw = 1)
    ax2.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
    ax2.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
    ax2.set_ylabel('t-value')
    ax2.set_ylim([-6,6])
    for mask in masks_errincorr:
        start, stop = clutimes[mask[1]].min(), clutimes[mask[1]].max()
        ax2.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#d7191c')
    for mask in masks_confincorr:
        start, stop = clutimes[mask[1]].min(), clutimes[mask[1]].max()
        ax2.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')
    
    ax3 = fig.add_subplot(413)
    ax3.set_title('incorrect-correct trials', loc = 'left')
    ax3.plot(alltimes, deepcopy(errivsc).pick_channels([channel]).data[0], color = '#d7191c', lw = 1)
    ax3.plot(alltimes, deepcopy(confivsc).pick_channels([channel]).data[0], color = '#2c7bb6', lw = 1)
    ax3.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
    ax3.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
    ax3.set_ylabel('t-value')
    ax3.set_ylim([-6,6])
    ax3.set_xlabel('Time relative to feedback onset (s)')
    for mask in masks_errivsc:
        start, stop = clutimes[mask[1]].min(), clutimes[mask[1]].max()
        ax3.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#d7191c')
    for mask in masks_confivsc:
        start, stop = clutimes[mask[1]].min(), clutimes[mask[1]].max()
        ax3.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')
                   
    ax4 = fig.add_subplot(414)
    ax4.set_title('all trials', loc = 'left')
    ax4.plot(alltimes, deepcopy(err).pick_channels([channel]).data[0], color = '#d7191c', lw = 1)
    ax4.plot(alltimes, deepcopy(conf).pick_channels([channel]).data[0], color = '#2c7bb6', lw = 1)
    ax4.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
    ax4.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
    ax4.set_ylabel('t-value')
    ax4.set_ylim([-6,6])
    ax4.set_xlabel('Time relative to feedback onset (s)')
    for mask in masks_err:
        start, stop = clutimes[mask[1]].min(), clutimes[mask[1]].max()
        ax4.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#d7191c')
    for mask in masks_conf:
        start, stop = clutimes[mask[1]].min(), clutimes[mask[1]].max()
        ax4.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')
    
    fig.legend(loc = 'upper left')
    plt.tight_layout()

#%%