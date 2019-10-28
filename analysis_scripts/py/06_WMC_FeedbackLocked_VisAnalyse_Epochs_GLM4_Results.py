#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:30:29 2019

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

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm



contrasts = ['grandmean', 'neutral', 'cued', 'cuedvsneutral', 'pside',
             'incorrvscorr', 'error', 'conferror', 'confidence',
             'error_corr', 'error_incorr', 'error_incorrvscorr',
             'conferr_corr', 'conferr_incorr', 'conferror_incorrvscorr',
             'confidence_corr', 'confidence_incorr', 'confidence_incorrvscorr']

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
        data[name].append(   mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm4', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_betas-ave.fif'))[0])
        data_t[name].append( mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm4', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_tstats-ave.fif'))[0])
#%%
#drop right mastoid from literally everything here lol its not useful anymore
for cope in data.keys():
    for i in range(subs.size):
        data[cope][i]   = data[cope][i].drop_channels(['RM'])
        data_t[cope][i] = data_t[cope][i].drop_channels(['RM'])
#%%
        
gave_pside = mne.grand_average(data_t['pside']); gave_pside.data = toverparam(data_t['pside'])
gave_pside.plot_joint(title = 'probed side (left i right)', picks = 'eeg', topomap_args = dict(outlines='head', contours = 0))


#%%
gave_grandmean    = mne.grand_average(data_t['grandmean']); gave_grandmean.data = toverparam(data_t['grandmean'])
gave_incorrvscorr = mne.grand_average(data_t['incorrvscorr']); gave_incorrvscorr.data = toverparam(data_t['incorrvscorr'])
gave_incorrvscorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0))

mne.viz.plot_compare_evokeds(
        evokeds = dict(
                grand_mean = data_t['grandmean'],
                incorrvscorr = data_t['incorrvscorr'],),
        colors = dict(
                grand_mean = '#252525',
                incorrvscorr = '#4292c6'),
        show_legend = 'upper right', picks = 'FCZ',
        ci = .68, show_sensors = False,
        truncate_xaxis = False,
        )
plt.title('grand mean and difference wave between incorrect and incorrect trials at FCz')

#nonpara cluster t test to see where diff is significant
tmin, tmax = 0.0, 1.0 #specify time window for the cluster test to work
X_diff = np.empty(shape = (len(subs), 1, deepcopy(gave_incorrvscorr).crop(tmin=tmin, tmax=tmax).times.size))
for i in range(len(data_t['incorrvscorr'])):
    tmp = deepcopy(data_t['incorrvscorr'][i])
    tmp.pick_channels(['FCZ'])
    tmp.crop(tmin = tmin, tmax = tmax) #take only first 600ms for cluster test, time window for ERN and PE components
    X_diff[i,:,:] = tmp.data
np.random.seed(seed=1)
t_diff, clusters_diff, cluster_pv_diff, H0_diff = mne.stats.permutation_cluster_1samp_test(X_diff, out_type = 'indices')
mask_diff_05 = np.asarray(clusters_diff)[cluster_pv_diff<0.05]



fig = plt.figure()
ax = plt.axes()
alltimes = gave_incorrvscorr.times
times = deepcopy(gave_incorrvscorr).crop(tmin=tmin, tmax=tmax).times

ax.plot(alltimes, deepcopy(gave_incorrvscorr).pick_channels(['FCZ']).data[0], label = 'incorrect vs correct', color = '#4292c6', lw = 1.5)
ax.plot(alltimes, deepcopy(gave_grandmean).pick_channels(['FCZ']).data[0], label = 'grand mean', color = '#252525', lw = 1.5)
ax.set_title('feedback evoked response at electrode FCz')
ax.hlines(y = 0, linestyles = 'dashed', color = '#000000', lw = .75, xmin = alltimes.min(), xmax = alltimes.max())
ax.vlines(x = 0, linestyles = 'dashed', color = '#000000', lw = .75, ymin = -8, ymax = 10)
ax.set_ylabel('t-value')
ax.set_xlabel('Time relative to feedback onset (s)')
for mask in range(len(mask_diff_05)):
    ax.hlines(y = 0,
              xmin = np.min(times[mask_diff_05[mask][1]]),
              xmax = np.max(times[mask_diff_05[mask][1]]),
              lw=5, color = '#4292c6', alpha = .5) #plot significance timepoints for difference effect
              
              
#this will plot these significant times as an overlay onto the plot_joint image
#requires finding the right axis within the subplot thats drawn in order to draw them on properly              

#%%
fig1 = gave_incorrvscorr.plot_joint(picks = 'eeg',
                                    topomap_args = dict(contours=0, outlines='head', vmin = -5, vmax=5, scalings = dict(eeg=1), units = 'tstat'),
                                    ts_args = dict(unit = False, ylim = dict(eeg=[-9,9]), units = 'tstat'))
ax1 = fig1.axes[0]
for mask in range(len(mask_diff_05)):
    #x =  times[mask_diff_05[mask][1]]
    #y = np.zeros(len(x)); y.fill(-5.3)
    #ax.scatter(x, y, color = '#998ec3', alpha = .5, marker='s')
    ax1.hlines(y = -8,
              xmin = np.min(times[mask_diff_05[mask][1]]),
              xmax = np.max(times[mask_diff_05[mask][1]]),
              lw=5, color = '#4292c6', alpha = .5) #plot significance timepoints for difference effect
#%%    
gave_errorcorr   = mne.grand_average(data_t['error_corr']);   gave_errorcorr.data = toverparam(data_t['error_corr'])
gave_errorincorr = mne.grand_average(data_t['error_incorr']); gave_errorincorr.data = toverparam(data_t['error_incorr'])
gave_error_incorrvscorr = mne.grand_average(data_t['error_incorrvscorr']); gave_error_incorrvscorr.data = toverparam(data_t['error_incorrvscorr'])
gave_error = mne.grand_average(data_t['error']); gave_error.data =toverparam(data_t['error'])


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(alltimes, deepcopy(gave_errorcorr).pick_channels(['FCZ']).data[0], label = 'error - correct', color = '#2ca25f', lw = 1.5)
ax.plot(alltimes, deepcopy(gave_errorincorr).pick_channels(['FCZ']).data[0], label = 'error - incorrect', color = '#2ca25f', ls = 'dashed', lw = 1.5)
ax.plot(alltimes, deepcopy(gave_error_incorrvscorr).pick_channels(['FCZ']).data[0], label = 'error - incorrect vs correct', color = '#3182bd', lw = 1.5)
ax.plot(alltimes, deepcopy(gave_error).pick_channels(['FCZ']).data[0], label = 'error-alltrials', color = '#7b3294', lw = 1.5)
ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), linestyles = 'dashed', lw = .75, color = '#000000')
ax.vlines(x = 0, ymin = -4, ymax = 4, linestyles = 'dashed', color = '#000000', lw = .75)
fig.legend(loc = 'upper right')


#%%
gave_conferrcorr   = mne.grand_average(data_t['conferr_corr']);   gave_conferrcorr.data = toverparam(data_t['conferr_corr'])

gave_conferrcorr.plot_joint(picks = 'eeg', title = 'confidence error - correct',
                            topomap_args = dict(contours=0, outlines='head', vmin = -4, vmax=4, scalings = dict(eeg=1), units = 'tstat'),
                            ts_args = dict(unit = False, ylim = dict(eeg=[-6,6]), units = 'tstat'))

gave_conferrincorr = mne.grand_average(data_t['conferr_incorr']); gave_conferrincorr.data = toverparam(data_t['conferr_incorr'])
gave_conferrincorr.plot_joint(picks = 'eeg',title = 'confidence error - incorrect',
                              topomap_args = dict(contours=0, outlines='head', vmin = -4, vmax=4, scalings = dict(eeg=1), units = 'tstat'),
                              ts_args = dict(unit = False, ylim = dict(eeg=[-6,6]), units = 'tstat'))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(alltimes, deepcopy(gave_conferrcorr).pick_channels(['FCZ']).data[0], label = 'confidence error - correct', color = '#2ca25f', lw = 1.5)
ax.plot(alltimes, deepcopy(gave_conferrincorr).pick_channels(['FCZ']).data[0], label = 'confidence error - incorrect', color = '#2ca25f', ls = 'dashed', lw = 1.5)
ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), linestyles = 'dashed', lw = .75, color = '#000000')
ax.vlines(x = 0, ymin = -4, ymax = 4, linestyles = 'dashed', color = '#000000', lw = .75)
fig.legend(loc = 'upper right')
    
#%%

gave_conferr_incorrvscorr = mne.grand_average(data_t['conferror_incorrvscorr']); gave_conferr_incorrvscorr.data = toverparam(data_t['conferror_incorrvscorr'])
gave_conferr_incorrvscorr.plot_joint(picks = 'eeg',
                            topomap_args = dict(contours=0, outlines='head', vmin = -4, vmax=4, scalings = dict(eeg=1), units = 'tstat'),
                            ts_args = dict(unit = False, ylim = dict(eeg=[-6,6]), units = 'tstat'))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(alltimes, deepcopy(gave_conferrcorr).pick_channels(['FCZ']).data[0], label = 'confidence error - correct', color = '#2ca25f', lw = 1.5)
ax.plot(alltimes, deepcopy(gave_conferrincorr).pick_channels(['FCZ']).data[0], label = 'confidence error - incorrect', color = '#2ca25f', ls = 'dashed', lw = 1.5)
ax.plot(alltimes, deepcopy(gave_conferr_incorrvscorr).pick_channels(['FCZ']).data[0], label = 'confidence error - incorrect-correct', color = '#3182bd', lw = 1.5)
ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), linestyles = 'dashed', lw = .75, color = '#000000')
ax.vlines(x = 0, ymin = -4, ymax = 4, linestyles = 'dashed', color = '#000000', lw = .75)
fig.legend(loc = 'upper right')

#%%

#gave_pside = mne.grand_average(data_t['pside']); gave_pside.data = toverparam(data_t['pside'])
#gave_pside.plot_joint(picks = 'eeg',
#                      topomap_args = dict(contours=0, outlines='head', vmin = -3, vmax=3, scalings = dict(eeg=1), units = 'tstat'),
#                      ts_args = dict(unit = False, ylim = dict(eeg=[-6,6]), units = 'tstat'))

#%%
gave_confcorr = mne.grand_average(data_t['confidence_corr']); gave_confcorr.data = toverparam(data_t['confidence_corr'])
gave_confincorr = mne.grand_average(data_t['confidence_incorr']); gave_confincorr.data = toverparam(data_t['confidence_incorr'])
gave_confincorrvscorr = mne.grand_average(data_t['confidence_incorrvscorr']); gave_confincorrvscorr.data = toverparam(data_t['confidence_incorrvscorr'])



gave_confcorr.plot_joint(picks = 'eeg', title = 'confidence - correct trials',
                            topomap_args = dict(contours=0, outlines='head', vmin = -4, vmax=4, scalings = dict(eeg=1), units = 'tstat'),
                            ts_args = dict(unit = False, ylim = dict(eeg=[-6,6]), units = 'tstat'))
gave_confincorr.plot_joint(picks = 'eeg', title = 'confidence - incorrect trials',
                            topomap_args = dict(contours=0, outlines='head', vmin = -4, vmax=4, scalings = dict(eeg=1), units = 'tstat'),
                            ts_args = dict(unit = False, ylim = dict(eeg=[-6,6]), units = 'tstat'))
gave_confincorrvscorr.plot_joint(picks = 'eeg', title = 'confidence - incorrect vs correct trials',
                            topomap_args = dict(contours=0, outlines='head', vmin = -4, vmax=4, scalings = dict(eeg=1), units = 'tstat'),
                            ts_args = dict(unit = False, ylim = dict(eeg=[-6,6]), units = 'tstat'))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(alltimes, deepcopy(gave_confcorr).pick_channels(['FCZ']).data[0], label = 'confidence - correct', color = '#2ca25f', lw = 1.5)
ax.plot(alltimes, deepcopy(gave_confincorr).pick_channels(['FCZ']).data[0], label = 'confidence - incorrect', color = '#2ca25f', ls = 'dashed', lw = 1.5)
ax.plot(alltimes, deepcopy(gave_confincorrvscorr).pick_channels(['FCZ']).data[0], label = 'confidence - incorrect vs correct', color = '#3182bd', lw = 1.5)
ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), linestyles = 'dashed', lw = .75, color = '#000000')
ax.vlines(x = 0, ymin = -4, ymax = 4, linestyles = 'dashed', color = '#000000', lw = .75)
fig.legend(loc = 'upper right')



#%%
#nonpara cluster t test to see where diff is significant
tmin, tmax = 0.0, 1.0 #specify time window for the cluster test to work
X_diff = np.empty(shape = (len(subs), 1, deepcopy(gave_confincorrvscorr).crop(tmin=tmin, tmax=tmax).times.size))
for i in range(len(data_t['confidence_incorrvscorr'])):
    tmp = deepcopy(data_t['confidence_incorrvscorr'][i])
    tmp.pick_channels(['FCZ'])
    tmp.crop(tmin = tmin, tmax = tmax) #take only first 600ms for cluster test, time window for ERN and PE components
    X_diff[i,:,:] = tmp.data
np.random.seed(seed=1)
t_diff, clusters_diff, cluster_pv_diff, H0_diff = mne.stats.permutation_cluster_1samp_test(X_diff, out_type = 'indices')
mask_diff_05 = np.asarray(clusters_diff)[cluster_pv_diff<0.05]
#%%


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(alltimes, deepcopy(gave_confcorr).pick_channels(['FCZ']).data[0], label = 'confidence - correct', color = '#2ca25f', lw = 1.5)
ax.plot(alltimes, deepcopy(gave_confincorr).pick_channels(['FCZ']).data[0], label = 'confidence - incorrect', color = '#2ca25f', ls = 'dashed', lw = 1.5)
ax.plot(alltimes, deepcopy(gave_confincorrvscorr).pick_channels(['FCZ']).data[0], label = 'confidence - incorrect vs correct', color = '#3182bd', ls = 'dashed', lw = 1.5)
ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), linestyles = 'dashed', lw = .75, color = '#000000')
ax.vlines(x = 0, ymin = -4, ymax = 4, linestyles = 'dashed', color = '#000000', lw = .75)
fig.legend(loc = 'upper right')
for mask in range(len(mask_diff_05)):
    ax.hlines(y = -4,
              xmin = np.min(times[mask_diff_05[mask][1]]),
              xmax = np.max(times[mask_diff_05[mask][1]]),
              lw=5, color = '#3182bd', alpha = .5) #plot significance timepoints for difference effect





fig1 = gave_confincorrvscorr.plot_joint(picks = 'eeg',
                                    topomap_args = dict(contours=0, outlines='head', vmin = -3, vmax=3, scalings = dict(eeg=1), units = 'tstat'),
                                    ts_args = dict(unit = False, ylim = dict(eeg=[-5,5]), units = 'tstat'))
ax1 = fig1.axes[0]
for mask in range(len(mask_diff_05)):
    #x =  times[mask_diff_05[mask][1]]
    #y = np.zeros(len(x)); y.fill(-5.3)
    #ax.scatter(x, y, color = '#998ec3', alpha = .5, marker='s')
    ax1.hlines(y = -4,
              xmin = np.min(times[mask_diff_05[mask][1]]),
              xmax = np.max(times[mask_diff_05[mask][1]]),
              lw=5, color = '#3182bd', alpha = .5) #plot significance timepoints for difference effect
#%%

fig = plt.figure(figsize = (10,7))
ax  = fig.add_subplot(311)
ax.set_title('correct trials')
ax.plot(alltimes, deepcopy(gave_errorcorr).pick_channels(['FCZ']).data[0], label = 'error', color = '#d7191c', lw = 1)
ax.plot(alltimes, deepcopy(gave_confcorr).pick_channels(['FCZ']).data[0],  label = 'confidence', color = '#2c7bb6', lw = 1)
ax.plot(alltimes, deepcopy(gave_conferrcorr).pick_channels(['FCZ']).data[0], label = 'confidence error', color = '#a6dba0', lw = 1)
ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
ax.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax.set_ylabel('t-value')
ax.set_ylim([-6,3.5])

ax2 = fig.add_subplot(312)
ax2.set_title('incorrect trials')
ax2.plot(alltimes, deepcopy(gave_errorincorr).pick_channels(['FCZ']).data[0], color = '#d7191c', lw = 1)
ax2.plot(alltimes, deepcopy(gave_confincorr).pick_channels(['FCZ']).data[0], color = '#2c7bb6', lw = 1)
ax2.plot(alltimes, deepcopy(gave_conferrincorr).pick_channels(['FCZ']).data[0], color = '#a6dba0', lw = 1)
ax2.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
ax2.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax2.set_ylabel('t-value')
ax2.set_ylim([-4,4.5])

ax3 = fig.add_subplot(313)
ax3.set_title('incorrect-correct trials')
ax3.plot(alltimes, deepcopy(gave_error_incorrvscorr).pick_channels(['FCZ']).data[0], color = '#d7191c', lw = 1)
ax3.plot(alltimes, deepcopy(gave_confincorrvscorr).pick_channels(['FCZ']).data[0], color = '#2c7bb6', lw = 1)
ax3.plot(alltimes, deepcopy(gave_conferr_incorrvscorr).pick_channels(['FCZ']).data[0], color = '#a6dba0', lw = 1)
ax3.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
ax3.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax3.set_ylabel('t-value')
ax3.set_ylim([-4,4.5])
ax3.set_xlabel('Time relative to feedback onset (s)')
fig.legend(loc = 'upper left')

plt.tight_layout()

#%%

t_errcorr, clusters_errcorr, clusters_pv_errcorr, _                 = runclustertest_epochs(data = data_t, contrast_name = 'error_corr', channels = ['FCZ'], tmin = 0, tmax = 1) 
t_confcorr, clusters_confcorr, clusters_pv_confcorr, _              = runclustertest_epochs(data = data_t, contrast_name = 'confidence_corr', channels = ['FCZ'], tmin = 0, tmax = 1) 
t_conferr_corr, clusters_conferr_corr, clusters_pv_conferr_corr, _  = runclustertest_epochs(data = data_t, contrast_name = 'conferr_corr', channels = ['FCZ'], tmin = 0, tmax = 1) 


t_errincorr, clusters_errincorr, clusters_pv_errincorr, _                   = runclustertest_epochs(data = data_t, contrast_name = 'error_incorr', channels = ['FCZ'], tmin = 0, tmax = 1) 
t_confincorr, clusters_confincorr, clusters_pv_confincorr, _                = runclustertest_epochs(data = data_t, contrast_name = 'confidence_incorr', channels = ['FCZ'], tmin = 0, tmax = 1) 
t_conferr_incorr, clusters_conferr_incorr, clusters_pv_conferr_incorr, _    = runclustertest_epochs(data = data_t, contrast_name = 'conferr_incorr', channels = ['FCZ'], tmin = 0, tmax = 1) 

t_errivsc, clusters_errivsc, clusters_pv_errivsc, _                 = runclustertest_epochs(data = data_t, contrast_name = 'error_incorrvscorr', channels = ['FCZ'], tmin = 0, tmax = 1) 
t_confivsc, clusters_confivsc, clusters_pv_confivsc, _              = runclustertest_epochs(data = data_t, contrast_name = 'confidence_incorrvscorr', channels = ['FCZ'], tmin = 0, tmax = 1) 
t_conferr_ivsc, clusters_conferr_ivsc, clusters_pv_conferr_ivsc, _  = runclustertest_epochs(data = data_t, contrast_name = 'conferror_incorrvscorr', channels = ['FCZ'], tmin = 0, tmax = 1) 
