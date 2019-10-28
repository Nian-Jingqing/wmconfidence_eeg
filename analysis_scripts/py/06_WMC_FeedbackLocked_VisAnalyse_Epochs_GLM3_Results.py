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
from wmConfidence_funcs import gesd, plot_AR, toverparam, smooth

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
subs = np.array([1,       4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm
subs = np.array([1,       4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16])

alldata_grandmean       = []
alldata_neutral         = []
alldata_cued            = []
alldata_errorcorr       = []
alldata_errorincorr     = []
alldata_confcorr        = []
alldata_confincorr      = []
alldata_incorrvscorr    = []

alldata_grandmean_t       = []
alldata_neutral_t         = []
alldata_cued_t            = []
alldata_errorcorr_t       = []
alldata_errorincorr_t     = []
alldata_confcorr_t        = []
alldata_confincorr_t      = []
alldata_incorrvscorr_t    = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    alldata_grandmean.append(mne.read_evokeds(fname     = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_grandmean_betas-ave.fif'))[0])
    alldata_neutral.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_neutral_betas-ave.fif'))[0])
    alldata_cued.append(mne.read_evokeds(fname          = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_cued_betas-ave.fif'))[0])
    alldata_errorcorr.append(mne.read_evokeds(fname     = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_errorcorrect_betas-ave.fif'))[0])
    alldata_errorincorr.append(mne.read_evokeds(fname   = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_errorincorrect_betas-ave.fif'))[0])
    alldata_confcorr.append(mne.read_evokeds(fname      = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confcorrect_betas-ave.fif'))[0])
    alldata_confincorr.append(mne.read_evokeds(fname    = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confincorrect_betas-ave.fif'))[0])
    alldata_incorrvscorr.append(mne.read_evokeds(fname  = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_incorrvscorr_betas-ave.fif'))[0])


    alldata_grandmean_t.append(mne.read_evokeds(fname     = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_grandmean_tstats-ave.fif'))[0])
    alldata_neutral_t.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_neutral_tstats-ave.fif'))[0])
    alldata_cued_t.append(mne.read_evokeds(fname          = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_cued_tstats-ave.fif'))[0])
    alldata_errorcorr_t.append(mne.read_evokeds(fname     = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_errorcorrect_tstats-ave.fif'))[0])
    alldata_errorincorr_t.append(mne.read_evokeds(fname   = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_errorincorrect_tstats-ave.fif'))[0])
    alldata_confcorr_t.append(mne.read_evokeds(fname      = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confcorrect_tstats-ave.fif'))[0])
    alldata_confincorr_t.append(mne.read_evokeds(fname    = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confincorrect_tstats-ave.fif'))[0])
    alldata_incorrvscorr_t.append(mne.read_evokeds(fname  = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_incorrvscorr_tstats-ave.fif'))[0])
    
#%%
#ERN difference
gave_incorrvscorr = mne.grand_average(alldata_incorrvscorr_t); gave_incorrvscorr.drop_channels(['RM'])
#gave_incorrvscorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0))

#mne.viz.plot_compare_evokeds(
#        evokeds = dict(
#                grand_mean = alldata_grandmean,
#                incorrvscorr = alldata_incorrvscorr),
#        colors = dict(
#                grand_mean = '#252525',
#                incorrvscorr = '#4292c6'),
#        show_legend = 'upper right', picks = 'FCZ',
#        ci = .68, show_sensors = False,
#        truncate_xaxis = False,
#        )
#plt.title('grand mean and difference wave between incorrect and incorrect trials at FCz')

#nonpara cluster t test to see where diff is significant
tmin, tmax = 0.0, 1.0 #specify time window for the cluster test to work
X_diff = np.empty(shape = (len(subs), 1, gave_incorrvscorr.crop(tmin=tmin, tmax=tmax).times.size))
for i in range(len(alldata_incorrvscorr_t)):
    tmp = deepcopy(alldata_incorrvscorr_t[i])
    tmp.pick_channels(['FCZ'])
    tmp.crop(tmin = tmin, tmax = tmax) #take only first 600ms for cluster test, time window for ERN and PE components
    X_diff[i,:,:] = tmp.data



np.random.seed(seed=1)
t_diff, clusters_diff, cluster_pv_diff, H0_diff = mne.stats.permutation_cluster_1samp_test(X_diff, out_type = 'indices')
mask_diff_05 = np.asarray(clusters_diff)[cluster_pv_diff<0.05]

fig = plt.figure()
ax = plt.axes()
times = gave_incorrvscorr.times

mne.viz.plot_compare_evokeds(
        evokeds = dict(
                grand_mean = alldata_grandmean,
                incorrvscorr = alldata_incorrvscorr),
        colors = dict(
                grand_mean = '#252525',
                incorrvscorr = '#4292c6'),
        show_legend = 'upper right', picks = 'FCZ',
        ci = .68, show_sensors = False,
        truncate_xaxis = False, ylim = dict(eeg=[-3,8]),
        vlines = [0, 1],
        axes = ax
        )
        
ax.set_title('feedback evoked response at electrode FCz')
ax.set_ylabel('average beta (AU)')
for mask in range(len(mask_diff_05)):
    #x =  times[mask_diff_05[mask][1]]
    #y = np.zeros(len(x)); y.fill(-5.3)
    #ax.scatter(x, y, color = '#998ec3', alpha = .5, marker='s')
    ax.hlines(y = -2.5,
              xmin = np.min(times[mask_diff_05[mask][1]]),
              xmax = np.max(times[mask_diff_05[mask][1]]),
              lw=5, color = '#4292c6', alpha = .5) #plot significance timepoints for difference effect
              
              
#this will plot these significant times as an overlay onto the plot_joint image
#requires finding the right axis within the subplot thats drawn in order to draw them on properly              
gave_incorrvscorr = mne.grand_average(alldata_incorrvscorr_t); gave_incorrvscorr.drop_channels(['RM'])
fig1 = gave_incorrvscorr.plot_joint(picks = 'eeg', topomap_args = dict(contours=0, outlines='head', vmin = -2e6, vmax=2e6))
ax1 = fig1.axes[0]
for mask in range(len(mask_diff_05)):
    #x =  times[mask_diff_05[mask][1]]
    #y = np.zeros(len(x)); y.fill(-5.3)
    #ax.scatter(x, y, color = '#998ec3', alpha = .5, marker='s')
    ax1.hlines(y = -3.5e6,
              xmin = np.min(times[mask_diff_05[mask][1]]),
              xmax = np.max(times[mask_diff_05[mask][1]]),
              lw=5, color = '#4292c6', alpha = .5) #plot significance timepoints for difference effect
#%%
gave_gmean = mne.grand_average(alldata_grandmean_t); gave_gmean.data = toverparam(alldata_grandmean_t);gave_gmean.drop_channels(['RM'])
gave_badvsgood = mne.grand_average(alldata_incorrvscorr_t);  gave_badvsgood.data = toverparam(alldata_incorrvscorr_t) ; gave_badvsgood.drop_channels(['RM'])


fig = plt.figure()
ax = plt.axes()
mne.viz.plot_compare_evokeds(
        evokeds = dict(grand_mean = gave_gmean,
                       difference = gave_badvsgood,
                       correct    = mne.combine_evoked(all_evoked = [gave_gmean, -gave_badvsgood], weights = 'equal'),
                       incorrect  = mne.combine_evoked(all_evoked = [gave_gmean, gave_badvsgood], weights = 'equal')
                       ),
        colors = dict(
                grand_mean = 'black',
                difference = 'blue',
                correct = 'green',
                incorrect = 'red'
                ),
        show_legend = 'upper right',
        show_sensors = False,
        picks = 'FCZ', ci = .68#, axes = ax
)


fig = plt.figure()
ax = fig.subplots(1)
times2 = gave_gmean.times
ax.plot(times2, np.squeeze(deepcopy(gave_badvsgood).pick_channels(['FCZ']).data), label = 'difference', lw = 1.5, color = '#8da0cb')
ax.plot(times2, np.squeeze(mne.combine_evoked(all_evoked = [gave_gmean, -gave_badvsgood], weights = 'equal').pick_channels(['FCZ']).data), label = 'correct', lw = 1.5, color = '#66c2a5')
ax.plot(times2, np.squeeze(mne.combine_evoked(all_evoked = [gave_gmean, gave_badvsgood], weights = 'equal').pick_channels(['FCZ']).data), label = 'incorrect', lw = 1.5, color = '#fc8d62')
ax.hlines(0, xmin=np.min(times2), xmax = np.max(times2), lw = .75, linestyle='dashed')
ax.vlines(0, ymin=-6, ymax=6, lw=.75, linestyle='--')
ax.legend(loc='upper right')
        
ax.set_ylabel('Time rel to feedback onset (s)')
for mask in range(len(mask_diff_05)):
    #x =  times[mask_diff_05[mask][1]]
    #y = np.zeros(len(x)); y.fill(-5.3)
    #ax.scatter(x, y, color = '#998ec3', alpha = .5, marker='s')
    ax.hlines(y = -6.5,
              xmin = np.min(times[mask_diff_05[mask][1]]),
              xmax = np.max(times[mask_diff_05[mask][1]]),
              lw=5, color = '#4292c6', alpha = .5) #plot significance timepoints for difference effect




#%%
gave_errorcorr = mne.grand_average(alldata_errorcorr); gave_errorcorr.drop_channels(['RM'])
gave_errorcorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                          title = 'error regressor correct trials')

gave_errorincorr = mne.grand_average(alldata_errorincorr); gave_errorincorr.drop_channels(['RM'])
gave_errorincorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'error regressor incorrect trials')

gave_confcorr = mne.grand_average(alldata_confcorr); gave_confcorr.drop_channels(['RM'])
gave_confcorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                         title = 'confidence regressor correct trials')


#%%
#show FCz for error main effect in correct and incorrect trials
mne.viz.plot_compare_evokeds(
        evokeds = dict(
                errorcorr = alldata_errorcorr,
                errorincorr = alldata_errorincorr),
        colors = dict(
                errorcorr   = '#2171b5',
                errorincorr = '#41ab5d'),
        linestyles = dict(errorincorr = '--'),
        picks = 'FCZ',
        show_sensors   = False,
        show_legend    = 'upper right',
        truncate_xaxis = False,
        ci             = .68
        )
plt.title('error regressor for correct and incorrect trials at electrode FCz')

#show FCz for confidence main effect in correct and incorrect trials
mne.viz.plot_compare_evokeds(
        evokeds = dict(
                confcorr = alldata_confcorr,
                confincorr = alldata_confincorr
                ),
        colors = dict(
                confcorr    = '#2171b5',
                confincorr  = '#41ab5d'
                ),
        linestyles = dict(confincorr  = '--'),
        picks = 'FCZ',
        show_sensors   = False,
        show_legend    = 'upper right',
        truncate_xaxis = False,
        ci             = .68
        )
plt.title('confidence regressor for correct and incorrect trials at electrode FCz')

#%%

gave_errorcorr_t = mne.grand_average(alldata_errorcorr_t); gave_errorcorr_t.data = toverparam(alldata_errorcorr_t); gave_errorcorr_t.drop_channels(['RM'])
gave_errorincorr_t = mne.grand_average(alldata_errorcorr_t); gave_errorincorr_t.data = toverparam(alldata_errorincorr_t); gave_errorincorr_t.drop_channels(['RM'])


gave_confcorr_t = mne.grand_average(alldata_confcorr_t); gave_confcorr_t.data = toverparam(alldata_confcorr_t); gave_confcorr_t.drop_channels(['RM'])
gave_confincorr_t = mne.grand_average(alldata_confincorr_t); gave_confincorr_t.data = toverparam(alldata_confincorr_t); gave_confincorr_t.drop_channels(['RM'])

times3 = gave_errorcorr_t.times

fig = plt.figure()
fig.suptitle('FCz - effect of error')
ax = fig.subplots(1)
ax.plot(times3, np.squeeze(deepcopy(gave_errorcorr_t).pick_channels(['FCZ']).data), label = 'error - correct trials', lw = 1.5, color = '#2171b5')
ax.plot(times3, np.squeeze(deepcopy(gave_errorincorr_t).pick_channels(['FCZ']).data), label = 'error - incorrect trials', lw = 1.5, color = '#2171b5', linestyle = 'dashed')
ax.legend()
ax.hlines(y=0, linestyle='--', color='k', lw=.75, xmin=np.min(times3), xmax = np.max(times3))
ax.vlines(x=0, linestyle='--', color='k', lw=.75, ymin=-6, ymax = 4)
ax.set_ylabel('t value');
ax.set_xlabel('time relative to feedback onset (s)')


fig = plt.figure()
fig.suptitle('FCz - effect of confidence')
ax = fig.subplots(1)
ax.plot(times3, np.squeeze(deepcopy(gave_confcorr_t).pick_channels(['FCZ']).data), label = 'confidence - correct trials', lw = 1.5, color = '#41ab5d')
ax.plot(times3, np.squeeze(deepcopy(gave_confincorr_t).pick_channels(['FCZ']).data), label = 'confidence - incorrect trials', lw = 1.5, color = '#41ab5d', linestyle = 'dashed')
ax.legend()
ax.hlines(y=0, linestyle='--', color='k', lw=.75, xmin=np.min(times3), xmax = np.max(times3))
ax.vlines(x=0, linestyle='--', color='k', lw=.75, ymin=-6, ymax = 4)
ax.set_ylabel('t value');
ax.set_xlabel('time relative to feedback onset (s)')