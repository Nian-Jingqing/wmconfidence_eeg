#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 12:25:02 2019

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
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15])

alldata_grandmean       = []
alldata_neutral         = []
alldata_cued            = []
alldata_error           = []
alldata_conf            = []
alldata_correct         = []
alldata_incorrect       = []
alldata_errorcorr       = []
alldata_errorincorr     = []
alldata_confcorr        = []
alldata_confincorr      = []
alldata_badvsgood       = []

alldata_grandmean_t       = []
alldata_neutral_t         = []
alldata_cued_t            = []
alldata_error_t           = []
alldata_conf_t            = []
alldata_correct_t         = []
alldata_incorrect_t       = []
alldata_errorcorr_t       = []
alldata_errorincorr_t     = []
alldata_confcorr_t        = []
alldata_confincorr_t      = []
alldata_badvsgood_t       = []




for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    alldata_grandmean.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_grandmean_betas-ave.fif'))[0])
    alldata_neutral.append(mne.read_evokeds(fname         = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_neutral_betas-ave.fif'))[0])
    alldata_cued.append(mne.read_evokeds(fname            = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_cued_betas-ave.fif'))[0])
    alldata_error.append(mne.read_evokeds(fname           = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_error_betas-ave.fif'))[0])
    alldata_conf.append(mne.read_evokeds(fname            = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confidence_betas-ave.fif'))[0])
    alldata_correct.append(mne.read_evokeds(fname         = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_correct_betas-ave.fif'))[0])
    alldata_incorrect.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_incorrect_betas-ave.fif'))[0])
    alldata_errorcorr.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_errorcorrect_betas-ave.fif'))[0])
    alldata_errorincorr.append(mne.read_evokeds(fname     = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_errorincorrect_betas-ave.fif'))[0])
    alldata_confcorr.append(mne.read_evokeds(fname        = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confcorrect_betas-ave.fif'))[0])
    alldata_confincorr.append(mne.read_evokeds(fname      = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confincorrect_betas-ave.fif'))[0])
    alldata_badvsgood.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_redvsgreen_betas-ave.fif'))[0])
    

    alldata_grandmean_t.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_grandmean_tstats-ave.fif'))[0])
    alldata_neutral_t.append(mne.read_evokeds(fname         = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_neutral_tstats-ave.fif'))[0])
    alldata_cued_t.append(mne.read_evokeds(fname            = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_cued_tstats-ave.fif'))[0])
    alldata_error_t.append(mne.read_evokeds(fname           = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_error_tstats-ave.fif'))[0])
    alldata_conf_t.append(mne.read_evokeds(fname            = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confidence_tstats-ave.fif'))[0])
    alldata_correct_t.append(mne.read_evokeds(fname         = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_correct_tstats-ave.fif'))[0])
    alldata_incorrect_t.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_incorrect_tstats-ave.fif'))[0])
    alldata_errorcorr_t.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_errorcorrect_tstats-ave.fif'))[0])
    alldata_errorincorr_t.append(mne.read_evokeds(fname     = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_errorincorrect_tstats-ave.fif'))[0])
    alldata_confcorr_t.append(mne.read_evokeds(fname        = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confcorrect_tstats-ave.fif'))[0])
    alldata_confincorr_t.append(mne.read_evokeds(fname      = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confincorrect_tstats-ave.fif'))[0])
    alldata_badvsgood_t.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_glm1', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_redvsgreen_tstats-ave.fif'))[0])
    
#%%

# Question 1 - is there an error related signal in the scalp voltage, coarsely on whether there was green or red feedback?


gave_correct   = mne.grand_average(alldata_correct);      gave_correct.drop_channels(['RM'])
gave_incorrect = mne.grand_average(alldata_incorrect);    gave_incorrect.drop_channels(['RM'])
gave_corrvsincorr = mne.grand_average(alldata_badvsgood); gave_corrvsincorr.drop_channels(['RM'])

#clustering yea boi
#get the data together to begin with
X_diff = np.empty(shape = (len(subs), 1, 1000))
for i in range(len(alldata_badvsgood)):
    tmp = deepcopy(alldata_badvsgood[i])
    tmp.pick_channels(['FCZ']) #select only this channel
    X_diff[i,:,:] = tmp.data

np.random.seed(seed=1)
t_diff, clusters_diff, cluster_pv_diff, H0_diff = mne.stats.permutation_cluster_1samp_test(X_diff, out_type = 'indices')
mask_diff_05 = np.asarray(clusters_diff)[cluster_pv_diff<0.05]

#get data together for correct trials
X_corr = np.empty(shape = (len(subs), 1, 1000))
for i in range(len(alldata_correct)):
    tmp = deepcopy(alldata_correct[i])
    tmp.pick_channels(['FCZ']) #select only this channel
    X_corr[i,:,:] = tmp.data
    
t_corr, clusters_corr, cluster_pv_corr, H0_corr = mne.stats.permutation_cluster_1samp_test(X_corr, out_type = 'indices')
mask_corr_05 = np.asarray(clusters_corr)[cluster_pv_corr<0.05]


#get data together for incorrect trials
X_incorr = np.empty(shape = (len(subs), 1, 1000))
for i in range(len(alldata_incorrect)):
    tmp = deepcopy(alldata_incorrect[i])
    tmp.pick_channels(['FCZ']) #select only this channel
    X_incorr[i,:,:] = tmp.data
    
t_incorr, clusters_incorr, cluster_pv_incorr, H0_incorr = mne.stats.permutation_cluster_1samp_test(X_incorr, out_type = 'indices')
mask_incorr_05 = np.asarray(clusters_incorr)[cluster_pv_incorr<0.05]
#%% PLOT THESE CLUSTERS -- show the error-related negativity and where the differences are significant
#plot the data
fig = plt.figure()
ax = plt.axes()
times = alldata_correct[0].times

mne.viz.plot_compare_evokeds(evokeds = dict(
                                            incorrect   = alldata_incorrect,
                                            correct     = alldata_correct,
                                            diff        = alldata_badvsgood),
                                            #error       = alldata_error,
                                            #errorcorr   = alldata_errorcorr,
                                            #errorincorr = alldata_errorincorr),
                             picks = 'FCZ',
                             colors = dict(incorrect   = '#ef8a62',
                                           correct     = '#91cf60',
                                           diff        = '#998ec3'),
                                           #error       = '#377eb8',
                                           #errorcorr   = '#91cf60',
                                           #errorincorr = '#ef8a62'),
                             #linestyles = dict(errorcorr = '--', errorincorr = '--'),
                             ci = .68, axes = ax, ylim = dict(eeg=[-7,4]),
                             truncate_xaxis = False, show_legend = 'upper right',
                             show_sensors = False) #plots std error of mean

ax.set_title('feedback evoked response at electrode FCz')
ax.set_ylabel('average beta (AU)')
for mask in range(len(mask_diff_05)):
    #x =  times[mask_diff_05[mask][1]]
    #y = np.zeros(len(x)); y.fill(-5.3)
    #ax.scatter(x, y, color = '#998ec3', alpha = .5, marker='s')
    ax.hlines(y = -5.3,
              xmin = np.min(times[mask_diff_05[mask][1]]),
              xmax = np.max(times[mask_diff_05[mask][1]]),
              lw=5, color = '#998ec3', alpha = .5) #plot significance timepoints for difference effect

for mask in range(len(mask_corr_05)):
    #x =  times[mask_corr_05[mask][1]]
    #y = np.zeros(len(x)); y.fill(-5.1)
    #ax.scatter(x, y, color = '#91cf60', alpha = .5, marker='s')
    ax.hlines(y = -5,
              xmin = np.min(times[mask_corr_05[mask][1]]),
              xmax = np.max(times[mask_corr_05[mask][1]]),
              lw=5, color = '#91cf60', alpha = .5) #plot significance timepoints for difference effect

for mask in range(len(mask_incorr_05)):
    #x =  times[mask_incorr_05[mask][1]]
    #y = np.zeros(len(x)); y.fill(-5.2)
    #ax.scatter(x, y, color = '#ef8a62', alpha = .5, marker='s')
    ax.hlines(y = -5.2,
              xmin = np.min(times[mask_incorr_05[mask][1]]),
              xmax = np.max(times[mask_incorr_05[mask][1]]),
              lw=5, color = '#ef8a62', alpha = .5) #plot significance timepoints for difference effect
#%%

gave_error = mne.grand_average(alldata_error); gave_error.drop_channels(['RM'])
gave_error.plot_joint(picks = 'eeg', title = 'main effect of error', topomap_args = dict(outlines='head', contours=0))
#around 200ms post feedback onset there is a big frontal negativity that seems like an ERN like component
#these are betas - negative beta here means that lower voltages are associated with higher error
# not quite an ERN (which could be classically binary good vs bad feedback)
# but shows some error related modulation of frontal negativity
gave_error.plot(picks = 'FCZ')
plt.title('main effect of error at FCz')


gave_conf = mne.grand_average(alldata_conf); gave_conf.drop_channels(['RM'])
gave_conf.plot_joint(picks = 'eeg', title = 'main effect of confidence', topomap_args = dict(outlines='head', contours=0))

gave_conf.plot(picks = 'FCZ')
plt.title('main effect of confidence at FCz')

mne.viz.plot_compare_evokeds(picks = 'FCZ',
        evokeds = dict(
                error = alldata_error,
                confidence = alldata_conf),
        colors = dict(
                error = '#377eb8',
                confidence = '#4daf4a'),
        ci = .68
        )
#%% do some cluster tests to see where these regressors affect the feedback response at FCz
        
X_error = np.zeros(shape = (len(subs), 1, 1000)) #only running one channel
for i in range(len(alldata_error_t)):
    tmp = deepcopy(alldata_error_t[i])
    tmp.pick_channels(['FCZ'])
    X_error[i,:,:] = tmp.data
    
t_error, clusters_error, cluster_pv_error, H0_error = mne.stats.permutation_cluster_1samp_test(X_error, out_type = 'indices')
mask_error_05 = np.asarray(clusters_error)[cluster_pv_error<0.1]
        
X_conf = np.zeros(shape = (len(subs), 1, 1000)) #only running one channel
for i in range(len(alldata_conf_t)):
    tmp = deepcopy(alldata_conf_t[i])
    tmp.pick_channels(['FCZ'])
    X_conf[i,:,:] = tmp.data
    
t_conf, clusters_conf, cluster_pv_conf, H0_error = mne.stats.permutation_cluster_1samp_test(X_conf, out_type = 'indices')
mask_conf_05 = np.asarray(clusters_conf)[cluster_pv_conf<0.1]     


#%%plot the error and confidence regressors on FCz
fig = plt.figure()
ax = plt.axes()
times = alldata_error[0].times
mne.viz.plot_compare_evokeds(evokeds = dict(error = alldata_error_t,confidence = alldata_conf_t),
                             picks = 'FCZ',
                             colors = dict(error = '#377eb8', confidence = '#4daf4a'),
                             #linestyles = dict(errorcorr = '--', errorincorr = '--'),
                             ci = .68, axes = ax, ylim = dict(eeg=[-3*10e5,3*10e5]),
                             truncate_xaxis = False, show_legend = 'upper right',
                             show_sensors = False) #plots std error of mean

ax.set_title('feedback evoked response at electrode FCz')
ax.set_ylabel('average beta (AU)')
for mask in range(len(mask_error_05)):
    ax.hlines(y = -3*10e5,
              xmin = np.min(times[mask_error_05[mask][1]]),
              xmax = np.max(times[mask_error_05[mask][1]]),
              lw=5, color = '#377eb8', alpha = .5) #plot significance timepoints for difference effect

for mask in range(len(mask_conf_05)):
    ax.hlines(y = -3.1*10e5,
              xmin = np.min(times[mask_conf_05[mask][1]]),
              xmax = np.max(times[mask_conf_05[mask][1]]),
              lw=5, color = '#4daf4a', alpha = .5) #plot significance timepoints for difference effect
#%%

mne.viz.plot_compare_evokeds(
        evokeds = dict(
                target_inside   = alldata_errorcorr,
                target_outside = alldata_errorincorr
                ),
        colors = dict(
                target_inside   = '#91cf60',
                target_outside = '#ef8a62'
                ),
        picks = 'FCZ',
        ci = .68,
        truncate_xaxis = False,
        show_legend = 'upper right', show_sensors = False,
        title = 'main effect of error')
        
mne.viz.plot_compare_evokeds(
        evokeds = dict(
                target_inside   = alldata_confcorr,
                target_outside = alldata_confincorr
                ),
        colors = dict(
                target_inside   = '#91cf60',
                target_outside = '#ef8a62'
                ),
        picks = 'FCZ',
        ci = .68,
        truncate_xaxis = False,
        show_legend = 'upper right', show_sensors = False,
        title = 'main effect of confidence')
                
        
        
        

#%% some clustering? spatiotemporal?

X_error = np.zeros(shape = (len(subs), 61, 1000)); X_error.fill(np.nan)
#populate this data structure
for i in range(len(alldata_error)):
    tmp = deepcopy(alldata_error[i])
    tmp.drop_channels(['RM', 'VEOG', 'HEOG'])
    X_error[i,:,:] = tmp.data
#reshape this so channels are last
X_error = np.transpose(X_error, (0,2,1))

t_error, clusters_error, cluster_pv_error, H0_error = mne.stats.spatio_temporal_cluster_1samp_test(
        X_error,
        connectivity = mne.channels.find_ch_connectivity(gave_error.info, ch_type='eeg')[0],
        out_type = 'mask')
good_cluster_inds = np.where(cluster_pv_error < 0.1)[0]

tmins, tmaxs = np.min(times[clusters_error[good_cluster_inds[0]][0]]), np.max(times[clusters_error[good_cluster_inds[0]][0]])
cluster_error_channels = np.unique(clusters_error[good_cluster_inds[0]][1])
gave_error.plot_joint()
gave_error.drop_channels(['VEOG', 'HEOG'])
gave_error.plot_topomap(mask = clusters_error[good_cluster_inds[0]].T,
                        mask_params = dict(marker='x',markerfacecolor='k', markeredgecolor='k', linewidth = 0, markersize = 5),
                        contours = 0, average = .1,
                        outlines = 'head', vmin = -.75, vmax = .75,
                        times = np.arange(start = 0.15,stop = 0.25,step = .01))














