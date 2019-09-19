#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:04:58 2019

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

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm


alldata_grandmean     = []
alldata_cued          = []
alldata_neut          = []
alldata_pside         = []
alldata_pside_neut    = []
alldata_pside_cued    = []
alldata_pside_cvsn    = []
alldata_errorpside_neut = []
alldata_errorpside_cued = []
alldata_confpside_neut  = []
alldata_confpside_cued  = []
alldata_dtpside_neut    = []
alldata_dtpside_cued    = []

alldata_errorpside_cvsn = []
alldata_dtpside_cvsn    = []
alldata_confpside_cvsn  = []
alldata_errorpside      = []
alldata_dtpside         = []
alldata_confpside       = []


#tstat structures
alldata_grandmean_t     = []
alldata_cued_t          = []
alldata_neut_t          = []
alldata_pside_t         = []
alldata_pside_neut_t    = []
alldata_pside_cued_t    = []
alldata_pside_cvsn_t    = []
alldata_errorpside_neut_t = []
alldata_errorpside_cued_t = []
alldata_confpside_neut_t  = []
alldata_confpside_cued_t  = []
alldata_dtpside_neut_t    = []
alldata_dtpside_cued_t    = []

alldata_errorpside_cvsn_t = []
alldata_dtpside_cvsn_t    = []
alldata_confpside_cvsn_t  = []
alldata_errorpside_t      = []
alldata_dtpside_t         = []
alldata_confpside_t       = []


for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    alldata_grandmean.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_grandmean_betas-tfr.h5'))[0])
    alldata_cued.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_cued_betas-tfr.h5'))[0])
    alldata_neut.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_neutral_betas-tfr.h5'))[0])
    alldata_pside.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_pside_betas-tfr.h5'))[0])
    alldata_pside_neut.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_psideneutral_betas-tfr.h5'))[0])
    alldata_pside_cued.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_psidecued_betas-tfr.h5'))[0])
    alldata_pside_cvsn.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_pside_cvsn_betas-tfr.h5'))[0])
    alldata_errorpside_neut.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_errorxpside_neutral_betas-tfr.h5'))[0])
    alldata_errorpside_cued.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_errorxpside_cued_betas-tfr.h5'))[0])
    alldata_confpside_neut.append(mne.time_frequency.read_tfrs(fname    = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_confxpside_neutral_betas-tfr.h5'))[0])
    alldata_confpside_cued.append(mne.time_frequency.read_tfrs(fname    = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_confxpside_cued_betas-tfr.h5'))[0])
    alldata_dtpside_neut.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_DTxpside_neutral_betas-tfr.h5'))[0])
    alldata_dtpside_cued.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_DTxpside_cued_betas-tfr.h5'))[0])
    alldata_errorpside_cvsn.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_errorxpsidexcvsn_betas-tfr.h5'))[0])
    alldata_dtpside_cvsn.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_DTxpsidexcvsn_betas-tfr.h5'))[0])
    alldata_confpside_cvsn.append(mne.time_frequency.read_tfrs(fname    = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_confxpsidexcvsn_betas-tfr.h5'))[0])
    alldata_errorpside.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_errorxpside_betas-tfr.h5'))[0])
    alldata_dtpside.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_DTxpside_betas-tfr.h5'))[0])
    alldata_confpside.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_confxpside_betas-tfr.h5'))[0])
    
    #tstats
    alldata_grandmean_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_grandmean_tstats-tfr.h5'))[0])
    alldata_cued_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_cued_tstats-tfr.h5'))[0])
    alldata_neut_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_neutral_tstats-tfr.h5'))[0])
    alldata_pside_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_pside_tstats-tfr.h5'))[0])
    alldata_pside_neut_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_psideneutral_tstats-tfr.h5'))[0])
    alldata_pside_cued_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_psidecued_tstats-tfr.h5'))[0])
    alldata_pside_cvsn_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_pside_cvsn_tstats-tfr.h5'))[0])
    alldata_errorpside_neut_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_errorxpside_neutral_tstats-tfr.h5'))[0])
    alldata_errorpside_cued_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_errorxpside_cued_tstats-tfr.h5'))[0])
    alldata_confpside_neut_t.append(mne.time_frequency.read_tfrs(fname    = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_confxpside_neutral_tstats-tfr.h5'))[0])
    alldata_confpside_cued_t.append(mne.time_frequency.read_tfrs(fname    = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_confxpside_cued_tstats-tfr.h5'))[0])
    alldata_dtpside_neut_t.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_DTxpside_neutral_tstats-tfr.h5'))[0])
    alldata_dtpside_cued_t.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_DTxpside_cued_tstats-tfr.h5'))[0])
    
    alldata_errorpside_cvsn_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_errorxpsidexcvsn_tstats-tfr.h5'))[0])
    alldata_dtpside_cvsn_t.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_DTxpsidexcvsn_tstats-tfr.h5'))[0])
    alldata_confpside_cvsn_t.append(mne.time_frequency.read_tfrs(fname    = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_confxpsidexcvsn_tstats-tfr.h5'))[0])
    alldata_errorpside_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_errorxpside_tstats-tfr.h5'))[0])
    alldata_dtpside_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_DTxpside_tstats-tfr.h5'))[0])
    alldata_confpside_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_leftvsright_confxpside_tstats-tfr.h5'))[0])
    
#%%
#get some behavioural data that we should probably plot to make some sense of things
allbdata = []
for i in subs:
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    bdata = pd.read_csv(param['resplocked_tfr_meta'], index_col = None, header = 0) #read the metadata associated with this subject
    allbdata.append(bdata)

DTs = []
DTs_neut = []
DTs_cued = []
for i in allbdata:
    DTs.append(np.nanmean(i.DT))
    DTs_cued.append(np.nanmean(i.query('cond == \'cued\'').DT))
    DTs_neut.append(np.nanmean(i.query('cond == \'neutral\'').DT))
gave_mean_dt = np.nanmean(DTs)
gave_mean_dt_neut = np.nanmean(DTs_neut)
gave_mean_dt_cued = np.nanmean(DTs_cued)

#%%
chans_no_midline =np.array([           1,     3,
                                   4,  5,     7,  8,
                           9, 10, 11, 12,    14, 15, 16, 17,
                          18, 19, 20, 21,    23, 24, 25, 26,
                          27, 28, 29, 30,    32, 33, 34, 35,
                          36, 37, 38, 39,    41, 42, 43, 44,
                          45, 46, 47, 48,    50, 51, 52, 53,
                                  54, 55,    57, 58,
                                      59,    61,
                                          ])
chans_no_midline = np.subtract(chans_no_midline, 1)
right_chans = np.array([3,
                        7,  8,
                        14, 15, 16, 17,
                        23, 24, 25, 26,
                        32, 33, 34, 35,
                        41, 42, 43, 44,
                        50, 51, 52, 53,
                        57, 58,
                        61
        ])
right_chans = np.subtract(right_chans, 1)
channames_no_midline = np.array([
                           'FP1', 'FP2', 
                   ' AF7', 'AF3', 'AF4','AF8',
        'F7', 'F5',  'F3',  'F1',  'F2', 'F4',  'F6',   'F8',
       'FT7','FC5', 'FC3', 'FC1','FC2', 'FC4',' FC6',  'FT8',
        'T7', 'C5',  'C3',  'C1', 'C2',  'C4',  'C6',   'T8',
       'TP7','CP5', 'CP3', 'CP1','CP2', 'CP4', 'CP6',  'TP8',
        'P7', 'P5',  'P3',  'P1', 'P2',  'P4',  'P6',   'P8',
       'PO7',       'PO3',              'PO4',         'PO8',
                            'O1', 'O2'])


timefreqs_alpha = {(-0.6, 10):(.4, 4),
                   (-0.4, 10):(.4, 4),
                   (-0.2, 10):(.4, 4),
                   ( 0.0, 10):(.4, 4),
                   ( 0.2, 10):(.4, 4)}

timefreqs_beta  = {(-0.2, 22):(.4, 14),
                   ( 0.0, 22):(.4, 14),
                   ( 0.2, 22):(.4, 14)}
visrightchans = np.array(['PO4', 'PO8', 'O2'])
motrightchans = np.array(['C4', 'C2'])
#%%

gave_pside = mne.grand_average(alldata_pside_t); gave_pside.data = toverparam(alldata_pside_t)
gave_pside_neut = mne.grand_average(alldata_pside_neut_t); gave_pside_neut.data = toverparam(alldata_pside_neut_t)
gave_pside_cued = mne.grand_average(alldata_pside_cued_t); gave_pside_cued.data = toverparam(alldata_pside_cued_t)
gave_pside_cvsn = mne.grand_average(alldata_pside_cvsn_t); gave_pside_cvsn.data = toverparam(alldata_pside_cvsn_t)

vmin_vis = np.divide(np.min(np.nanmean(deepcopy(gave_pside).pick_channels(visrightchans).data, 0)),  1); vmin_vis = -2
vmax_vis = np.divide(np.min(np.nanmean(deepcopy(gave_pside).pick_channels(visrightchans).data, 0)), -1); vmax_vis =  2

vmin_mot = np.divide(np.min(np.nanmean(deepcopy(gave_pside).pick_channels(motrightchans).data,0)), 5);  vmin_mot = -2
vmax_mot = np.divide(np.min(np.nanmean(deepcopy(gave_pside).pick_channels(motrightchans).data,0)), -5); vmax_mot = 2

times, freqs = gave_pside.times, gave_pside.freqs

#%% visual channels
pside_data = [gave_pside, gave_pside_neut, gave_pside_cued, gave_pside_cvsn]
titles  = ['all trials', 'neutral trials', 'cued trials', 'cued vs neutral']
vlines_dt = [gave_mean_dt, gave_mean_dt_neut, gave_mean_dt_cued, gave_mean_dt]

fig = plt.figure()
fig.suptitle('cvsi to probed item, relative to probe, right visual channels')
ax = fig.subplots(4,1)
for i in range(len(ax)):
    ax[i].contourf(times, freqs, np.nanmean(deepcopy(pside_data[i]).pick_channels(visrightchans).data,0), levels=100, cmap ='RdBu_r', vmin = vmin_vis, vmax = vmax_vis)
    ax[i].vlines([0.0, vlines_dt[i]], ymin=1, ymax=39, lw = .5, linestyle = '--')
    ax[i].set_title(titles[i])

# clusters in the visual channels on cued trials?
x_cued_vis = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_cued_vis[i,:,:] = np.nanmean(deepcopy(alldata_pside_cued_t[i]).pick_channels(visrightchans).data,0)

t_cued_vis, clusters_cued_vis, cluster_pv_cued_vis, _ = mne.stats.permutation_cluster_1samp_test(x_cued_vis, n_permutations='all')
masks_cued_vis = np.asarray(clusters_cued_vis)[cluster_pv_cued_vis<0.05]

fig = plt.figure()
fig.suptitle('cvsi to probed item, relative to probe, right visual channels')
ax = fig.subplots(4,1)
for i in range(len(ax)):
    ax[i].contourf(times, freqs, np.nanmean(deepcopy(pside_data[i]).pick_channels(visrightchans).data,0), levels=100, cmap ='RdBu_r', vmin = vmin_vis, vmax = vmax_vis)
    ax[i].vlines([0.0, vlines_dt[i]], ymin=1, ymax=39, lw = .5, linestyle = '--')
    ax[i].set_title(titles[i])

for mask in masks_cued_vis:
    bigmask_cued = np.kron(mask, np.ones((10,10)))
    ax[2].contour(bigmask_cued, colors='black', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), linewidths = .5, corner_mask=False, antialiased = False)
    

# clusters in the visual channels on all trials?
x_all_vis = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_all_vis[i,:,:] = np.nanmean(deepcopy(alldata_pside_t[i]).pick_channels(visrightchans).data,0)

t_all_vis, clusters_all_vis, cluster_pv_all_vis, _ = mne.stats.permutation_cluster_1samp_test(x_all_vis, n_permutations='all', tail = -1)
masks_all_vis = np.asarray(clusters_all_vis)[cluster_pv_all_vis<0.05]

fig = plt.figure()
fig.suptitle('cvsi to probed item, relative to probe, right visual channels')
ax = fig.subplots(4,1)
for i in range(len(ax)):
    ax[i].contourf(times, freqs, np.nanmean(deepcopy(pside_data[i]).pick_channels(visrightchans).data,0), levels=100, cmap ='RdBu_r', vmin = vmin_vis, vmax = vmax_vis)
    ax[i].vlines([0.0, vlines_dt[i]], ymin=1, ymax=39, lw = .5, linestyle = '--')
    ax[i].set_title(titles[i])

for mask in masks_all_vis:
    bigmask_cued = np.kron(mask, np.ones((10,10)))
    ax[0].contour(bigmask_cued, colors='black', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), linewidths = .5, corner_mask=False, antialiased = False)
for mask in masks_cued_vis:
    bigmask_cued = np.kron(mask, np.ones((10,10)))
    ax[2].contour(bigmask_cued, colors='black', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), linewidths = .5, corner_mask=False, antialiased = False)
    
#for cued trial clusters
times_sig_cued = []
freqs_sig_cued = []
for mask in masks_cued_vis:
    times_sig_cued.append(times[np.where(mask == True)[1]])
    freqs_sig_cued.append(freqs[np.where(mask == True)[0]])
#gave_pside_cued.plot_topomap(tmin = np.min(times_sig_cued))
for cluster in range(len(times_sig_cued)):
    tmin, tmax = np.min(times_sig_cued[cluster]), np.max(times_sig_cued[cluster])
    fmin, fmax = np.min(freqs_sig_cued[cluster]), np.max(freqs_sig_cued[cluster])
    deepcopy(gave_pside_cued).pick_channels(channames_no_midline).plot_topomap(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, vmin=-2, vmax=2, contours=0)
    
#for all trials clusters
times_sig_cued = []
freqs_sig_cued = []
for mask in masks_all_vis:
    times_sig_cued.append(times[np.where(mask == True)[1]])
    freqs_sig_cued.append(freqs[np.where(mask == True)[0]])
#gave_pside_cued.plot_topomap(tmin = np.min(times_sig_cued))
for cluster in range(len(times_sig_cued)):
    tmin, tmax = np.min(times_sig_cued[cluster]), np.max(times_sig_cued[cluster])
    fmin, fmax = np.min(freqs_sig_cued[cluster]), np.max(freqs_sig_cued[cluster])
    deepcopy(gave_pside).pick_channels(channames_no_midline).plot_topomap(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, vmin=-2, vmax=2, contours=0)
    
# now for motor channels
fig = plt.figure()
fig.suptitle('cvsi to probed item, relative to probe, right motor channels')
ax = fig.subplots(4,1)
for i in range(len(ax)):
    ax[i].contourf(times, freqs, np.nanmean(deepcopy(pside_data[i]).pick_channels(motrightchans).data,0), levels=100, cmap ='RdBu_r', vmin = vmin_vis, vmax = vmax_vis)
    ax[i].vlines([0.0, vlines_dt[i]], ymin=1, ymax=39, lw = .5, linestyle = '--')
    ax[i].set_title(titles[i])

x_all_mot = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_all_mot[i,:,:] = np.nanmean(deepcopy(alldata_pside_t[i]).pick_channels(motrightchans).data,0)

t_all_mot, clusters_all_mot, clusters_pv_all_mot, _ = mne.stats.permutation_cluster_1samp_test(x_all_mot, n_permutations='all')
#no motor clusters in all trials TFR

x_cued_mot = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_cued_mot[i,:,:] = np.nanmean(deepcopy(alldata_pside_cued_t[i]).pick_channels(motrightchans).data,0)

t_cued_mot, clusters_cued_mot, clusters_pv_cued_mot, _ = mne.stats.permutation_cluster_1samp_test(x_cued_mot, n_permutations='all')
#no motor clusters in cued trials TFR either

#%% error ~ lateralisation relative to the probed item

gave_errorpside = mne.grand_average(alldata_errorpside_t); gave_errorpside.data = toverparam(alldata_errorpside_t)
gave_errorpside_neut = mne.grand_average(alldata_errorpside_neut_t); gave_errorpside_neut.data = toverparam(alldata_errorpside_neut_t)
gave_errorpside_cued = mne.grand_average(alldata_errorpside_cued_t); gave_errorpside_cued.data = toverparam(alldata_errorpside_cued_t)
gave_errorpside_cvsn = mne.grand_average(alldata_errorpside_cvsn_t); gave_errorpside_cvsn.data = toverparam(alldata_errorpside_cvsn_t)

errorpside_data = [gave_errorpside, gave_errorpside_neut, gave_errorpside_cued, gave_errorpside_cvsn]
titles  = ['all trials', 'neutral trials', 'cued trials', 'cued vs neutral']
vlines_dt = [gave_mean_dt, gave_mean_dt_neut, gave_mean_dt_cued, gave_mean_dt]

fig = plt.figure()
fig.suptitle('error ~ cvsi to probed item, relative to probe, right visual channels')
ax = fig.subplots(4,1)
for i in range(len(ax)):
    ax[i].contourf(times, freqs, np.nanmean(deepcopy(errorpside_data[i]).pick_channels(visrightchans).data,0), levels=100, cmap ='RdBu_r', vmin = vmin_vis, vmax = vmax_vis)
    ax[i].vlines([0.0, vlines_dt[i]], ymin=1, ymax=39, lw = .5, linestyle = '--')
    ax[i].set_title(titles[i])
fig

x_errorpside_cued = np.empty(shape = (subs.size, freqs.size, times.size))#deepcopy(gave_errorpside).crop(tmin=-1,tmax=0).times.size))
for i in range(subs.size):
    x_errorpside_cued[i,:,:] = np.nanmean(deepcopy(alldata_errorpside_t[i]).pick_channels(visrightchans).data,0)

t_errorpside_cued, clusters_errorpside_cued, cluster_pv_errorpside_cued, _ = mne.stats.permutation_cluster_1samp_test(x_errorpside_cued, n_permutations='all')
masks_errorpside_cued = np.asarray(clusters_errorpside_cued)[cluster_pv_errorpside_cued<0.99]

fig = plt.figure()
fig.suptitle('error ~ cvsi to probed item, relative to probe, right visual channels')
ax = fig.subplots(4,1)
for i in range(len(ax)):
    ax[i].contourf(times, freqs, np.nanmean(deepcopy(errorpside_data[i]).pick_channels(visrightchans).data,0), levels=100, cmap ='RdBu_r', vmin = vmin_vis, vmax = vmax_vis)
    ax[i].vlines([0.0, vlines_dt[i]], ymin=1, ymax=39, lw = .5, linestyle = '--')
    ax[i].set_title(titles[i])
#for mask in masks_errorpside_cued:
#    bigmask = np.kron(mask, np.ones((10,10)))
#    ax[2].contour(bigmask, colors='black', lw=.5, extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), linewidths = .5, corner_mask=False, antialiased = False)




x_errorpside_cvsn = np.empty(shape = (subs.size, freqs.size, times.size))#deepcopy(gave_errorpside).crop(tmin=-1,tmax=0).times.size))
for i in range(subs.size):
    x_errorpside_cvsn[i,:,:] = np.nanmean(deepcopy(alldata_errorpside_cvsn_t[i]).pick_channels(visrightchans).data,0)

t_errorpside_cvsn, clusters_errorpside_cvsn, cluster_pv_errorpside_cvsn, _ = mne.stats.permutation_cluster_1samp_test(x_errorpside_cvsn, n_permutations='all')
masks_errorpside_cvsn = np.asarray(clusters_errorpside_cvsn)[cluster_pv_errorpside_cvsn<0.5]

fig = plt.figure()
fig.suptitle('error ~ cvsi to probed item, relative to probe, right visual channels')
ax = fig.subplots(4,1)
for i in range(len(ax)):
    ax[i].contourf(times, freqs, np.nanmean(deepcopy(errorpside_data[i]).pick_channels(visrightchans).data,0), levels=100, cmap ='RdBu_r', vmin = vmin_vis, vmax = vmax_vis)
    ax[i].vlines([0.0, vlines_dt[i]], ymin=1, ymax=39, lw = .5, linestyle = '--')
    ax[i].set_title(titles[i])
for mask in masks_errorpside_cvsn:
    bigmask = np.kron(mask, np.ones((10,10)))
    ax[3].contour(bigmask, colors='black', lw=.5, extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), linewidths = .5, corner_mask=False, antialiased = False)
#%%

#DT ~ lateralisation relative to the probed item?

gave_dtpside = mne.grand_average(alldata_dtpside_t); gave_dtpside.data = toverparam(alldata_dtpside_t)
gave_dtpside_neut = mne.grand_average(alldata_dtpside_neut_t); gave_dtpside_neut.data = toverparam(alldata_dtpside_neut_t)
gave_dtpside_cued = mne.grand_average(alldata_dtpside_cued_t); gave_dtpside_cued.data = toverparam(alldata_dtpside_cued_t)
gave_dtpside_cvsn = mne.grand_average(alldata_dtpside_cvsn_t); gave_dtpside_cvsn.data = toverparam(alldata_dtpside_cvsn_t)

dtpside_data = [gave_dtpside, gave_dtpside_neut, gave_dtpside_cued, gave_dtpside_cvsn]
titles  = ['all trials', 'neutral trials', 'cued trials', 'cued vs neutral']
vlines_dt = [gave_mean_dt, gave_mean_dt_neut, gave_mean_dt_cued, gave_mean_dt]

fig = plt.figure()
fig.suptitle('DT ~ cvsi to probed item, relative to probe, right visual channels')
ax = fig.subplots(4,1)
for i in range(len(ax)):
    ax[i].contourf(times, freqs, np.nanmean(deepcopy(dtpside_data[i]).pick_channels(visrightchans).data,0), levels=100, cmap ='RdBu_r', vmin = vmin_vis, vmax = vmax_vis)
    ax[i].vlines([0.0, vlines_dt[i]], ymin=1, ymax=39, lw = .5, linestyle = '--')
    ax[i].set_title(titles[i])
fig

x_dtpside_cued = np.empty(shape = (subs.size, freqs.size, times.size))#deepcopy(gave_dtpside).crop(tmin=-1,tmax=0).times.size))
for i in range(subs.size):
    x_dtpside_cued[i,:,:] = np.nanmean(deepcopy(alldata_dtpside_cued_t[i]).pick_channels(visrightchans).data,0)

t_dtpside_cued, clusters_dtpside_cued, cluster_pv_dtpside_cued, _ = mne.stats.permutation_cluster_1samp_test(x_dtpside_cued, n_permutations='all')
masks_dtpside_cued = np.asarray(clusters_dtpside_cued)[cluster_pv_dtpside_cued<0.1]

fig = plt.figure()
fig.suptitle('DT ~ cvsi to probed item, relative to probe, right visual channels')
ax = fig.subplots(4,1)
for i in range(len(ax)):
    ax[i].contourf(times, freqs, np.nanmean(deepcopy(dtpside_data[i]).pick_channels(visrightchans).data,0), levels=100, cmap ='RdBu_r', vmin = vmin_vis, vmax = vmax_vis)
    ax[i].vlines([0.0, vlines_dt[i]], ymin=1, ymax=39, lw = .5, linestyle = '--')
    ax[i].set_title(titles[i])
for mask in masks_dtpside_cued:
    bigmask = np.kron(mask, np.ones((10,10)))
    ax[2].contour(bigmask, colors='black', lw=.5, extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), linewidths = .5, corner_mask=False, antialiased = False)




x_dtpside_cvsn = np.empty(shape = (subs.size, freqs.size, times.size))#deepcopy(gave_dtpside).crop(tmin=-1,tmax=0).times.size))
for i in range(subs.size):
    x_dtpside_cvsn[i,:,:] = np.nanmean(deepcopy(alldata_dtpside_cvsn_t[i]).pick_channels(visrightchans).data,0)

t_dtpside_cvsn, clusters_dtpside_cvsn, cluster_pv_dtpside_cvsn, _ = mne.stats.permutation_cluster_1samp_test(x_dtpside_cvsn, n_permutations='all')
masks_dtpside_cvsn = np.asarray(clusters_dtpside_cvsn)[cluster_pv_dtpside_cvsn<0.05]

fig = plt.figure()
fig.suptitle('DT ~ cvsi to probed item, relative to probe, right visual channels')
ax = fig.subplots(4,1)
for i in range(len(ax)):
    ax[i].contourf(times, freqs, np.nanmean(deepcopy(dtpside_data[i]).pick_channels(visrightchans).data,0), levels=100, cmap ='RdBu_r', vmin = vmin_vis, vmax = vmax_vis)
    ax[i].vlines([0.0, vlines_dt[i]], ymin=1, ymax=39, lw = .5, linestyle = '--')
    ax[i].set_title(titles[i])
for mask in masks_dtpside_cvsn:
    bigmask = np.kron(mask, np.ones((10,10)))
    ax[3].contour(bigmask, colors='black', lw=.5, extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), linewidths = .5, corner_mask=False, antialiased = False)

#for cvsn clusters
times_sig_cvsn = []
freqs_sig_cvsn = []
for mask in masks_dtpside_cvsn:
    times_sig_cvsn.append(times[np.where(mask == True)[1]])
    freqs_sig_cvsn.append(freqs[np.where(mask == True)[0]])
#gave_pside_cued.plot_topomap(tmin = np.min(times_sig_cued))
    

for cluster in range(len(times_sig_cvsn)):
    min_alpha = np.where(freqs_sig_cvsn[cluster]>=8)
    sigtimes_cvsn = times_sig_cvsn[cluster][min_alpha]
    sigfreqs_cvsn = freqs_sig_cvsn[cluster][min_alpha]
    tmin, tmax = np.min(sigtimes_cvsn[cluster]), np.max(sigtimes_cvsn[cluster])
    fmin, fmax = np.min(sigfreqs_cvsn[cluster]), np.max(sigfreqs_cvsn[cluster])
    
    fig = plt.figure()
    fig.suptitle('topomaps of significant cluster for DT ~ probed side, 8-12Hz')
    ax = fig.subplots(3,2)
    deepcopy(gave_pside_cvsn).pick_channels(channames_no_midline).plot_topomap(tmin=tmin, tmax=tmax, fmin=8, fmax=12, vmin=-3, vmax=3, contours=0, outlines='head', axes = ax[0,0], colorbar = False)
    deepcopy(gave_pside_neut).pick_channels(channames_no_midline).plot_topomap(tmin=tmin, tmax=tmax, fmin=8, fmax=12, vmin=-3, vmax=3, contours=0, outlines='head', axes = ax[1,0], colorbar = False)
    deepcopy(gave_pside_cued).pick_channels(channames_no_midline).plot_topomap(tmin=tmin, tmax=tmax, fmin=8, fmax=12, vmin=-3, vmax=3, contours=0, outlines='head', axes = ax[2,0], colorbar = False)
    ax[0,0].set_title('cvsn')
    ax[1,0].set_title('neutral trials')
    ax[2,0].set_title('cued trials')
    fig = plt.figure()

    deepcopy(gave_pside_cvsn).pick_channels(channames_no_midline).plot_topomap(tmin=tmin, tmax=tmax, fmin=13, fmax=20, vmin=-3, vmax=3, contours=0, outlines='head', axes = ax[0,1], colorbar = False)
    deepcopy(gave_pside_neut).pick_channels(channames_no_midline).plot_topomap(tmin=tmin, tmax=tmax, fmin=13, fmax=20, vmin=-3, vmax=3, contours=0, outlines='head', axes = ax[1,1], colorbar = False)
    deepcopy(gave_pside_cued).pick_channels(channames_no_midline).plot_topomap(tmin=tmin, tmax=tmax, fmin=13, fmax=20, vmin=-3, vmax=3, contours=0, outlines='head', axes = ax[2,1])
    ax[0,1].set_title('cvsn')
    ax[1,1].set_title('neutral trials')
    ax[2,1].set_title('cued trials')
    
    
#%%
    
#confidence ~ cvsi relative to the probed item
    










