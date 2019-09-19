#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:49:46 2019

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
alldata_pside_nvsc    = []
alldata_errorpside_neut = []
alldata_errorpside_cued = []
alldata_confpside_neut  = []
alldata_confpside_cued  = []
alldata_dtpside_neut    = []
alldata_dtpside_cued    = []
alldata_errorpside_cvsn = []
alldata_confpside_cvsn  = []
alldata_dtpside_cvsn    = []
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
alldata_pside_nvsc_t    = []
alldata_errorpside_neut_t = []
alldata_errorpside_cued_t = []
alldata_confpside_neut_t  = []
alldata_confpside_cued_t  = []
alldata_dtpside_neut_t    = []
alldata_dtpside_cued_t    = []
alldata_errorpside_cvsn_t = []
alldata_confpside_cvsn_t  = []
alldata_dtpside_cvsn_t    = []
alldata_errorpside_t      = []
alldata_dtpside_t         = []
alldata_confpside_t       = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    alldata_grandmean.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_grandmean_betas-tfr.h5'))[0])
    alldata_cued.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_cued_betas-tfr.h5'))[0])
    alldata_neut.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_neutral_betas-tfr.h5'))[0])
    alldata_pside.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_pside_betas-tfr.h5'))[0])
    alldata_pside_neut.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_psideneutral_betas-tfr.h5'))[0])
    alldata_pside_cued.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_psidecued_betas-tfr.h5'))[0])
    alldata_pside_cvsn.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_pside_cvsn_betas-tfr.h5'))[0])
    alldata_pside_nvsc.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_pside_nvsc_betas-tfr.h5'))[0])    
    alldata_errorpside_neut.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_errorxpside_neutral_betas-tfr.h5'))[0])
    alldata_errorpside_cued.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_errorxpside_cued_betas-tfr.h5'))[0])
    alldata_confpside_neut.append(mne.time_frequency.read_tfrs(fname    = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_confxpside_neutral_betas-tfr.h5'))[0])
    alldata_confpside_cued.append(mne.time_frequency.read_tfrs(fname    = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_confxpside_cued_betas-tfr.h5'))[0])
    alldata_dtpside_neut.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_DTxpside_neutral_betas-tfr.h5'))[0])
    alldata_dtpside_cued.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_DTxpside_cued_betas-tfr.h5'))[0])
    alldata_errorpside_cvsn.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_errorxpsidexcvsn_betas-tfr.h5'))[0])
    alldata_confpside_cvsn.append(mne.time_frequency.read_tfrs(fname    = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_confxpsidexcvsn_betas-tfr.h5'))[0])
    alldata_dtpside_cvsn.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_DTxpsidexcvsn_betas-tfr.h5'))[0])
    alldata_errorpside.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_errorxpside_betas-tfr.h5'))[0])
    alldata_dtpside.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_DTxpside_betas-tfr.h5'))[0])
    alldata_confpside.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_confxpside_betas-tfr.h5'))[0])
    
    #tstats
    alldata_grandmean_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_grandmean_tstats-tfr.h5'))[0])
    alldata_cued_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_cued_tstats-tfr.h5'))[0])
    alldata_neut_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_neutral_tstats-tfr.h5'))[0])
    alldata_pside_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_pside_tstats-tfr.h5'))[0])
    alldata_pside_neut_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_psideneutral_tstats-tfr.h5'))[0])
    alldata_pside_cued_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_psidecued_tstats-tfr.h5'))[0])
    alldata_pside_cvsn_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_pside_cvsn_tstats-tfr.h5'))[0])
    alldata_pside_nvsc_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_pside_nvsc_tstats-tfr.h5'))[0])    
    alldata_errorpside_neut_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_errorxpside_neutral_tstats-tfr.h5'))[0])
    alldata_errorpside_cued_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_errorxpside_cued_tstats-tfr.h5'))[0])
    alldata_confpside_neut_t.append(mne.time_frequency.read_tfrs(fname    = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_confxpside_neutral_tstats-tfr.h5'))[0])
    alldata_confpside_cued_t.append(mne.time_frequency.read_tfrs(fname    = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_confxpside_cued_tstats-tfr.h5'))[0])
    alldata_dtpside_neut_t.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_DTxpside_neutral_tstats-tfr.h5'))[0])
    alldata_dtpside_cued_t.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_DTxpside_cued_tstats-tfr.h5'))[0])
    alldata_errorpside_cvsn_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_errorxpsidexcvsn_tstats-tfr.h5'))[0])
    alldata_confpside_cvsn_t.append(mne.time_frequency.read_tfrs(fname    = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_confxpsidexcvsn_tstats-tfr.h5'))[0])
    alldata_dtpside_cvsn_t.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_DTxpsidexcvsn_tstats-tfr.h5'))[0])
    alldata_errorpside_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_errorxpside_tstats-tfr.h5'))[0])
    alldata_dtpside_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_DTxpside_tstats-tfr.h5'))[0])
    alldata_confpside_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_confxpside_tstats-tfr.h5'))[0])
    

#%%
#get some behavioural data that we should probably plot to make some sense of things
allbdata = []
for i in subs:
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    bdata = pd.read_csv(param['resplocked_tfr_meta'], index_col = 0, header = 0) #read the metadata associated with this subject
    allbdata.append(bdata)

DTs = []
for i in allbdata:
    DTs.append(np.nanmean(i.DT))
gave_mean_dt = np.nanmean(DTs)


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
    
    
gave_pside_cued = mne.grand_average(alldata_pside_cued_t); gave_pside_cued.data = toverparam(alldata_pside_cued);
#gave_pside_cued.drop_channels(['RM'])
#gave_pside_cued.plot_joint(title = 'resp locked, right minus left, cued trials, average betas',
#                           timefreqs = timefreqs_alpha, picks = chans_no_midline,
#                           topomap_args = dict(outlines = 'head', contours = 0))
#gave_pside_cued.plot_joint(title = 'resp locked, right minus left, cued trials, t over tstats',
#                           timefreqs = timefreqs_beta, picks = chans_no_midline,
#                           topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_pside_neut = mne.grand_average(alldata_pside_neut_t); gave_pside_neut.data = toverparam(alldata_pside_neut);
#gave_pside_neut.drop_channels(['RM'])
#gave_pside_neut.plot_joint(title = 'resp locked, right minus left, neutraltrials, average betas',
#                           timefreqs = timefreqs_alpha, picks = chans_no_midline,
#                           topomap_args = dict(outlines = 'head', contours = 0))
#gave_pside_neut.plot_joint(title = 'resp locked, right minus left, neutraltrials, t over tstats',
#                           timefreqs = timefreqs_beta, picks = chans_no_midline,
#                           topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_pside_cvsn = mne.grand_average(alldata_pside_cvsn_t); gave_pside_cvsn.data = toverparam(alldata_pside_cvsn);
#gave_pside_cvsn.drop_channels(['RM'])
#gave_pside_cvsn.plot_joint(title = 'resp locked, right minus left, cued vs neutral, average betas',
#                           timefreqs = timefreqs_alpha, picks = chans_no_midline,
#                           topomap_args = dict(outlines = 'head', contours = 0))
#gave_pside_cvsn.plot_joint(title = 'resp locked, right minus left, cued vs neutral, t over tstats',
#                           timefreqs = timefreqs_beta, picks = chans_no_midline,
#                           topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))
#gave_pside_cvsn.plot(picks='C4', 
#                     vmin = np.divide(np.min(deepcopy(gave_pside_cvsn).pick_channels(['C4']).data), 5),
#                     vmax = np.divide(np.min(deepcopy(gave_pside_cvsn).pick_channels(['C4']).data), -5))
#gave_pside_cvsn.plot(picks=visrightchans, 
#                     vmin = np.divide(np.min(deepcopy(gave_pside_cvsn).pick_channels(visrightchans).data), 5),
#                     vmax = np.divide(np.min(deepcopy(gave_pside_cvsn).pick_channels(visrightchans).data), -5))

gave_pside = mne.grand_average(alldata_pside_t); gave_pside.data = toverparam(alldata_pside_t)


vmin = np.divide(np.min(np.nanmean(deepcopy(gave_pside_cued).pick_channels(visrightchans).data,0)), 5)
vmax = np.divide(np.min(np.nanmean(deepcopy(gave_pside_cued).pick_channels(visrightchans).data,0)), -5)

vmin, vmax = -2, 2
#vmin, vmax = -1, 1
times, freqs = gave_pside_cued.times, gave_pside_cued.freqs

fig = plt.figure()
fig.suptitle('right motor channels: C2 & C4')
ax = fig.subplots(4,1)
ax[0].contourf( times, freqs, np.nanmean(deepcopy(gave_pside_cued).pick_channels(['C4', 'C2']).data, 0), vmin = vmin, vmax = vmax, levels = 80, cmap = 'RdBu_r'); #ax[0].colorbar()
ax[1].contourf( times, freqs, np.nanmean(deepcopy(gave_pside_neut).pick_channels(['C4', 'C2']).data, 0), vmin = vmin, vmax = vmax, levels = 80, cmap = 'RdBu_r'); #ax[0].colorbar()
ax[2].contourf( times, freqs, np.nanmean(deepcopy(gave_pside_cvsn).pick_channels(['C4', 'C2']).data, 0), vmin = vmin, vmax = vmax, levels = 80, cmap = 'RdBu_r'); #ax[0].colorbar()
ax[3].contourf( times, freqs, np.nanmean(deepcopy(gave_pside).pick_channels(['C4', 'C2']).data, 0),      vmin = vmin, vmax = vmax, levels = 80, cmap = 'RdBu_r');
ax[0].set_title('cued trials')
ax[1].set_title('neutral trials')
ax[2].set_title('cued vs neutral')
ax[3].set_title('all trials')
for axis in ax:
    axis.vlines(0.0, ymin=1, ymax=39, linestyle = '--', lw = .5)
    
fig = plt.figure()
fig.suptitle('right visual channels: PO8, PO4, O2')
ax = fig.subplots(4,1)
ax[0].contourf( times, freqs, np.nanmean(deepcopy(gave_pside_cued).pick_channels(visrightchans).data, 0), vmin = vmin, vmax = vmax, levels = 80, cmap = 'RdBu_r'); #ax[0].colorbar()
ax[1].contourf( times, freqs, np.nanmean(deepcopy(gave_pside_neut).pick_channels(visrightchans).data, 0), vmin = vmin, vmax = vmax, levels = 80, cmap = 'RdBu_r'); #ax[0].colorbar()
ax[2].contourf( times, freqs, np.nanmean(deepcopy(gave_pside_cvsn).pick_channels(visrightchans).data, 0), vmin = vmin, vmax = vmax, levels = 80, cmap = 'RdBu_r'); #ax[0].colorbar()
ax[3].contourf( times, freqs, np.nanmean(deepcopy(gave_pside).pick_channels(visrightchans).data, 0),      vmin = vmin, vmax = vmax, levels = 80, cmap = 'RdBu_r');
ax[0].set_title('cued trials')
ax[1].set_title('neutral trials')
ax[2].set_title('cued vs neutral')
ax[3].set_title('all trials')
for axis in ax:
    axis.vlines(0.0, ymin=1, ymax=39, linestyle = '--', lw = .5)
#%% MOTOR CHANNELS

#permutation cluster test over motor channels for all trials, rel to probe (spacebar press)
x_dat = np.empty(shape = (subs.size, gave_pside_cued.freqs.size, gave_pside_cued.times.size))
for i in range(subs.size):
    x_dat[i,:,:] = np.nanmean(deepcopy(alldata_pside[i]).pick_channels(['C4', 'C2']).data, 0)

t_alltrls_mot, clusters_alltrls_mot, clusters_pv_alltrls_mot , _ = mne.stats.permutation_cluster_1samp_test(x_dat, n_permutations='all')

mask_alltrls_mot = np.squeeze(np.asarray(clusters_alltrls_mot)[clusters_pv_alltrls_mot<.4])
bigmask_alltrls_mot = np.kron(mask_alltrls_mot, np.ones((10,10)))

#plot motor channels for cued, neutral, all trials and the diff between cued and neutral
fig = plt.figure()
ax = fig.subplots(4,1)
fig.suptitle('cvsi to probed left vs right, all trials, right motor channels (C2, C4)')
ax[0].contourf( times, freqs, np.nanmean(deepcopy(gave_pside_cued).pick_channels(motrightchans).data, 0), vmin = vmin, vmax = vmax, levels = 100, cmap = 'RdBu_r'); #ax[0].colorbar()
ax[1].contourf( times, freqs, np.nanmean(deepcopy(gave_pside_neut).pick_channels(motrightchans).data, 0), vmin = vmin, vmax = vmax, levels = 100, cmap = 'RdBu_r'); #ax[0].colorbar()
ax[2].contourf( times, freqs, np.nanmean(deepcopy(gave_pside).pick_channels(motrightchans).data, 0),      vmin = vmin, vmax = vmax, levels = 100, cmap = 'RdBu_r')
ax[3].contourf( times, freqs, np.nanmean(deepcopy(gave_pside_cvsn).pick_channels(motrightchans).data, 0), vmin = vmin, vmax = vmax, levels = 100, cmap = 'RdBu_r'); #ax[0].colorbar()
ax[0].set_title('cued trials')
ax[1].set_title('neutral trials')
ax[2].set_title('all trials')
ax[3].set_title('cued vs neutral')
for axis in ax:
    axis.vlines(0.0, ymin=1, ymax=39, linestyle = '--', lw = .5)
    axis.vlines(-gave_mean_dt, ymin = 1, ymax = 39, linestyle = '--', lw = .5)

ax[2].contour(bigmask_alltrls_mot, colors = 'black', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), linewidths = .5, corner_mask=False, antialiased = False)

#get time and frequencies for this significant cluster, and plot the topography of it
times_sig  = times[np.where(mask_alltrls_mot == True)[1]]
freqs_sig  = freqs[np.where(mask_alltrls_mot == True)[0]]
deepcopy(gave_pside).pick_channels(channames_no_midline).plot_topomap(tmin = np.min(times_sig), tmax = np.max(times_sig), fmin = np.min(freqs_sig), fmax = np.max(freqs_sig), cmap = 'RdBu_r',
         vmin = -2, vmax = 2, contours = 0)

#%% VISUAL CHANNELS


#permutation cluster test over motor channels for all trials, rel to probe (spacebar press)
x_dat = np.empty(shape = (subs.size, gave_pside_cued.freqs.size, gave_pside_cued.times.size))
for i in range(subs.size):
    x_dat[i,:,:] = np.nanmean(deepcopy(alldata_pside[i]).pick_channels(visrightchans).data, 0)

t_alltrls_vis, clusters_alltrls_vis, clusters_pv_alltrls_vis , _ = mne.stats.permutation_cluster_1samp_test(x_dat, n_permutations='all')

mask_alltrls_vis = np.squeeze(np.asarray(clusters_alltrls_vis)[clusters_pv_alltrls_vis<.06])
bigmask_alltrls_vis = np.kron(mask_alltrls_vis, np.ones((10,10)))

#plot motor channels for cued, neutral, all trials and the diff between cued and neutral
fig = plt.figure()
ax = fig.subplots(4,1)
fig.suptitle('cvsi to probed left vs right, all trials, right visual channels (PO8, PO4, O2)')
ax[0].contourf( times, freqs, np.nanmean(deepcopy(gave_pside_cued).pick_channels(visrightchans).data, 0), vmin = vmin, vmax = vmax, levels = 100, cmap = 'RdBu_r'); #ax[0].colorbar()
ax[1].contourf( times, freqs, np.nanmean(deepcopy(gave_pside_neut).pick_channels(visrightchans).data, 0), vmin = vmin, vmax = vmax, levels = 100, cmap = 'RdBu_r'); #ax[0].colorbar()
ax[2].contourf( times, freqs, np.nanmean(deepcopy(gave_pside).pick_channels(visrightchans).data, 0),      vmin = vmin, vmax = vmax, levels = 100, cmap = 'RdBu_r')
ax[3].contourf( times, freqs, np.nanmean(deepcopy(gave_pside_cvsn).pick_channels(visrightchans).data, 0), vmin = vmin, vmax = vmax, levels = 100, cmap = 'RdBu_r'); #ax[0].colorbar()
ax[0].set_title('cued trials')
ax[1].set_title('neutral trials')
ax[2].set_title('all trials')
ax[3].set_title('cued vs neutral')
for axis in ax:
    axis.vlines(0.0, ymin=1, ymax=39, linestyle = '--', lw = .5)
    axis.vlines(-gave_mean_dt, ymin = 1, ymax = 39, linestyle = '--', lw = .5)
ax[2].contour(bigmask_alltrls_vis, colors = 'black', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), linewidths = .5, corner_mask=False, antialiased = False)

#get time and frequencies for this significant cluster, and plot the topography of it
times_sig  = times[np.where(mask_alltrls_vis == True)[1]]
freqs_sig  = freqs[np.where(mask_alltrls_vis == True)[0]]
deepcopy(gave_pside).pick_channels(channames_no_midline).plot_topomap(tmin = np.min(times_sig), tmax = np.max(times_sig), fmin = np.min(freqs_sig), fmax = np.max(freqs_sig), cmap = 'RdBu_r',
         vmin = -2, vmax = 2, contours = 0)

#%% ERROR AS A FUNCTION OF LATERALISED RETRIEVAL OF THE PROBED ITEM

gave_errorpside      = mne.grand_average(alldata_errorpside_t);      gave_errorpside.data      = toverparam(alldata_errorpside_t)
gave_errorpside_neut = mne.grand_average(alldata_errorpside_neut_t); gave_errorpside_neut.data = toverparam(alldata_errorpside_neut_t)
gave_errorpside_cued = mne.grand_average(alldata_errorpside_cued_t); gave_errorpside_cued.data = toverparam(alldata_errorpside_cued_t)
gave_errorpside_cvsn = mne.grand_average(alldata_errorpside_cvsn_t); gave_errorpside_cvsn.data = toverparam(alldata_errorpside_cvsn_t)

vmin_vis = np.divide(np.min(np.nanmean(deepcopy(gave_errorpside).pick_channels(visrightchans).data, 0)),  1); vmin_vis = -2
vmax_vis = np.divide(np.min(np.nanmean(deepcopy(gave_errorpside).pick_channels(visrightchans).data, 0)), -1); vmax_vis =  2

vmin_mot = np.divide(np.min(np.nanmean(deepcopy(gave_pside_cued).pick_channels(motrightchans).data,0)), 5);  vmin_mot = -2
vmax_mot = np.divide(np.min(np.nanmean(deepcopy(gave_pside_cued).pick_channels(motrightchans).data,0)), -5); vmax_mot = 2

fig = plt.figure()
fig.suptitle('error, contra vs ipsi to the probed item, right visual channels')
ax = fig.subplots(4,1)
ax[0].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[1].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_neut).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[2].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_cued).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[3].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_cvsn).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[0].set_title('all trials')
ax[1].set_title('neutral trials')
ax[2].set_title('cued trials')
ax[3].set_title('cued vs neutral')
for axis in ax:
    axis.vlines([0.0, -gave_mean_dt], ymin = 1, ymax=39, linestyle = '--', lw=.5)


fig = plt.figure()
fig.suptitle('error, contra vs ipsi to the probed item, right motor channels')
ax = fig.subplots(4,1)
ax[0].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside).pick_channels(motrightchans).data, 0), vmin = vmin_mot, vmax = vmax_mot, levels = 100, cmap = 'RdBu_r')
ax[1].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_neut).pick_channels(motrightchans).data, 0), vmin = vmin_mot, vmax = vmax_mot, levels = 100, cmap = 'RdBu_r')
ax[2].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_cued).pick_channels(motrightchans).data, 0), vmin = vmin_mot, vmax = vmax_mot, levels = 100, cmap = 'RdBu_r')
ax[3].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_cvsn).pick_channels(motrightchans).data, 0), vmin = vmin_mot, vmax = vmax_mot, levels = 100, cmap = 'RdBu_r')
ax[0].set_title('all trials')
ax[1].set_title('neutral trials')
ax[2].set_title('cued trials')
ax[3].set_title('cued vs neutral')
for axis in ax:
    axis.vlines([0.0, -gave_mean_dt], ymin = 1, ymax=39, linestyle = '--', lw=.5)

x_errorpside_cvsn_mot = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_errorpside_cvsn_mot[i,:,:] = np.nanmean(deepcopy(alldata_errorpside_cvsn_t[i]).pick_channels(motrightchans).data, 0)

t_errorpside_cvsn_mot, clusters_errorpside_cvsn_mot, clusters_pv_errorpside_cvsn_mot, _ = mne.stats.permutation_cluster_1samp_test(x_errorpside_cvsn_mot, n_permutations='all')
#nothing significant, so no statistical difference between trials

#is there anything sig diff to 0 across all trials?

x_errorpside_alltrls_mot = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_errorpside_alltrls_mot[i,:,:] = np.nanmean(deepcopy(alldata_errorpside_t[i]).pick_channels(motrightchans).data, 0)

t_errorpside_alltrls_mot, clusters_errorpside_alltrls_mot, clusters_pv_errorpside_alltrls_mot, _ = mne.stats.permutation_cluster_1samp_test(x_errorpside_alltrls_mot, n_permutations='all')
#nope nothing in all trials either


#and now across visual channels?
x_errorpside_cvsn_vis = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_errorpside_cvsn_vis[i,:,:] = np.nanmean(deepcopy(alldata_errorpside_cvsn_t[i]).pick_channels(visrightchans).data, 0)

t_errorpside_cvsn_vis, clusters_errorpside_cvsn_vis, clusters_pv_errorpside_cvsn_vis, _ = mne.stats.permutation_cluster_1samp_test(x_errorpside_cvsn_vis, n_permutations='all')
#nothing significant, so no statistical difference between trials in lateralisation of visual channels

#is there anything sig diff to 0 across all trials?

x_errorpside_alltrls_vis = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_errorpside_alltrls_vis[i,:,:] = np.nanmean(deepcopy(alldata_errorpside_t[i]).pick_channels(visrightchans).data, 0)

t_errorpside_alltrls_vis, clusters_errorpside_alltrls_vis, clusters_pv_errorpside_alltrls_vis, _ = mne.stats.permutation_cluster_1samp_test(x_errorpside_alltrls_vis, n_permutations='all')
#nope nothing in all trials either in terms of lateralisation of visual channels


#%% reaction time (DT) as a function of probed side and cue?

gave_dtpside = mne.grand_average(alldata_dtpside_t); gave_dtpside.data = toverparam(alldata_dtpside_t)
gave_dtpside_neut = mne.grand_average(alldata_dtpside_neut_t); gave_dtpside_neut.data = toverparam(alldata_dtpside_neut_t)
gave_dtpside_cued = mne.grand_average(alldata_dtpside_cued_t); gave_dtpside_cued.data = toverparam(alldata_dtpside_cued_t)
gave_dtpside_cvsn = mne.grand_average(alldata_dtpside_cvsn_t); gave_dtpside_cvsn.data = toverparam(alldata_dtpside_cvsn_t)


vmin_vis = np.divide(np.min(np.nanmean(deepcopy(gave_dtpside).pick_channels(visrightchans).data, 0)),  1); vmin_vis = -2
vmax_vis = np.divide(np.min(np.nanmean(deepcopy(gave_dtpside).pick_channels(visrightchans).data, 0)), -1); vmax_vis =  2

vmin_mot = np.divide(np.min(np.nanmean(deepcopy(gave_dtpside).pick_channels(motrightchans).data,0)), 5);  vmin_mot = -2
vmax_mot = np.divide(np.min(np.nanmean(deepcopy(gave_dtpside).pick_channels(motrightchans).data,0)), -5); vmax_mot = 2

fig = plt.figure()
fig.suptitle('DT, contra vs ipsi to the probed item, right visual channels')
ax = fig.subplots(4,1)
ax[0].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[1].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_neut).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[2].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_cued).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[3].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_cvsn).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[0].set_title('all trials')
ax[1].set_title('neutral trials')
ax[2].set_title('cued trials')
ax[3].set_title('cued vs neutral')
for axis in ax:
    axis.vlines([0.0, -gave_mean_dt], ymin = 1, ymax=39, linestyle = '--', lw=.5)


fig = plt.figure()
fig.suptitle('DT, contra vs ipsi to the probed item, right motor channels')
ax = fig.subplots(4,1)
ax[0].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside).pick_channels(motrightchans).data, 0), vmin = vmin_mot, vmax = vmax_mot, levels = 100, cmap = 'RdBu_r')
ax[1].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_neut).pick_channels(motrightchans).data, 0), vmin = vmin_mot, vmax = vmax_mot, levels = 100, cmap = 'RdBu_r')
ax[2].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_cued).pick_channels(motrightchans).data, 0), vmin = vmin_mot, vmax = vmax_mot, levels = 100, cmap = 'RdBu_r')
ax[3].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_cvsn).pick_channels(motrightchans).data, 0), vmin = vmin_mot, vmax = vmax_mot, levels = 100, cmap = 'RdBu_r')
ax[0].set_title('all trials')
ax[1].set_title('neutral trials')
ax[2].set_title('cued trials')
ax[3].set_title('cued vs neutral')
for axis in ax:
    axis.vlines([0.0, -gave_mean_dt], ymin = 1, ymax=39, linestyle = '--', lw=.5)

x_dtpside_all_vis = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_dtpside_all_vis[i,:,:] = np.nanmean(deepcopy(gave_dtpside).pick_channels(visrightchans).data, 0)

t_dtpside_all_vis, clusters_dtpside_all_vis, clusters_pv_dtpside_all_vis, _ = mne.stats.permutation_cluster_1samp_test(x_dtpside_all_vis, n_permutations='all')
masks_alltrls_vis = np.squeeze(np.asarray(clusters_dtpside_all_vis)[clusters_pv_dtpside_all_vis<0.05])

fig = plt.figure()
fig.suptitle('DT, contra vs ipsi to the probed item, right visual channels')
ax = fig.subplots(4,1)
ax[0].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[1].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_neut).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[2].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_cued).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[3].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_cvsn).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[0].set_title('all trials')
ax[1].set_title('neutral trials')
ax[2].set_title('cued trials')
ax[3].set_title('cued vs neutral')
for axis in ax:
    axis.vlines([0.0, -gave_mean_dt], ymin = 1, ymax=39, linestyle = '--', lw=.5)
for mask in masks_alltrls_vis:
    ibigmask = np.kron(mask, np.ones((10,10)))
    ax[0].contour(ibigmask, colors='black', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), linewidths = .5, corner_mask=False, antialiased = False)



x_dtpside_cvsn_vis = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_dtpside_cvsn_vis[i,:,:] = np.nanmean(deepcopy(gave_dtpside_cvsn).pick_channels(visrightchans).data, 0)

t_dtpside_cvsn_vis, clusters_dtpside_cvsn_vis, clusters_pv_dtpside_cvsn_vis, _ = mne.stats.permutation_cluster_1samp_test(x_dtpside_cvsn_vis, n_permutations='all')
masks_cvsn_vis = np.squeeze(np.asarray(clusters_dtpside_cvsn_vis)[clusters_pv_dtpside_cvsn_vis<0.05])

fig = plt.figure()
fig.suptitle('DT, contra vs ipsi to the probed item, right visual channels')
ax = fig.subplots(4,1)
ax[0].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[1].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_neut).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[2].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_cued).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[3].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_cvsn).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[0].set_title('all trials')
ax[1].set_title('neutral trials')
ax[2].set_title('cued trials')
ax[3].set_title('cued vs neutral')
for axis in ax:
    axis.vlines([0.0, -gave_mean_dt], ymin = 1, ymax=39, linestyle = '--', lw=.5)
for mask in masks_cvsn_vis:
    ibigmask = np.kron(mask, np.ones((10,10)))
    ax[3].contour(ibigmask, colors='black', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), linewidths = .5, corner_mask=False, antialiased = False)

gave_dtpside_cvsn.plot_joint(topomap_args = dict(outlines='head', contours = 0, vmin=-2, vmax=2), picks=chans_no_midline, timefreqs=timefreqs_alpha)

#%% Confidence as a function of lateralisation to item side & previously cued?


gave_confpside      = mne.grand_average(alldata_confpside_t);      gave_confpside.data = toverparam(alldata_confpside_t)
gave_confpside_neut = mne.grand_average(alldata_confpside_neut_t); gave_confpside_neut.data = toverparam(alldata_confpside_neut_t)
gave_confpside_cued = mne.grand_average(alldata_confpside_cued_t); gave_confpside_cued.data = toverparam(alldata_confpside_cued_t)
gave_confpside_cvsn = mne.grand_average(alldata_confpside_cvsn_t); gave_confpside_cvsn.data = toverparam(alldata_confpside_cvsn_t) 

vmin_vis = np.divide(np.min(np.nanmean(deepcopy(gave_confpside).pick_channels(visrightchans).data, 0)),  1); vmin_vis = -2
vmax_vis = np.divide(np.min(np.nanmean(deepcopy(gave_confpside).pick_channels(visrightchans).data, 0)), -1); vmax_vis =  2

vmin_mot = np.divide(np.min(np.nanmean(deepcopy(gave_confpside).pick_channels(motrightchans).data,0)), 5);  vmin_mot = -2
vmax_mot = np.divide(np.min(np.nanmean(deepcopy(gave_confpside).pick_channels(motrightchans).data,0)), -5); vmax_mot = 2

fig = plt.figure()
fig.suptitle('confidence ~ contra vs ipsi to the probed item, right visual channels, rel to space press')
ax = fig.subplots(4,1)
ax[0].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[1].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside_neut).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[2].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside_cued).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[3].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside_cvsn).pick_channels(visrightchans).data, 0), vmin = vmin_vis, vmax = vmax_vis, levels = 100, cmap = 'RdBu_r')
ax[0].set_title('all trials')
ax[1].set_title('neutral trials')
ax[2].set_title('cued trials')
ax[3].set_title('cued vs neutral')
for axis in ax:
    axis.vlines([0.0, -gave_mean_dt], ymin = 1, ymax=39, linestyle = '--', lw=.5)


fig = plt.figure()
fig.suptitle('confidence, contra vs ipsi to the probed item, right motor channels')
ax = fig.subplots(4,1)
ax[0].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside).pick_channels(motrightchans).data, 0), vmin = vmin_mot, vmax = vmax_mot, levels = 100, cmap = 'RdBu_r')
ax[1].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside_neut).pick_channels(motrightchans).data, 0), vmin = vmin_mot, vmax = vmax_mot, levels = 100, cmap = 'RdBu_r')
ax[2].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside_cued).pick_channels(motrightchans).data, 0), vmin = vmin_mot, vmax = vmax_mot, levels = 100, cmap = 'RdBu_r')
ax[3].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside_cvsn).pick_channels(motrightchans).data, 0), vmin = vmin_mot, vmax = vmax_mot, levels = 100, cmap = 'RdBu_r')
ax[0].set_title('all trials')
ax[1].set_title('neutral trials')
ax[2].set_title('cued trials')
ax[3].set_title('cued vs neutral')
for axis in ax:
    axis.vlines([0.0, -gave_mean_dt], ymin = 1, ymax=39, linestyle = '--', lw=.5)























#%%
#error as a function of pside
gave_errorpside_neut = mne.grand_average(alldata_errorpside_neut_t); gave_errorpside_neut.data = toverparam(alldata_errorpside_neut_t)
gave_errorpside_neut.drop_channels(['RM'])
gave_errorpside_neut.plot_joint(title = 'error x probed side, neutral trials, t over tstat',
                                timefreqs = timefreqs_alpha, picks = chans_no_midline,
                                topomap_args = dict(outlines= 'head', contours = 0, vmin=-2, vmax=2))
gave_errorpside_neut.plot(picks = visrightchans, vmin = -2, vmax = 2, combine = 'mean')
gave_errorpside_neut.plot(picks = 'C4', vmin = -2, vmax = 2 )

gave_errorpside_cued = mne.grand_average(alldata_errorpside_cued_t); gave_errorpside_cued.data = toverparam(alldata_errorpside_cued_t)
gave_errorpside_cued.drop_channels(['RM'])
gave_errorpside_cued.plot_joint(title = 'error x probed side, cued trials, t over tstat',
                                timefreqs = timefreqs_alpha, picks = chans_no_midline,
                                topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))
gave_errorpside_cued.plot(picks = visrightchans, vmin = -2, vmax = 2, combine = 'mean')
gave_errorpside_cued.plot(picks = 'C4', vmin = -2, vmax = 2 )

gave_errorpside_cvsn = mne.grand_average(alldata_errorpside_cvsn_t); gave_errorpside_cvsn.data = toverparam(alldata_errorpside_cvsn_t)
gave_errorpside_cvsn.drop_channels(['RM'])
gave_errorpside_cvsn.plot_joint(title = 'error x probed left vs right x cued vs neutral, t over tstats',
                                timefreqs = timefreqs_alpha, picks = chans_no_midline,
                                topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

times, freqs = gave_errorpside_neut.times, gave_errorpside_neut.freqs
fig = plt.figure()
fig.suptitle('regressor of error x probed left vs right x cued vs neutral')
ax = fig.subplots(nrows=3, ncols=2)
ax[0,0].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_neut).pick_channels(visrightchans).data, 0), vmin = -2, vmax = 2, levels = 100, cmap = 'RdBu_r')
ax[0,0].vlines(0.0, ymin=1, ymax=39, linestyle = '--', lw=.5); ax[0,0].set_title('neutral trials, visual channels (PO4, PO8, O2), cvsi probed left vs right')

ax[0,1].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_neut).pick_channels(['C4']).data, 0), vmin = -2, vmax = 2, levels = 100, cmap = 'RdBu_r')
ax[0,1].vlines(0.0, ymin=1, ymax=39, linestyle = '--', lw=.5); ax[0,1].set_title('neutral trials, motor channel (C4), cvsi probed left vs right')

ax[1,0].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_cued).pick_channels(visrightchans).data, 0), vmin = -2, vmax = 2, levels = 100, cmap = 'RdBu_r')
ax[1,0].vlines(0.0, ymin=1, ymax=39, linestyle = '--', lw=.5); ax[1,0].set_title('cued trials, visual channels (PO4, PO8, O2), cvsi probed left vs right')

ax[1,1].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_cued).pick_channels(['C4']).data, 0), vmin = -2, vmax = 2, levels = 100, cmap = 'RdBu_r')
ax[1,1].vlines(0.0, ymin=1, ymax=39, linestyle = '--', lw=.5); ax[1,1].set_title('cued trials, motor channel (C4), cvsi probed left vs right')

ax[2,0].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_cvsn).pick_channels(visrightchans).data, 0), vmin = -2, vmax = 2, levels = 100, cmap = 'RdBu_r')
ax[2,0].vlines(0.0, ymin=1, ymax=39, linestyle = '--', lw=.5); ax[2,0].set_title('cvsn trials, visual channels (PO4, PO8, O2), cvsi probed left vs right')

ax[2,1].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_cvsn).pick_channels(['C4']).data, 0), vmin = -2, vmax = 2, levels = 100, cmap = 'RdBu_r')
ax[2,1].vlines(0.0, ymin=1, ymax=39, linestyle = '--', lw=.5); ax[2,1].set_title('cvsn trials, motor channel (C4), cvsi probed left vs right')



#now just collapse across cued and neutral trials as no significant difference between the two, so possibly more powered now

gave_errorpside = mne.grand_average(alldata_errorpside_t); gave_errorpside.data = toverparam(alldata_errorpside_t); gave_errorpside.drop_channels(['RM'])
gave_errorpside.plot_joint(title = 'error x pside, cvsi to retrieved item, all trials, t over tstats', timefreqs = timefreqs_alpha,
                           picks = chans_no_midline,
                           topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

times, freqs = gave_errorpside.times, gave_errorpside.freqs
fig = plt.figure()
ax = fig.subplots(2,1);
ax[0].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside).pick_channels(['C4', 'C2']).data, 0), levels=100, vmin=-2, vmax=2, cmap='RdBu_r')
ax[0].vlines(0.0, ymin=1, ymax=29, linestyle='--', lw = .5); ax[0].set_title('motor channels (C4, C2)')
ax[1].contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside).pick_channels(visrightchans).data,0), levels=100, vmin=-2, vmax=2, cmap = 'RdBu_r')
ax[1].vlines(0, ymin=1, ymax=39, linestyle='--', lw=.5); ax[1].set_title('right visual channels')
fig.suptitle('effect of error, contra vs ipsi to retrieved item')



x_errorpside = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_errorpside[i,:,:] = np.nanmean(deepcopy(alldata_errorpside_t[i]).pick_channels(['C4']).data,0)
t_errorpside, clusters_errorpside, clusters_pv_errorpside, _ = mne.stats.permutation_cluster_1samp_test(x_errorpside, n_permutations='all')

#%%
dat = deepcopy(alldata_errorpside_cvsn_t)
x_dat = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_dat[i,:,:] = np.nanmean(deepcopy(alldata_errorpside_cvsn_t[i]).pick_channels(['C4']).data,0)

t_cvsn, clusters_cvsn, clusters_pv_cvsn, h0_cvsn = mne.stats.permutation_cluster_1samp_test(x_dat, n_permutations = 'all', out_type = 'mask')

x_errorpsideneut = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_errorpsideneut[i,:,:] = np.nanmean(deepcopy(alldata_errorpside_neut_t[i]).pick_channels(['C4']).data,0)

t_errorneut, clusters_errorneut, clusters_pv_errorneut, h0_cvsn = mne.stats.permutation_cluster_1samp_test(x_errorpsideneut, n_permutations='all')

#%%

#DT on pside (and as a function of cue)

gave_dtpside_neut = mne.grand_average(alldata_dtpside_neut_t); gave_dtpside_neut.data = toverparam(alldata_dtpside_neut_t); gave_dtpside_neut.drop_channels(['RM'])
#gave_dtpside_neut.plot_joint(
#        title = 'DT x probed left vs right, neutral trials, t over tstats', timefreqs = timefreqs_alpha, picks = chans_no_midline,
#        topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

fig = plt.figure()
ax = fig.subplots(2,1)
ax[0].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_neut).pick_channels(['C4']).data,0), levels = 100, cmap='RdBu_r', vmin=-2, vmax=2)
ax[1].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_neut).pick_channels(visrightchans).data,0), levels = 100, cmap = 'RdBu_r', vmin = -2, vmax =2)
ax[0].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax[1].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax[0].set_title('motor channel (C4)'); ax[1].set_title('visual channels (PO4, PO8, O2)')
ax[0].set_xlabel('')
fig.suptitle('neutral trials')


gave_dtpside_cued = mne.grand_average(alldata_dtpside_cued_t); gave_dtpside_cued.data = toverparam(alldata_dtpside_cued_t); gave_dtpside_cued.drop_channels(['RM'])
#gave_dtpside_cued.plot_joint(
#        title = 'DT x probed left vs right, cued trials, t over tstats', timefreqs = timefreqs_alpha, picks = chans_no_midline,
#        topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

fig2 = plt.figure()
ax2 = fig2.subplots(2,1)
ax2[0].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_cued).pick_channels(['C4']).data,0), levels = 100, cmap='RdBu_r', vmin=-2, vmax=2)
ax2[1].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_cued).pick_channels(visrightchans).data,0), levels = 100, cmap = 'RdBu_r', vmin = -2, vmax =2)
ax2[0].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax2[1].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax2[0].set_title('motor channel (C4)'); ax2[1].set_title('visual channels (PO4, PO8, O2)')
ax2[0].set_xlabel('')
fig2.suptitle('cued trials')



gave_dtpside_cvsn = mne.grand_average(alldata_dtpside_cvsn_t); gave_dtpside_cvsn.data = toverparam(alldata_dtpside_cvsn_t)
gave_dtpside_cvsn.drop_channels(['RM'])

fig3 = plt.figure()
ax3 = fig3.subplots(2,1)
ax3[0].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_cvsn).pick_channels(['C4']).data,0), levels = 100, cmap='RdBu_r', vmin=-2, vmax=2)
ax3[1].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_cvsn).pick_channels(visrightchans).data,0), levels = 100, cmap = 'RdBu_r', vmin = -2, vmax =2)
ax3[0].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax3[1].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax3[0].set_title('motor channel (C4)'); ax3[1].set_title('visual channels (PO4, PO8, O2)')
ax3[0].set_xlabel('')
fig3.suptitle('cued vs neutral trials')

x_cvsn = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_cvsn[i,:,:] = np.nanmean(deepcopy(alldata_dtpside_cvsn_t[i]).pick_channels(['C4']).data,0)

t_dtcvsn, clusters_dtcvsn, clusters_pv_dtcvsn, h0_dtcvsn = mne.stats.permutation_cluster_1samp_test(x_cvsn, n_permutations='all')

#all trials as no difference in cue conditions

gave_dtpside = mne.grand_average(alldata_dtpside_t); gave_dtpside.data = toverparam(alldata_dtpside_t); gave_dtpside.drop_channels(['RM'])
gave_dtpside.plot_joint(title = 'DT ~ probed left vs right, contra vs ipsi', timefreqs = timefreqs_alpha, picks = chans_no_midline,
                        topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))


fig3 = plt.figure()
ax3 = fig3.subplots(2,1)
ax3[0].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside).pick_channels(['C4']).data,0), levels = 100, cmap='RdBu_r', vmin=-2, vmax=2)
ax3[1].contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside).pick_channels(visrightchans).data,0), levels = 100, cmap = 'RdBu_r', vmin = -2, vmax =2)
ax3[0].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax3[1].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax3[0].set_title('motor channel (C4)'); ax3[1].set_title('visual channels (PO4, PO8, O2)')
ax3[0].set_xlabel('')
fig3.suptitle('DT ~ probed left vs right')

x_dtpside = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_dtpside[i,:,:] = np.nanmean(deepcopy(alldata_dtpside_t[i]).pick_channels(['C4']).data,0)

t_dtpside, clusters_dtpside, clusters_pv_dtpside, h0_dtpside = mne.stats.permutation_cluster_1samp_test(x_dtpside, n_permutations='all')



#%%%

#selection influencing confidence relative to the response phase

gave_confpside_neut = mne.grand_average(alldata_confpside_neut_t); gave_confpside_neut.data = toverparam(alldata_confpside_neut_t); gave_confpside_neut.drop_channels(['RM'])
#gave_confpside_neut.plot_joint(
#        title = 'conf x probed left vs right, neutral trials, t over tstats', timefreqs = timefreqs_alpha, picks = chans_no_midline,
#        topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

fig = plt.figure()
ax = fig.subplots(2,1)
ax[0].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside_neut).pick_channels(['C4']).data,0), levels = 100, cmap='RdBu_r', vmin=-2, vmax=2)
ax[1].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside_neut).pick_channels(visrightchans).data,0), levels = 100, cmap = 'RdBu_r', vmin = -2, vmax =2)
ax[0].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax[1].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax[0].set_title('motor channel (C4)'); ax[1].set_title('visual channels (PO4, PO8, O2)')
ax[0].set_xlabel('')
fig.suptitle('neutral trials')


gave_confpside_cued = mne.grand_average(alldata_confpside_cued_t); gave_confpside_cued.data = toverparam(alldata_confpside_cued_t); gave_confpside_cued.drop_channels(['RM'])
#gave_dtpside_cued.plot_joint(
#        title = 'DT x probed left vs right, cued trials, t over tstats', timefreqs = timefreqs_alpha, picks = chans_no_midline,
#        topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

fig2 = plt.figure()
ax2 = fig2.subplots(2,1)
ax2[0].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside_cued).pick_channels(['C4']).data,0), levels = 100, cmap='RdBu_r', vmin=-2, vmax=2)
ax2[1].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside_cued).pick_channels(visrightchans).data,0), levels = 100, cmap = 'RdBu_r', vmin = -2, vmax =2)
ax2[0].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax2[1].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax2[0].set_title('motor channel (C4)'); ax2[1].set_title('visual channels (PO4, PO8, O2)')
ax2[0].set_xlabel('')
fig2.suptitle('cued trials')



gave_confpside_cvsn = mne.grand_average(alldata_confpside_cvsn_t); gave_confpside_cvsn.data = toverparam(alldata_confpside_cvsn_t)
gave_confpside_cvsn.drop_channels(['RM'])

fig3 = plt.figure()
ax3 = fig3.subplots(2,1)
ax3[0].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside_cvsn).pick_channels(['C4']).data,0), levels = 100, cmap='RdBu_r', vmin=-2, vmax=2)
ax3[1].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside_cvsn).pick_channels(visrightchans).data,0), levels = 100, cmap = 'RdBu_r', vmin = -2, vmax =2)
ax3[0].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax3[1].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax3[0].set_title('motor channel (C4)'); ax3[1].set_title('visual channels (PO4, PO8, O2)')
ax3[0].set_xlabel('')
fig3.suptitle('cued vs neutral trials')

x_cvsn_motor  = np.empty(shape = (subs.size, freqs.size, times.size))
x_cvsn_visual = np.empty(shape = (subs.size, freqs.size, times.size))

for i in range(subs.size):
    x_cvsn_motor[i,:,:] = np.nanmean(deepcopy(alldata_dtpside_cvsn_t[i]).pick_channels(['C4']).data,0)
    x_cvsn_visual[i,:,:] = np.nanmean(deepcopy(alldata_dtpside_cvsn_t[i]).pick_channels(visrightchans).data,0)

t_dtcvsn_mot, clusters_dtcvsn_mot, clusters_pv_dtcvsn_mot, h0_dtcvsn_mot = mne.stats.permutation_cluster_1samp_test(x_cvsn_motor, n_permutations='all')
t_dtcvsn_vis, clusters_dtcvsn_vis, clusters_pv_dtcvsn_vis, h0_dtcvsn_vis = mne.stats.permutation_cluster_1samp_test(x_cvsn_visual, n_permutations='all')

#%% confidence ~ probed side across both cue conditions (as no cue differences in the association with confidence)
gave_confpside = mne.grand_average(alldata_confpside_t); gave_confpside.data =toverparam(alldata_confpside_t); gave_confpside.drop_channels(['RM'])
gave_confpside.plot_joint(timefreqs = timefreqs_alpha, picks = chans_no_midline, title = 'confidence ~ pside(left vs right)',
                          topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax=2))

fig = plt.figure()
ax = fig.subplots(2,1)
ax[0].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside).pick_channels(['C4']).data,0), levels = 100, cmap='RdBu_r', vmin=-2, vmax=2)
ax[1].contourf(times, freqs, np.nanmean(deepcopy(gave_confpside).pick_channels(visrightchans).data,0), levels = 100, cmap = 'RdBu_r', vmin = -2, vmax =2)
ax[0].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax[1].vlines(0.0, ymin=1, ymax=39, linestyle='--', lw=.5)
ax[0].set_title('motor channel (C4)'); ax[1].set_title('visual channels (PO4, PO8, O2)')
ax[0].set_xlabel('')
fig.suptitle('confidence ~ probed side, cvsi')

x_confpside = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_confpside[i,:,:] = np.nanmean(deepcopy(alldata_confpside_t[i]).pick_channels(['C4']).data, 0)

t_confpside, clusters_confpside, clusters_pv_confpside, _ = mne.stats.permutation_cluster_1samp_test(x_confpside, n_permutations='all')
mask_confpside = np.asarray(clusters_confpside)[clusters_pv_confpside<0.2]

vmin, vmax = -2, 2
extent = (np.min(gave_confpside.times), np.max(gave_confpside.times), np.min(gave_confpside.freqs), np.max(gave_confpside.freqs))


fig = plt.figure()
ax = fig.subplots(nrows=2, ncols = 1)
ax[0].contourf(gave_confpside.times, gave_confpside.freqs, np.nanmean(deepcopy(gave_confpside).pick_channels(visrightchans).data , 0), levels = 80, cmap = 'RdBu_r', vmin=vmin, vmax=vmax)
ax[1].contourf(gave_confpside.times, gave_confpside.freqs, np.nanmean(deepcopy(gave_confpside).pick_channels(['C4']).data , 0), levels = 80, cmap = 'RdBu_r', vmin=vmin, vmax=vmax)
for axis in ax:
    axis.vlines(0.0, ymin = 1, ymax = 40, linestyle = '--')
    axis.set_ylim(1,39)
    axis.set_yticks([1,5,10,15,20,25,30,35])
for imask in mask_confpside:
    bigmask = np.kron(imask, np.ones((10,10)))
    ax[1].contour(bigmask, colors='black', extent = extent, linewidths = .5, corner_mask=False, antialiased = False)
fig.suptitle('confidence ~ probed left vs right')
ax[0].set_title('visual channels')
ax[1].set_title('C4 channel (motor)')


