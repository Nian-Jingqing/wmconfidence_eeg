#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:59:35 2019

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


alldata_grandmean       = []
alldata_neutral         = []
alldata_cued            = []
alldata_cuedlvsr        = []
alldata_cvsn            = []
alldata_error           = []
alldata_confidence      = []
alldata_DT              = []
alldata_errorxside      = []
alldata_confidencexside = []
alldata_DTxside         = []

alldata_grandmean_baselined       = []
alldata_neutral_baselined         = []
alldata_cued_baselined            = []
alldata_cuedlvsr_baselined        = []
alldata_cvsn_baselined            = []
alldata_error_baselined           = []
alldata_confidence_baselined      = []
alldata_DT_baselined              = []
alldata_errorxside_baselined      = []
alldata_confidencexside_baselined = []
alldata_DTxside_baselined         = []

alldata_grandmean_t       = []
alldata_neutral_t         = []
alldata_cued_t            = []
alldata_cuedlvsr_t        = []
alldata_cvsn_t            = []
alldata_error_t           = []
alldata_confidence_t      = []
alldata_DT_t              = []
alldata_errorxside_t      = []
alldata_confidencexside_t = []
alldata_DTxside_t         = []

alldata_grandmean_baselined_t       = []
alldata_neutral_baselined_t         = []
alldata_cued_baselined_t            = []
alldata_cuedlvsr_baselined_t        = []
alldata_cvsn_baselined_t            = []
alldata_error_baselined_t           = []
alldata_confidence_baselined_t      = []
alldata_DT_baselined_t              = []
alldata_errorxside_baselined_t      = []
alldata_confidencexside_baselined_t = []
alldata_DTxside_baselined_t         = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
        
    alldata_grandmean.append( mne.time_frequency.read_tfrs( fname      = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_grandmean_betas-tfr.h5'))[0])
    alldata_neutral.append(mne.time_frequency.read_tfrs( fname         = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_neutral_betas-tfr.h5'))[0])
    alldata_cuedlvsr.append(mne.time_frequency.read_tfrs( fname        = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_cued_betas-tfr.h5'))[0])
    alldata_cvsn.append(mne.time_frequency.read_tfrs( fname            = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_cuedvsneutral_betas-tfr.h5'))[0])
    alldata_error.append(mne.time_frequency.read_tfrs( fname           = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_error_betas-tfr.h5'))[0])
    alldata_confidence.append(mne.time_frequency.read_tfrs( fname      = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_confidence_betas-tfr.h5'))[0])
    alldata_DT.append(mne.time_frequency.read_tfrs( fname              = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_dt_betas-tfr.h5'))[0])
    alldata_errorxside.append(mne.time_frequency.read_tfrs( fname      = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_errorxcuedlvsr_betas-tfr.h5'))[0])
    alldata_confidencexside.append(mne.time_frequency.read_tfrs( fname = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_confxcuedlvsr_betas-tfr.h5'))[0])
    alldata_DTxside.append(mne.time_frequency.read_tfrs( fname         = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_dtxcuedlvsr_betas-tfr.h5'))[0])
    
    alldata_grandmean_t.append( mne.time_frequency.read_tfrs( fname      = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_grandmean_tstats-tfr.h5'))[0])
    alldata_neutral_t.append(mne.time_frequency.read_tfrs( fname         = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_neutral_tstats-tfr.h5'))[0])
    alldata_cuedlvsr_t.append(mne.time_frequency.read_tfrs( fname        = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_cued_tstats-tfr.h5'))[0])
    alldata_cvsn_t.append(mne.time_frequency.read_tfrs( fname            = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_cuedvsneutral_tstats-tfr.h5'))[0])
    alldata_error_t.append(mne.time_frequency.read_tfrs( fname           = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_error_tstats-tfr.h5'))[0])
    alldata_confidence_t.append(mne.time_frequency.read_tfrs( fname      = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_confidence_tstats-tfr.h5'))[0])
    alldata_DT_t.append(mne.time_frequency.read_tfrs( fname              = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_dt_tstats-tfr.h5'))[0])
    alldata_errorxside_t.append(mne.time_frequency.read_tfrs( fname      = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_errorxcuedlvsr_tstats-tfr.h5'))[0])
    alldata_confidencexside_t.append(mne.time_frequency.read_tfrs( fname = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_confxcuedlvsr_tstats-tfr.h5'))[0])
    alldata_DTxside_t.append(mne.time_frequency.read_tfrs( fname         = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_dtxcuedlvsr_tstats-tfr.h5'))[0])
    
    alldata_grandmean_baselined.append( mne.time_frequency.read_tfrs( fname      = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_grandmean_betas_baselined-tfr.h5'))[0])
    alldata_neutral_baselined.append(mne.time_frequency.read_tfrs( fname         = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_neutral_betas_baselined-tfr.h5'))[0])
    alldata_cuedlvsr_baselined.append(mne.time_frequency.read_tfrs( fname        = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_cued_betas_baselined-tfr.h5'))[0])
    alldata_cvsn_baselined.append(mne.time_frequency.read_tfrs( fname            = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_cuedvsneutral_betas_baselined-tfr.h5'))[0])
    alldata_error_baselined.append(mne.time_frequency.read_tfrs( fname           = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_error_betas_baselined-tfr.h5'))[0])
    alldata_confidence_baselined.append(mne.time_frequency.read_tfrs( fname      = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_confidence_betas_baselined-tfr.h5'))[0])
    alldata_DT_baselined.append(mne.time_frequency.read_tfrs( fname              = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_dt_betas_baselined-tfr.h5'))[0])
    alldata_errorxside_baselined.append(mne.time_frequency.read_tfrs( fname      = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_errorxcuedlvsr_betas_baselined-tfr.h5'))[0])
    alldata_confidencexside_baselined.append(mne.time_frequency.read_tfrs( fname = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_confxcuedlvsr_betas_baselined-tfr.h5'))[0])
    alldata_DTxside_baselined.append(mne.time_frequency.read_tfrs( fname         = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_dtxcuedlvsr_betas_baselined-tfr.h5'))[0])
    
    alldata_grandmean_baselined_t.append( mne.time_frequency.read_tfrs( fname      = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_grandmean_tstats_baselined-tfr.h5'))[0])
    alldata_neutral_baselined_t.append(mne.time_frequency.read_tfrs( fname         = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_neutral_tstats_baselined-tfr.h5'))[0])
    alldata_cuedlvsr_baselined_t.append(mne.time_frequency.read_tfrs( fname        = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_cued_tstats_baselined-tfr.h5'))[0])
    alldata_cvsn_baselined_t.append(mne.time_frequency.read_tfrs( fname            = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_cuedvsneutral_tstats_baselined-tfr.h5'))[0])
    alldata_error_baselined_t.append(mne.time_frequency.read_tfrs( fname           = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_error_tstats_baselined-tfr.h5'))[0])
    alldata_confidence_baselined_t.append(mne.time_frequency.read_tfrs( fname      = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_confidence_tstats_baselined-tfr.h5'))[0])
    alldata_DT_baselined_t.append(mne.time_frequency.read_tfrs( fname              = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_dt_tstats_baselined-tfr.h5'))[0])
    alldata_errorxside_baselined_t.append(mne.time_frequency.read_tfrs( fname      = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_errorxcuedlvsr_tstats_baselined-tfr.h5'))[0])
    alldata_confidencexside_baselined_t.append(mne.time_frequency.read_tfrs( fname = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_confxcuedlvsr_tstats_baselined-tfr.h5'))[0])
    alldata_DTxside_baselined_t.append(mne.time_frequency.read_tfrs( fname         = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_dtxcuedlvsr_tstats_baselined-tfr.h5'))[0])
    
    
#%%    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#visualise some regressors
#we're not going to plot the grandmean, this shouldn't be in the glm as it doesnt make sense anyways.
#we will visualise the regressors for neutral and cued trials though as these should be different (no lateralisation in in neutral trials)
timefreqs = {(.4, 10):(.4, 4),
             (.6, 10):(.4, 4),
             (.8, 10):(.4, 4),
             (.4, 22):(.4, 16),
             (.6, 22):(.4, 16),
             (.8, 22):(.4, 16)}

timefreqs_alpha ={(.4, 10):(.4, 4),
                  (.6, 10):(.4, 4),
                  (.8, 10):(.4, 4),
                  (1., 10):(.4, 4)}
visleftchans  = ['PO3', 'PO7', 'O1']

visrightchans = ['PO4','PO8','O2']

motrightchans = ['C2', 'C4']  #channels contra to the left hand (space bar)
motleftchans = ['C1', 'C3']   #channels contra to the right hand (mouse)

topoargs = dict(outlines= 'head', contours = 0)
topoargs_t = dict(outlines = 'head', contours = 0, vmin=-2, vmax = 2)

#%%


gave_clvsr = mne.grand_average(alldata_cuedlvsr_t); gave_clvsr.data = toverparam(alldata_cuedlvsr_t); gave_clvsr.drop_channels(['RM'])
gave_clvsr.plot_joint(topomap_args=topoargs_t, timefreqs = timefreqs_alpha)

times, freqs = gave_clvsr.times, gave_clvsr.freqs

cvsn_vis = np.subtract( np.nanmean(deepcopy(gave_clvsr).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(gave_clvsr).pick_channels(visleftchans).data, 0))
cvsn_mouse = np.subtract( np.nanmean( deepcopy( gave_clvsr).pick_channels(motleftchans).data,0), np.nanmean( deepcopy( gave_clvsr).pick_channels(motrightchans).data,0))
cvsn_space = np.subtract( np.nanmean( deepcopy( gave_clvsr).pick_channels(motrightchans).data,0), np.nanmean( deepcopy( gave_clvsr).pick_channels(motleftchans).data,0))


fig = plt.figure()
ax = fig.subplots(3,1)

ax[0].contourf(times, freqs, cvsn_vis, levels = 100, vmin = -2, vmax = 2, cmap = 'RdBu_r', antialiased=False)
ax[1].contourf(times, freqs, cvsn_mouse, levels = 100, vmin = -2, vmax = 2, cmap = 'RdBu_r', antialiased=False)
ax[2].contourf(times, freqs, cvsn_space, levels = 100, vmin = -2, vmax = 2, cmap = 'RdBu_r', antialiased=False)
ax[0].set_title('cvsi to attended item, visual channels')
ax[1].set_title('cvsi to mouse hand')
ax[2].set_title('cvsi to space bar hand')
ax[0].vlines([0], linestyle = '--', ymin = 1, ymax = 39, lw = .75)
ax[1].vlines([0], linestyle = '--', ymin = 1, ymax = 39, lw = .75)
ax[2].vlines([0], linestyle = '--', ymin = 1, ymax = 39, lw = .75)

#plot tfr of motor electrodes to look for any kind of motor beta build up prior to probe, or evoked by the cue
fig2 = plt.figure()
ax2 = fig2.subplots(2,1)
ax2[0].contourf(times, freqs, np.nanmean(deepcopy(gave_clvsr).pick_channels(motrightchans).data, 0), levels = 100, cmap = 'RdBu_r', vmin = -2, vmax = 2, antialiased = False)
ax2[1].contourf(times, freqs, np.nanmean(deepcopy(gave_clvsr).pick_channels(motleftchans).data, 0) , levels = 100, cmap = 'RdBu_r', vmin = -2, vmax = 2, antialiased = False)
ax2[0].set_title('TFR contra to space bar hand (C2, C4)')
ax2[1].set_title('TFR contra to mouse hand (C1, C3)')
ax2[0].vlines([0], linestyle = '--', ymin = 1, ymax = 39, lw = .75)
ax2[1].vlines([0], linestyle = '--', ymin = 1, ymax = 39, lw = .75)
#%%


gave_dtxclvsr = mne.grand_average(alldata_DTxside_t); gave_dtxclvsr.data = toverparam(alldata_DTxside_t); gave_dtxclvsr.drop_channels(['RM'])
gave_dtxclvsr.plot_joint(topomap_args = topoargs_t, timefreqs = timefreqs_alpha)


dtxclvsr_cvsi_vis = np.subtract( np.nanmean(deepcopy(gave_dtxclvsr).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(gave_dtxclvsr).pick_channels(visleftchans).data, 0))



fig = plt.figure()
ax = fig.subplots(3)
ax[0].contourf(times, freqs, dtxclvsr_cvsi_vis, levels= 100, cmap = 'RdBu_r', vmin=-2, vmax =2, antialiased = False)
ax[0].set_title('decision time ~ contra vs ipsi to cued item')
ax[0].vlines(0, linestyle='--', lw=.75, ymin=1, ymax=39)
ax[1].contourf(times, freqs, np.nanmean(deepcopy(gave_dtxclvsr).pick_channels(visrightchans).data,0), levels = 100, cmap = 'RdBu_r', vmin=-2, vmax=2, antialiased=False)
ax[1].vlines(0, linestyle='--', lw=.75, ymin=1, ymax=39)
ax[2].contourf(times, freqs, np.nanmean(deepcopy(gave_dtxclvsr).pick_channels(visleftchans).data,0), levels = 100, cmap = 'RdBu_r', vmin=-2, vmax=2, antialiased=False)
ax[2].vlines(0, linestyle='--', lw=.75, ymin=1, ymax=39)

x_dtxclvsr_cvsi_vis = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_dtxclvsr_cvsi_vis[i,:,:] = np.subtract( np.nanmean(deepcopy(alldata_DTxside_t[i]).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(alldata_DTxside_t[i]).pick_channels(visleftchans).data, 0))


t, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(x_dtxclvsr_cvsi_vis, n_permutations='all')
mask = np.asarray(clusters)[cluster_pv<0.2]

fig = plt.figure()
ax = fig.subplots(1)
ax.contourf(times, freqs, dtxclvsr_cvsi_vis, levels= 100, cmap = 'RdBu_r', vmin=-2, vmax =2, antialiased = False)
ax.set_title('decision time ~ contra vs ipsi to cued item')
ax.vlines(0, linestyle='--', lw=.75, ymin=1, ymax=39)
for imask in mask:
    bigmask = np.kron(imask, np.ones((10,10)))
    ax.contour(bigmask, colors='black', lw=.75, extent = ( np.min(times), np.max(times), np.min(freqs), np.max(freqs)), antialiased = False)

#%%
    
    
gave_errorxclvsr = mne.grand_average(alldata_errorxside_t); gave_errorxclvsr.data = toverparam(alldata_errorxside_t); gave_errorxclvsr.drop_channels(['RM'])
gave_errorxclvsr.plot_joint(topomap_args = topoargs_t, timefreqs = timefreqs_alpha)#, baseline = (None, None))


errorxclvsr_cvsi_vis = np.subtract( np.nanmean(deepcopy(gave_errorxclvsr).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(gave_errorxclvsr).pick_channels(visleftchans).data, 0))



fig = plt.figure()
ax = fig.subplots(1)
ax.contourf(times, freqs, errorxclvsr_cvsi_vis, levels= 100, cmap = 'RdBu_r', vmin=-2, vmax =2, antialiased = False)
ax.set_title('error ~ contra vs ipsi to cued item')
ax.vlines(0, linestyle='--', lw=.75, ymin=1, ymax=39)


x_errorxclvsr_cvsi_vis = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_errorxclvsr_cvsi_vis[i,:,:] = np.subtract( np.nanmean(deepcopy(alldata_errorxside_t[i]).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(alldata_errorxside_t[i]).pick_channels(visleftchans).data, 0))


t, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(x_errorxclvsr_cvsi_vis, n_permutations='all')
mask = np.asarray(clusters)[cluster_pv<0.2]

fig = plt.figure()
ax = fig.subplots(1)
ax.contourf(times, freqs, dtxclvsr_cvsi_vis, levels= 100, cmap = 'RdBu_r', vmin=-2, vmax =2, antialiased = False)
ax.set_title('error ~ contra vs ipsi to cued item')
ax.vlines(0, linestyle='--', lw=.75, ymin=1, ymax=39)
for imask in mask:
    bigmask = np.kron(imask, np.ones((10,10)))
    ax.contour(bigmask, colors='black', lw=.75, extent = ( np.min(times), np.max(times), np.min(freqs), np.max(freqs)), antialiased = False)

#%% motor? just the mouse hand



fig2 = plt.figure()
ax2 = fig2.subplots(3,1)
ax2[0].contourf(times, freqs, np.nanmean(deepcopy(gave_errorxclvsr).pick_channels(motrightchans).data, 0), levels = 100, cmap = 'RdBu_r', vmin = -2, vmax = 2, antialiased = False)
ax2[1].contourf(times, freqs, np.nanmean(deepcopy(gave_errorxclvsr).pick_channels(motleftchans).data, 0) , levels = 100, cmap = 'RdBu_r', vmin = -2, vmax = 2, antialiased = False)
ax2[2].contourf(times, freqs, np.subtract(np.nanmean(deepcopy(gave_errorxclvsr).pick_channels(motrightchans).data, 0), np.nanmean(deepcopy(gave_errorxclvsr).pick_channels(motleftchans).data, 0)),
   levels = 100, cmap = 'RdBu_r', vmin = -2, vmax = 2, antialiased = False)

ax2[0].set_title('Error -- TFR contra to space bar hand (C2, C4)')
ax2[1].set_title('Error -- TFR contra to mouse hand (C1, C3)')
ax2[2].set_title('Error -- TFR contra - ipsi to space bar hand')

ax2[0].vlines([0], linestyle = '--', ymin = 1, ymax = 39, lw = .75)
ax2[1].vlines([0], linestyle = '--', ymin = 1, ymax = 39, lw = .75)
ax2[2].vlines([0], linestyle = '--', ymin = 1, ymax = 39, lw = .75)



























