#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:14:21 2019

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

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
subs = np.array([1,       4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm


alldata_grandmean       = []
alldata_cuedlvsr        = []
alldata_cuedplsr        = []
alldata_error           = []
alldata_confidence      = []
alldata_DT              = []
alldata_errorxside      = []
alldata_confidencexside = []
alldata_DTxside         = []

alldata_grandmean_baselined       = []
alldata_cuedlvsr_baselined        = []
alldata_cuedplsr_baselined        = []
alldata_error_baselined           = []
alldata_confidence_baselined      = []
alldata_DT_baselined              = []
alldata_errorxside_baselined      = []
alldata_confidencexside_baselined = []
alldata_DTxside_baselined         = []

alldata_grandmean_t       = []
alldata_cuedlvsr_t        = []
alldata_cuedplsr_t        = []
alldata_error_t           = []
alldata_confidence_t      = []
alldata_DT_t              = []
alldata_errorxside_t      = []
alldata_confidencexside_t = []
alldata_DTxside_t         = []

alldata_grandmean_baselined_t       = []
alldata_cuedlvsr_baselined_t        = []
alldata_cuedplsr_baselined_t        = []
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
    
    alldata_grandmean.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_grandmean_betas-tfr.h5'))[0])
    alldata_cuedlvsr.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedlvsr_betas-tfr.h5'))[0])
    alldata_cuedplsr.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedlplusr_betas-tfr.h5'))[0])
    alldata_error.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_error_betas-tfr.h5'))[0])
    alldata_confidence.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidence_betas-tfr.h5'))[0])
    alldata_DT.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DT_betas-tfr.h5'))[0])
    alldata_errorxside.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_errorxside_betas-tfr.h5'))[0])
    alldata_confidencexside.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidencexside_betas-tfr.h5'))[0])
    alldata_DTxside.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DTxside_betas-tfr.h5'))[0])
    
    
    alldata_grandmean_baselined.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_grandmean_betas_baselined-tfr.h5'))[0])
    alldata_cuedlvsr_baselined.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedlvsr_betas_baselined-tfr.h5'))[0])
    alldata_cuedplsr_baselined.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedlplusr_betas_baselined-tfr.h5'))[0])
    alldata_error_baselined.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_error_betas_baselined-tfr.h5'))[0])
    alldata_confidence_baselined.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidence_betas_baselined-tfr.h5'))[0])
    alldata_DT_baselined.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DT_betas_baselined-tfr.h5'))[0])
    alldata_errorxside_baselined.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_errorxside_betas_baselined-tfr.h5'))[0])
    alldata_confidencexside_baselined.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidencexside_betas_baselined-tfr.h5'))[0])
    alldata_DTxside_baselined.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DTxside_betas_baselined-tfr.h5'))[0])
    
    
    #tstats
    alldata_grandmean_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_grandmean_tstats-tfr.h5'))[0])
    alldata_cuedlvsr_t.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedlvsr_tstats-tfr.h5'))[0])
    alldata_cuedplsr_t.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedlplusr_tstats-tfr.h5'))[0])
    alldata_error_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_error_tstats-tfr.h5'))[0])
    alldata_confidence_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidence_tstats-tfr.h5'))[0])
    alldata_DT_t.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DT_tstats-tfr.h5'))[0])
    alldata_errorxside_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_errorxside_tstats-tfr.h5'))[0])
    alldata_confidencexside_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidencexside_tstats-tfr.h5'))[0])
    alldata_DTxside_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DTxside_tstats-tfr.h5'))[0])
    
    
    alldata_grandmean_baselined_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_grandmean_tstats_baselined-tfr.h5'))[0])
    alldata_cuedlvsr_baselined_t.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedlvsr_tstats_baselined-tfr.h5'))[0])
    alldata_cuedplsr_baselined_t.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedlplusr_tstats_baselined-tfr.h5'))[0])
    alldata_error_baselined_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_error_tstats_baselined-tfr.h5'))[0])
    alldata_confidence_baselined_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidence_tstats_baselined-tfr.h5'))[0])
    alldata_DT_baselined_t.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DT_tstats_baselined-tfr.h5'))[0])
    alldata_errorxside_baselined_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_errorxside_tstats_baselined-tfr.h5'))[0])
    alldata_confidencexside_baselined_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidencexside_tstats_baselined-tfr.h5'))[0])
    alldata_DTxside_baselined_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DTxside_tstats_baselined-tfr.h5'))[0])

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

visleftchans  = ['PO4','PO8','O2']
visrightchans = ['PO3', 'PO7', 'O1']

#%% Question 1 -- Do cues facilitate selection of an item from WM in our task?
#analysis needed:
# -- cued vs neutral (i.e. do visual responses look different in the two?)
# -- cued left vs cued right (do things lateralise in a way that suggests selection)

#just plot the average for all neutral trials first
gave_neutral = mne.grand_average(alldata_neutral_baselined); gave_neutral.drop_channels(['RM'])
gave_neutral.plot_joint(title = 'evoked response to neutral cues, average betas',
                        timefreqs = timefreqs_alpha,
                        topomap_args = dict(outlines = 'head', contours = 0))

gave_neutral = mne.grand_average(alldata_neutral_baselined); gave_neutral.data = toverparam(alldata_neutral_baselined);
gave_neutral.drop_channels(['RM'])
gave_neutral.plot_joint(title     = 'evoked response to neutral cues, t over betas',
                        timefreqs = timefreqs_alpha, 
                        topomap_args = dict(outlines = 'head', contours = 0),
                        vmin = -2, vmax = 2)

gave_neutral_t = mne.grand_average(alldata_neutral_baselined_t); gave_neutral_t.drop_channels(['RM'])
gave_neutral_t.plot_joint(title = 'evoked response to neutral cues, average tstats',
                          timefreqs = timefreqs_alpha,
                          topomap_args = dict(outlines = 'head', contours = 0))

gave_neutral_t = mne.grand_average(alldata_neutral_baselined_t); gave_neutral_t.drop_channels(['RM'])
gave_neutral_t.plot_joint(title = 'evoked response to neutral cues, average tstats',
                          timefreqs = timefreqs_alpha,
                          topomap_args = dict(outlines = 'head', contours = 0))


#and not plot the average for all cued trials

gave_cued = mne.grand_average(alldata_cued_baselined); gave_cued.drop_channels(['RM'])
gave_cued.plot_joint(title = 'evoked response to retrocues, average betas',
                     timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0))

gave_cued = mne.grand_average(alldata_cued_baselined); gave_cued.data = toverparam(alldata_cued_baselined)
gave_cued.drop_channels(['RM'])
gave_cued.plot_joint(title = 'evoked response to retrocues, t over betas',
                     timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0),
                     vmin = -2, vmax = 2)

gave_cued_t = mne.grand_average(alldata_cued_baselined_t); gave_cued_t.drop_channels(['RM'])
gave_cued_t.plot_joint(title = 'evoked response to retrocues, average t stats',
                       timefreqs = timefreqs_alpha,
                       topomap_args = dict(outlines = 'head', contours = 0),
                       vmin = -1, vmax = 1)

gave_cued_t = mne.grand_average(alldata_cued_baselined_t); gave_cued_t.data = toverparam(alldata_cued_baselined_t)
gave_cued_t.drop_channels(['RM'])
gave_cued_t.plot_joint(title = 'evoked response to retrocues, t over tstats',
                       timefreqs = timefreqs_alpha,
                       topomap_args = dict(outlines = 'head', contours = 0, vmin = -2.5, vmax = 2.5),
                       vmin = -2.5, vmax = 2.5)



#now plot the difference between the two
#this is a contrast so we don't need to have the baselined version i don't think)

gave_cvsn = mne.grand_average(alldata_cvsn); gave_cvsn.drop_channels(['RM'])
gave_cvsn.plot_joint(title = 'cued vs neutral trials, average betas',
                     timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0))

gave_cvsn = mne.grand_average(alldata_cvsn); gave_cvsn.data = toverparam(alldata_cvsn); gave_cvsn.drop_channels(['RM'])
gave_cvsn.plot_joint(title = 'cued vs neutral trials, t over betas', 
                     timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0, vmin = -2.5, vmax = 2.5),
                     vmin = -2.5, vmax = 2.5)

gave_cvsn_t = mne.grand_average(alldata_cvsn_t); gave_cvsn_t.drop_channels(['RM'])
gave_cvsn_t.plot_joint(title = 'cued vs neutral, average tstats',
                       timefreqs = timefreqs_alpha,
                       topomap_args = dict(outlines = 'head', contours = 0, vmin = -1.5, vmax = 1.5),
                       vmin = -1.5, vmax = 1.5)

gave_cvsn_t = mne.grand_average(alldata_cvsn_t); gave_cvsn_t.data = toverparam(alldata_cvsn_t); gave_cvsn_t.drop_channels(['RM'])
gave_cvsn_t.plot_joint(title = 'cued vs neutral, t over tstats',
                       timefreqs = timefreqs_alpha,
                       topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2),
                       vmin = -2, vmax = 2) #brings the tf image onto same scale as the topomaps
#so this is nice, and shows that there are visual alpha suppression differences between cued and neutral trials, and its quite lateralised
#i.e. when there is a retrocue, there is (on average) some extra visual alpha suppression, compared to neutral trials
#this is a coarse way of looking at it, so you need to look at just cued trials, and look at cued side effects

gave_cuedside = mne.grand_average(alldata_cuedlvsr); gave_cuedside.drop_channels(['RM'])
gave_cuedside.plot_joint(title = 'cued left vs right, average betas',
                         timefreqs = timefreqs_alpha,
                         topomap_args = dict(outlines = 'head', contours = 0))

gave_cuedside = mne.grand_average(alldata_cuedlvsr); gave_cuedside.data = toverparam(alldata_cuedlvsr)
gave_cuedside.drop_channels(['RM'])
gave_cuedside.plot_joint(title = 'cued left vs right, t over betas',
                         timefreqs = timefreqs_alpha,
                         topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2),
                         vmin = -2, vmax = 2)

gave_cuedside_t = mne.grand_average(alldata_cuedlvsr_t); gave_cuedside_t.drop_channels(['RM'])
gave_cuedside_t.plot_joint(title = 'cued left vs right, average tstats',
                          timefreqs = timefreqs_alpha,
                          topomap_args = dict(outlines ='head', contours = 0, vmin = -1, vmax = 1),
                          vmin = -1, vmax = 1)


gave_cuedside_t = mne.grand_average(alldata_cuedlvsr_t); gave_cuedside_t.data = toverparam(alldata_cuedlvsr_t)
gave_cuedside_t.drop_channels(['RM'])
gave_cuedside_t.plot_joint(title = 'cued left vs right, t over tstats',
                          timefreqs = timefreqs_alpha,
                          topomap_args = dict(outlines ='head', contours = 0, vmin = -2, vmax = 2),
                          vmin = -2, vmax = 2)
gave_cuedside_t.plot(picks = ['PO4', 'PO8', 'O2'], combine = 'mean', vmin = -2, vmax = 2, title = 'cued side, t over tstats, average over PO4,PO8,O2')
gave_cuedside_t.plot(picks = ['PO4', 'PO8', 'O2'], vmin = -2, vmax = 2, title = 'cued side, t over tstats, average over PO4,PO8,O2')

clvsr_norm = []
for i in range(len(alldata_cuedlvsr_t)):
    tmp_lvsr  = deepcopy(alldata_cuedlvsr[i])
    tmp_lplsr = deepcopy(alldata_cuedplsr[i])
    tmp_lvsr.data = np.multiply(np.divide(tmp_lvsr.data, tmp_lplsr.data),1)
    clvsr_norm.append(tmp_lvsr)

gave_lat_norm = mne.grand_average(clvsr_norm); gave_lat_norm.drop_channels(['RM'])
gave_lat_norm.plot_joint(timefreqs = timefreqs_alpha,
                         topomap_args = dict(outlines = 'head', contours = 0, vmin=-10,vmax=10),
                         vmin=-10, vmax=10)
#%%
gave_lvsr = mne.grand_average(alldata_cuedlvsr); gave_lvsr.drop_channels(['RM'])

chids =         np.array([             1,  2,  3,
                                   4,  5,  6,  7,  8,
                           9, 10, 11, 12, 13, 14, 15, 16, 17,
                          18, 19, 20, 21, 22, 23, 24, 25, 26,
                          27, 28, 29, 30, 31, 32, 33, 34, 35,
                          36, 37, 38, 39, 40, 41, 42, 43, 44,
                          45, 46, 47, 48, 49, 50, 51, 52, 53,
                                  54, 55, 56, 57, 58,
                                      59, 60, 61,
                                          62
                                          ])
chids = np.subtract(chids,1)
flipids =       np.array([             3,  2,  1,
                                   8,  7,  6,  5,  4,
                          17, 16, 15, 14, 13, 12, 11, 10,  9,
                          26, 25, 24, 23, 22, 21, 20, 19, 18,
                          35, 34, 33, 32, 31, 30, 29, 28, 27,
                          44, 43, 42, 41, 40, 39, 38, 37, 36,
                          53, 52, 51, 50, 49, 48, 47, 46, 45,
                                  58, 57, 56, 55, 54,
                                      61, 60, 59,
                                          62
                                          ])
flipids = np.subtract(flipids,1)
flipids = flipids[:61]

lvsr_cvsi = []

for i in range(len(subs)):
    lvsr_cvsi.append(alldata_cuedlvsr[i])
    
for i in range(len(subs)):
    tmp_lvsr = deepcopy(lvsr_cvsi[i]).drop_channels(['RM'])
    tmp_flipped = deepcopy(lvsr_cvsi[i])
    tmp_flipped.data = tmp_flipped.data[flipids,:,:]
    tmp_lvsr.data = np.subtract(tmp_lvsr.data, tmp_flipped.data)
    lvsr_cvsi[i] = tmp_lvsr
    
    
gave_lvsr_cvsi = mne.grand_average(lvsr_cvsi); #gave_lvsr_cvsi.data = toverparam(lvsr_cvsi)
gave_lvsr_cvsi.plot_joint(timefreqs = timefreqs_alpha, topomap_args = dict(outlines='head', contours=0))
gave_lvsr_cvsi.plot(picks = ['PO8', 'PO4', 'O2'], combine = 'mean')


#%% can we do some cluster on this?
#take electrode selection down to the average of the left visual sensors
#cluster perms
#visualise any clusters?

tmpdata = deepcopy(lvsr_cvsi)
for i in tmpdata:
    i = i.crop(tmin = None, tmax = None) #restrict time window for the clustering
    i = i.pick_channels(['PO4', 'PO8', 'O2'])
    
times = tmpdata[0].times

tmin, tmax = -0.5, 1.5
X_lvsr_cvsi = np.empty(shape = (len(subs), gave_lvsr.freqs.size, gave_lvsr.crop(tmin = tmin, tmax = tmax).times.size))
for i in range(len(lvsr_cvsi)):
    tmp = deepcopy(lvsr_cvsi[i])
    tmp.pick_channels(['PO4', 'PO8', 'O2'])
    tmp.crop(tmin = tmin, tmax = tmax)
    data = np.nanmean(tmp.data, axis=0)
    X_lvsr_cvsi[i,:,:] = data

np.random.seed(seed=1)
t_lvsr_cvsi, clusters_lvsr_cvsi, cluster_pv_lvsr_cvsi, H0_lvsr_cvsi = mne.stats.permutation_cluster_1samp_test(X_lvsr_cvsi, out_type='indices')

mask_lvsr_cvsi = np.squeeze(np.asarray(clusters_lvsr_cvsi)[cluster_pv_lvsr_cvsi<0.4])

gave_lvsr.plot(picks = ['PO4', 'PO8', 'O2'], combine = 'mean',
               mask = mask_lvsr_cvsi,
               mask_style = 'both',
               mask_cmap = 'RdBu_r',
               mask_alpha=1)
#%% Question 2 -- if there is selection (suggests there is), what are the behavioural implications?
# is lateralisation associated with lower error? higher confidence? faster reaction times?

#selection and error
gave_errorcside = mne.grand_average(alldata_errorxside); gave_errorcside.drop_channels(['RM'])
gave_errorcside.plot_joint(title = 'effect of error on selection (cued left vs right), average betas',
                           timefreqs = timefreqs_alpha,
                           topomap_args = dict(outlines = 'head', contours = 0))

gave_errorcside = mne.grand_average(alldata_errorxside); gave_errorcside.data = toverparam(alldata_errorxside);
gave_errorcside.drop_channels(['RM'])
gave_errorcside.plot_joint(title = 'effect of error on selection (cued left vs right), t over betas',
                           timefreqs = timefreqs_alpha,
                           topomap_args = dict(outlines = 'head', contours = 0, vmin = -1.8, vmax = 1.8),
                           vmin = -1.8, vmax = 1.8)

gave_errorcside_t = mne.grand_average(alldata_errorxside_t); gave_errorcside_t.drop_channels(['RM'])
gave_errorcside_t.plot_joint(title = 'effect of error on selection (cued left vs right), average tstats',
                             timefreqs = timefreqs_alpha,
                             topomap_args = dict(outlines = 'head', contours = 0, vmin = -1, vmax = 1),
                             vmin = -1, vmax = 1)

gave_errorcside_t = mne.grand_average(alldata_errorxside_t); gave_errorcside_t.data = toverparam(alldata_errorxside_t)
gave_errorcside_t.drop_channels(['RM'])
gave_errorcside_t.plot_joint(title = 'effect of error on selection (cued left vs right), t over tstats',
                             timefreqs = timefreqs_alpha,
                             topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2),
                             vmin = -2, vmax = 2)
gave_errorcside_t.plot(picks = visleftchans, combine = 'mean', vmin = -2, vmax = 2,
                       title = 'TFR of visual left channels, error x cued side')

#do cvsi on this? at the moment this regressor looks at differences in topography between the conditions but it doesn't look at lateralisation
#this says what is the influence of error on cued left trials vs cued right trials
#any lateralisation here is just the contrast of cued left vs cued right on that specific sensor
#it doesn't look at lateralisation on that trial and how it relates to error
#so we need to take cvsi of this errorxcside and plot that i think
#or we possibly need to put in the contra vs ipsi data into the glm itself maybe








#this isn't super convincing just yet
#also need to understand this
# its cued (1) vs neutral (-1)
#contralateral suppression would be negative contra (right hand side)
#if increased lateralisation was related to lower error, then lower alpha & lower error (higher accuracy) would be a positive tstat
#so in this, we would hope to see a lateralised effect where it's positive on the right hand side and negative on the left
#(i.e flipped signs to normal alpha lateralisation effect)


#is alpha lateralisation/item selection related to confidence?

#selection and confidence

gave_confcside = mne.grand_average(alldata_confidencexside); gave_confcside.drop_channels(['RM'])
gave_confcside.plot_joint(title = 'effect of retrocue (cued left vs right) on confidence, average betas',
                          timefreqs = timefreqs_alpha,
                          topomap_args = dict(outlines = 'head', contours = 0))

gave_confcside = mne.grand_average(alldata_confidencexside); gave_confcside.data = toverparam(alldata_confidencexside)
gave_confcside.drop_channels(['RM'])
gave_confcside.plot_joint(title = 'effect of retrocue (cued left vs right) on confidence, t over betas',
                          timefreqs = timefreqs_alpha,
                          topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2),
                          vmin = -2, vmax = 2)

gave_confcside_t = mne.grand_average(alldata_confidencexside_t); gave_confcside_t.drop_channels(['RM'])
gave_confcside_t.plot_joint(title = 'effect of retrocue (cued left vs right) on confidence, average tstats',
                            timefreqs = timefreqs_alpha,
                            topomap_args = dict(outlines = 'head', contours = 0, vmin = -1, vmax = 1),
                            vmin = -1, vmax = 1)

gave_confcside_t = mne.grand_average(alldata_confidencexside_t); gave_confcside_t.data = toverparam(alldata_confidencexside_t)
gave_confcside_t.drop_channels(['RM'])
gave_confcside_t.plot_joint(title = 'effect of retrocue (cued left vs right) on confidence, t over tstats',
                            timefreqs = timefreqs_alpha,
                            topomap_args = dict(outlines = 'head', contours = 0, vmin = -1.8, vmax = 1.8),
                            vmin = -1.8, vmax = 1.8)
gave_confcside_t.plot(picks = visleftchans, combine = 'mean', vmin = -2, vmax = 2,
                      title = 'TFR of visual left channels, confidence x cued side')

#and what about reaction time? are reaction times faster on trials where the cue facilitates more item selection?

#selection and DT


gave_dtcside = mne.grand_average(alldata_DTxside); gave_dtcside.drop_channels(['RM'])
gave_dtcside.plot_joint(title = 'effect of retrocue (cued left vs right) on reaction time, average betas',
                        timefreqs = timefreqs_alpha,
                        topomap_args = dict(outlines = 'head', contours = 0))

gave_dtcside = mne.grand_average(alldata_DTxside); gave_dtcside.data = toverparam(alldata_DTxside)
gave_dtcside.drop_channels(['RM'])
gave_dtcside.plot_joint(title = 'effect of retrocue (cued left vs right) on reaction time, t over betas',
                        timefreqs = timefreqs_alpha,
                        topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2),
                        vmin = -2, vmax = 2)


gave_dtcside_t = mne.grand_average(alldata_DTxside_t); gave_dtcside_t.drop_channels(['RM'])
gave_dtcside_t.plot_joint(title = 'effect of retrocue (cued left vs right) on reaction time, average tstats',
                        timefreqs = timefreqs_alpha,
                        topomap_args = dict(outlines = 'head', contours = 0))

gave_dtcside_t = mne.grand_average(alldata_DTxside_t); gave_dtcside_t.data = toverparam(alldata_DTxside_t)
gave_dtcside_t.drop_channels(['RM'])
gave_dtcside_t.plot_joint(title = 'effect of retrocue (cued left vs right) on reaction time, t over tstats',
                        timefreqs = timefreqs_alpha,
                        topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2),
                        vmin = -2, vmax = 2)


gave_dtcside_t.plot(picks = visleftchans, combine = 'mean', vmin = -2, vmax = 2,
                    title = 'effect of retrocue (cued left vs right) on reaction time, t over tstats, left visual channels')

gave_dtcside_t.plot(picks = ['C2', 'C4'], vmin = -2, vmax = 2,
                    title = 'effect of retrocue (left vs right) on reaction time, t over tstats, right motor chans')


#doesn't seem like much convincing evidence that the selection relates to behaviour yet, maybe needs more subjects idk



#%% Question 3 -- Beyond item-specific selection signatures, are there any global states that relate to behaviour in the cue evoked response?


#error related states
gave_error = mne.grand_average(alldata_error_baselined); gave_error.drop_channels(['RM'])
gave_error.plot_joint(title = 'main effect of error, average betas',
                      timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0))

gave_error = mne.grand_average(alldata_error_baselined); gave_error.data = toverparam(alldata_error_baselined)
gave_error.drop_channels(['RM'])
gave_error.plot_joint(title = 'main effect of error, t over betas',
                      timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2),
                      vmin =- 2, vmax = 2)

gave_error_t = mne.grand_average(alldata_error_baselined_t); gave_error_t.drop_channels(['RM'])
gave_error_t.plot_joint(title = 'main effect of error, average tstats', 
                        timefreqs = timefreqs_alpha,
                        topomap_args = dict(outlines = 'head', contours = 0, vmin = -1, vmax = 1),
                        vmin = -1, vmax = 1)

gave_error_t = mne.grand_average(alldata_error_baselined_t); gave_error_t.data = toverparam(alldata_error_baselined_t)
gave_error_t.drop_channels(['RM'])
gave_error_t.plot_joint(title = 'main effect of error, t over tstats', 
                        timefreqs = timefreqs_alpha,
                        topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2),
                        vmin = -2, vmax = 2)


#confidence related states?
gave_conf = mne.grand_average(alldata_confidence_baselined); gave_conf.drop_channels(['RM'])
gave_conf.plot_joint(title = 'main effect of confidence, average betas',
                     timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0))

gave_conf = mne.grand_average(alldata_confidence_baselined); gave_conf.data = toverparam(alldata_confidence_baselined)
gave_conf.drop_channels(['RM'])
gave_conf.plot_joint(title = 'main effect of confidence, t over betas',
                     timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0, vmin = -1.5, vmax = 1.5),
                     vmin = -1.5, vmax = 1.5)


gave_conf_t = mne.grand_average(alldata_confidence_baselined_t); gave_conf_t.drop_channels(['RM'])
gave_conf_t.plot_joint(title = 'main effect of confidence, average tstats',
                     timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0))

gave_conf_t = mne.grand_average(alldata_confidence_baselined_t); gave_conf_t.data = toverparam(alldata_confidence_baselined_t)
gave_conf_t.drop_channels(['RM'])
gave_conf_t.plot_joint(title = 'main effect of confidence, t over tstats',
                     timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0, vmin = -1.5, vmax = 1.5),
                     vmin = -1.5, vmax = 1.5)


#reaction time states? what globally determines speed of access to working memory?

gave_dt = mne.grand_average(alldata_DT_baselined); gave_dt.drop_channels(['RM'])
gave_dt.plot_joint(title = 'main effect of reaction time, average betas',
                   timefreqs = timefreqs_alpha,
                   topomap_args = dict(outlines = 'head', contours = 0))

gave_dt = mne.grand_average(alldata_DT_baselined); gave_dt.data = toverparam(alldata_DT_baselined); gave_dt.drop_channels(['RM'])
gave_dt.plot_joint(title = 'main effect of reaction time, t over betas',
                   timefreqs = timefreqs_alpha,
                   topomap_args = dict(outlines = 'head', contours = 0, vmin = -1.5, vmax = 1.5),
                   vmin = -1.5, vmax = 1.5)

gave_dt_t = mne.grand_average(alldata_DT_baselined_t); gave_dt_t.drop_channels(['RM'])
gave_dt_t.plot_joint(title = 'main effect of reaction time, average tstats',
                   timefreqs = timefreqs_alpha,
                   topomap_args = dict(outlines = 'head', contours = 0))

gave_dt_t = mne.grand_average(alldata_DT_baselined_t); gave_dt_t.data = toverparam(alldata_DT_baselined_t); gave_dt_t.drop_channels(['RM'])
gave_dt_t.plot_joint(title = 'main effect of reaction time, t over tstats',
                   timefreqs = timefreqs_alpha,
                   topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2),
                   vmin = -2, vmax = 2)


#doesn't really seem like there's anything super convincing here just yet to be honest
