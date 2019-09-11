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
alldata_neutral         = []
alldata_cued            = []
alldata_cuedside        = []
alldata_error           = []
alldata_confidence      = []
alldata_DT              = []
alldata_errorxside      = []
alldata_confidencexside = []
alldata_DTxside         = []
alldata_cvsn            = []

alldata_grandmean_baselined       = []
alldata_neutral_baselined         = []
alldata_cued_baselined            = []
alldata_cuedside_baselined        = []
alldata_error_baselined           = []
alldata_confidence_baselined      = []
alldata_DT_baselined              = []
alldata_errorxside_baselined      = []
alldata_confidencexside_baselined = []
alldata_DTxside_baselined         = []
alldata_cvsn_baselined            = []

alldata_grandmean_t       = []
alldata_neutral_t         = []
alldata_cued_t            = []
alldata_cuedside_t        = []
alldata_error_t           = []
alldata_confidence_t      = []
alldata_DT_t              = []
alldata_errorxside_t      = []
alldata_confidencexside_t = []
alldata_DTxside_t         = []
alldata_cvsn_t            = []

alldata_grandmean_baselined_t       = []
alldata_neutral_baselined_t         = []
alldata_cued_baselined_t            = []
alldata_cuedside_baselined_t        = []
alldata_error_baselined_t           = []
alldata_confidence_baselined_t      = []
alldata_DT_baselined_t              = []
alldata_errorxside_baselined_t      = []
alldata_confidencexside_baselined_t = []
alldata_DTxside_baselined_t         = []
alldata_cvsn_baselined_t            = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    alldata_grandmean.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_grandmean_betas-tfr.h5'))[0])
    alldata_neutral.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_neutral_betas-tfr.h5'))[0])
    alldata_cued.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cued_betas-tfr.h5'))[0])
    alldata_cuedside.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedside_betas-tfr.h5'))[0])
    alldata_error.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_error_betas-tfr.h5'))[0])
    alldata_confidence.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidence_betas-tfr.h5'))[0])
    alldata_DT.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DT_betas-tfr.h5'))[0])
    alldata_errorxside.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_errorxside_betas-tfr.h5'))[0])
    alldata_confidencexside.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidencexside_betas-tfr.h5'))[0])
    alldata_DTxside.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DTxside_betas-tfr.h5'))[0])
    alldata_cvsn.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedvsneutral_betas-tfr.h5'))[0])
    
    
    alldata_grandmean_baselined.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_grandmean_betas_baselined-tfr.h5'))[0])
    alldata_neutral_baselined.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_neutral_betas_baselined-tfr.h5'))[0])
    alldata_cued_baselined.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cued_betas_baselined-tfr.h5'))[0])
    alldata_cuedside_baselined.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedside_betas_baselined-tfr.h5'))[0])
    alldata_error_baselined.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_error_betas_baselined-tfr.h5'))[0])
    alldata_confidence_baselined.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidence_betas_baselined-tfr.h5'))[0])
    alldata_DT_baselined.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DT_betas_baselined-tfr.h5'))[0])
    alldata_errorxside_baselined.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_errorxside_betas_baselined-tfr.h5'))[0])
    alldata_confidencexside_baselined.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidencexside_betas_baselined-tfr.h5'))[0])
    alldata_DTxside_baselined.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DTxside_betas_baselined-tfr.h5'))[0])
    alldata_cvsn_baselined.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedvsneutral_betas_baselined-tfr.h5'))[0])
    
    
    #tstats
    alldata_grandmean_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_grandmean_tstats-tfr.h5'))[0])
    alldata_neutral_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_neutral_tstats-tfr.h5'))[0])
    alldata_cued_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cued_tstats-tfr.h5'))[0])
    alldata_cuedside_t.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedside_tstats-tfr.h5'))[0])
    alldata_error_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_error_tstats-tfr.h5'))[0])
    alldata_confidence_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidence_tstats-tfr.h5'))[0])
    alldata_DT_t.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DT_tstats-tfr.h5'))[0])
    alldata_errorxside_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_errorxside_tstats-tfr.h5'))[0])
    alldata_confidencexside_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidencexside_tstats-tfr.h5'))[0])
    alldata_DTxside_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DTxside_tstats-tfr.h5'))[0])
    alldata_cvsn_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedvsneutral_tstats-tfr.h5'))[0])
    
    
    alldata_grandmean_baselined_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_grandmean_tstats_baselined-tfr.h5'))[0])
    alldata_neutral_baselined_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_neutral_tstats_baselined-tfr.h5'))[0])
    alldata_cued_baselined_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cued_tstats_baselined-tfr.h5'))[0])
    alldata_cuedside_baselined_t.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedside_tstats_baselined-tfr.h5'))[0])
    alldata_error_baselined_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_error_tstats_baselined-tfr.h5'))[0])
    alldata_confidence_baselined_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidence_tstats_baselined-tfr.h5'))[0])
    alldata_DT_baselined_t.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DT_tstats_baselined-tfr.h5'))[0])
    alldata_errorxside_baselined_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_errorxside_tstats_baselined-tfr.h5'))[0])
    alldata_confidencexside_baselined_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_confidencexside_tstats_baselined-tfr.h5'))[0])
    alldata_DTxside_baselined_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_DTxside_tstats_baselined-tfr.h5'))[0])
    alldata_cvsn_baselined_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cuedvsneutral_tstats_baselined-tfr.h5'))[0])

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

gave_cuedside = mne.grand_average(alldata_cuedside); gave_cuedside.drop_channels(['RM'])
gave_cuedside.plot_joint(title = 'cued left vs right, average betas',
                         timefreqs = timefreqs_alpha,
                         topomap_args = dict(outlines = 'head', contours = 0))

gave_cuedside = mne.grand_average(alldata_cuedside); gave_cuedside.data = toverparam(alldata_cuedside)
gave_cuedside.drop_channels(['RM'])
gave_cuedside.plot_joint(title = 'cued left vs right, t over betas',
                         timefreqs = timefreqs_alpha,
                         topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2),
                         vmin = -2, vmax = 2)

gave_cuedside_t = mne.grand_average(alldata_cuedside_t); gave_cuedside_t.drop_channels(['RM'])
gave_cuedside_t.plot_joint(title = 'cued left vs right, average tstats',
                          timefreqs = timefreqs_alpha,
                          topomap_args = dict(outlines ='head', contours = 0, vmin = -1, vmax = 1),
                          vmin = -1, vmax = 1)


gave_cuedside_t = mne.grand_average(alldata_cuedside_t); gave_cuedside_t.data = toverparam(alldata_cuedside_t)
gave_cuedside_t.drop_channels(['RM'])
gave_cuedside_t.plot_joint(title = 'cued left vs right, t over tstats',
                          timefreqs = timefreqs_alpha,
                          topomap_args = dict(outlines ='head', contours = 0, vmin = -2, vmax = 2),
                          vmin = -2, vmax = 2)


#can we do a spatiotemporal clustering on alpha here?
tmp = deepcopy(alldata_cuedside_t)
for i in tmp:
    i = i.crop(fmin = 8, fmax = 12) #restrict to alpha frequencies only
    i.drop_channels(['RM']) #and remove the right mastoid

temp = deepcopy(tmp)
for i in temp:
    i.data = np.nanmean(i.data, 1) #average across these frequencies now
    #output data shape is now just (channels, time) as we've collapsed down to just alpha 8=12Hz


connectivity, ch_names = mne.channels.find_ch_connectivity(tmp[0].info, ch_type = 'eeg')
#need to make an array that has (subjects, time, space) for the spatiotemporal clustering
#first make the array anyways
X = np.zeros(shape = (len(subs), temp[0].data.shape[0], temp[0].data.shape[1])) #create empty array to fill with data
X.fill(np.nan) #nan it first. now we'll populate it
for i in range(len(temp)):
    X[i,:,:] = temp[i].data
    
#now we need to switch around the dimensions to be (subjects, time, channels)
X = np.transpose(X, [0,2,1])
X_list = []
for i in range(X.shape[0]):
    X_list.append(np.squeeze(X[0,:,:]))
#t_obs, clusters, cluster_pv, H0 = mne.stats.spatio_temporal_cluster_1samp_test(X, 
#                                                                              connectivity = connectivity,
#                                                                               n_permutations = 1000)
#mask_alpha = np.asarray(clusters)[cluster_pv < 0.5]
#good_clusters = np.where(cluster_pv < 0.5)[0]

threshold = 10
p_accept = .05
t_obs, clusters, cluster_pv, H0 = mne.stats.spatio_temporal_cluster_test([X],
                                                                         n_permutations = 1000,
                                                                         threshold = threshold,
                                                                         connectivity = connectivity)
#I don't think this is the right way to set this up, i could really do with checking this out ... 





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
