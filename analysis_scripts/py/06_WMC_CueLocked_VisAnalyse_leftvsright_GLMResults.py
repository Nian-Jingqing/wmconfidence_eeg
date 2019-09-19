#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:27:09 2019

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
subs = np.array([         4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm


alldata_grandmean       = []
alldata_cleft           = []
alldata_cright          = []
alldata_cuedlvsr        = []
alldata_error           = []
alldata_confidence      = []
alldata_DT              = []
alldata_errorxside      = []
alldata_confidencexside = []
alldata_DTxside         = []

alldata_grandmean_t       = []
alldata_cleft_t           = []
alldata_cright_t          = []
alldata_cuedlvsr_t        = []
alldata_error_t           = []
alldata_confidence_t      = []
alldata_DT_t              = []
alldata_errorxside_t      = []
alldata_confidencexside_t = []
alldata_DTxside_t         = []



for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    alldata_grandmean.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_grandmean_betas-tfr.h5'))[0])
    alldata_cleft.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_cuedleft_betas-tfr.h5'))[0])
    alldata_cright.append(mne.time_frequency.read_tfrs(fname            = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_cuedright_betas-tfr.h5'))[0])
    alldata_cuedlvsr.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_cuedlvsr_betas-tfr.h5'))[0])
    alldata_error.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_error_betas-tfr.h5'))[0])
    alldata_confidence.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_confidence_betas-tfr.h5'))[0])
    alldata_DT.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_DT_betas-tfr.h5'))[0])
    alldata_errorxside.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_errorxside_betas-tfr.h5'))[0])
    alldata_confidencexside.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_confidencexside_betas-tfr.h5'))[0])
    alldata_DTxside.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_DTxside_betas-tfr.h5'))[0])
    
    #tstats
    alldata_grandmean_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_grandmean_tstats-tfr.h5'))[0])
    alldata_cleft_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_cuedleft_tstats-tfr.h5'))[0])
    alldata_cright_t.append(mne.time_frequency.read_tfrs(fname            = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_cuedright_tstats-tfr.h5'))[0])
    alldata_cuedlvsr_t.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_cuedlvsr_tstats-tfr.h5'))[0])
    alldata_error_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_error_tstats-tfr.h5'))[0])
    alldata_confidence_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_confidence_tstats-tfr.h5'))[0])
    alldata_DT_t.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_DT_tstats-tfr.h5'))[0])
    alldata_errorxside_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_errorxside_tstats-tfr.h5'))[0])
    alldata_confidencexside_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_confidencexside_tstats-tfr.h5'))[0])
    alldata_DTxside_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_DTxside_tstats-tfr.h5'))[0])

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
chnames_no_midline = np.array(alldata_grandmean[0].ch_names)[chans_no_midline]
#%%

gave_cleft = mne.grand_average(alldata_cleft); gave_cleft.drop_channels(['RM'])
gave_cleft.plot_joint(title = 'average betas, left-right for cued left trials', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0))


gave_cleft_t = mne.grand_average(alldata_cleft_t); gave_cleft_t.data = toverparam(alldata_cleft_t);

#normal plotting here will fail because the midline data was nan after the glm. we need to remove these channels and specify channels to plot
gave_cleft_t.plot_joint(title = 't over tstats, l-r, cued left', timefreqs = timefreqs_alpha, picks = chnames_no_midline,
                        topomap_args = dict(outlines = 'head', contours = 0))

gave_cright = mne.grand_average(alldata_cright); gave_cright.drop_channels(['RM'])
gave_cright.plot_joint(title = 'average betas, left-right hand side, cued right trials', timefreqs = timefreqs_alpha,
                       topomap_args = dict(outlines = 'head', contours = 0))

gave_cuedlvsr = mne.grand_average(alldata_cuedlvsr); gave_cuedlvsr.drop_channels(['RM'])
gave_cuedlvsr.plot_joint(title = 'average betas, left-right hand side, cued left vs right', timefreqs = timefreqs_alpha,
                         topomap_args = dict(outlines = 'head', contours = 0))

gave_cuedlvsr_t = mne.grand_average(alldata_cuedlvsr_t); gave_cuedlvsr_t.data = toverparam(alldata_cuedlvsr_t)
gave_cuedlvsr_t.plot_joint(title = 't over tstats, left-right hand side, cued left vs right', timefreqs =timefreqs_alpha, picks = chnames_no_midline,
                           topomap_args = dict(outlines = 'head', contours = 0, vmin = -3, vmax = 3))
#%%


gave_errorcside = mne.grand_average(alldata_errorxside); gave_errorcside.drop_channels(['RM'])
gave_errorcside.plot_joint(title = 'error x cued left vs right, left-right data, average betas', timefreqs = timefreqs_alpha,
                           topomap_args = dict(outlines='head', contours = 0))
gave_errorcside.plot(picks = visleftchans, combine = 'mean', title = 'vis left chans, error x cued left vs right, average betas')

gave_errorcside = mne.grand_average(alldata_errorxside); gave_errorcside.data = toverparam(alldata_errorxside); gave_errorcside.drop_channels(['RM'])
gave_errorcside.plot_joint(title = 'error x cued left vs right, left-right data, t over betas', timefreqs = timefreqs_alpha, picks = chans_no_midline,
                             topomap_args = dict(outlines='head', contours=0, vmin=-2, vmax=2))
gave_errorcside.plot(picks = visleftchans, combine = 'mean', title = 'vis left chans, error x cued left vs right, t over betas')


gave_errorcside_t = mne.grand_average(alldata_errorxside_t); gave_errorcside_t.data = toverparam(alldata_errorxside_t);
gave_errorcside_t.drop_channels(['RM'])
gave_errorcside_t.plot_joint(title = 'error x cued left vs right, left-right data, t over tstats', timefreqs = timefreqs_alpha, picks = chans_no_midline,
                             topomap_args = dict(outlines='head', contours=0, vmin=-2, vmax=2))
gave_errorcside_t.plot(picks = visleftchans, combine = 'mean', title = 'vis left chans, error x cued left vs right, t over tstats')

#%%


gave_confcside = mne.grand_average(alldata_confidencexside); gave_confcside.drop_channels(['RM'])
gave_confcside.plot_joint(title = 'confidence x cued left vs right, left-right data, ave betas', timefreqs = timefreqs_alpha,
                          topomap_args = dict(outlines = 'head', contours = 0))

gave_confcside = mne.grand_average(alldata_confidencexside); gave_confcside.data = toverparam(alldata_confidencexside); gave_confcside.drop_channels(['RM'])
gave_confcside.plot_joint(title = 'confidence x cued left vs right, left-right data, t over betas', timefreqs = timefreqs_alpha, picks=chans_no_midline,
                          topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax=2))



gave_confcside_t = mne.grand_average(alldata_confidencexside_t); gave_confcside_t.data = toverparam(alldata_confidencexside_t); gave_confcside_t.drop_channels(['RM'])
gave_confcside_t.plot_joint(title = 'confidence x cued left vs right, left-right data, t over tstats', timefreqs = timefreqs_alpha, picks=chans_no_midline,
                          topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax=2))

gave_confcside_t.plot(picks=visleftchans, combine='mean', title  = 'visleft chans, confidence x cued left vs right, left-right data, t over tstats')

#%%

gave_dtside = mne.grand_average(alldata_DTxside); gave_dtside.drop_channels(['RM'])
gave_dtside.plot_joint(title = 'DT x cued left vs right, left-right data, ave betas', timefreqs = timefreqs_alpha,
                       topomap_args = dict(outlines = 'head', contours = 0))


gave_dtside = mne.grand_average(alldata_DTxside); gave_dtside.data = toverparam(alldata_DTxside); gave_dtside.drop_channels(['RM'])
gave_dtside.plot_joint(title = 'DT x cued left vs right, left-right data, t over betas', timefreqs = timefreqs_alpha,picks = chans_no_midline,
                       topomap_args = dict(outlines = 'head', contours = 0))


gave_dtside_t = mne.grand_average(alldata_DTxside_t); gave_dtside_t.data = toverparam(alldata_DTxside_t); gave_dtside_t.drop_channels(['RM'])
gave_dtside_t.plot_joint(title = 'DT x cued left vs right, left-right data, t over tstats', timefreqs = timefreqs_alpha,picks = chans_no_midline,
                       topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

