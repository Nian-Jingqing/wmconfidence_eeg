#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:12:04 2019

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
alldata_psideneutral    = []
alldata_psidecued       = []
alldata_DT              = []
alldata_error           = []
alldata_pside           = []
alldata_DTxpside        = []
alldata_errorxpside     = []
alldata_conf            = []
alldata_confxpside      = []
alldata_confneut        = []
alldata_confcued        = []

alldata_grandmean_t       = []
alldata_neutral_t         = []
alldata_cued_t            = []
alldata_psideneutral_t    = []
alldata_psidecued_t       = []
alldata_DT_t              = []
alldata_error_t           = []
alldata_pside_t           = []
alldata_DTxpside_t        = []
alldata_errorxpside_t     = []
alldata_conf_t            = []
alldata_confxpside_t      = []
alldata_confneut_t        = []
alldata_confcued_t        = []

alldata_grandmean_baselined       = []
alldata_neutral_baselined         = []
alldata_cued_baselined            = []
alldata_psideneutral_baselined    = []
alldata_psidecued_baselined       = []
alldata_DT_baselined              = []
alldata_error_baselined           = []
alldata_pside_baselined           = []
alldata_DTxpside_baselined        = []
alldata_errorxpside_baselined     = []
alldata_conf_baselined            = []
alldata_confxpside_baselined      = []
alldata_confneut_baselined        = []
alldata_confcued_baselined        = []

alldata_grandmean_baselined_t       = []
alldata_neutral_baselined_t         = []
alldata_cued_baselined_t            = []
alldata_psideneutral_baselined_t    = []
alldata_psidecued_baselined_t       = []
alldata_DT_baselined_t              = []
alldata_error_baselined_t           = []
alldata_pside_baselined_t           = []
alldata_DTxpside_baselined_t        = []
alldata_errorxpside_baselined_t     = []
alldata_conf_baselined_t            = []
alldata_confxpside_baselined_t      = []
alldata_confneut_baselined_t        = []
alldata_confcued_baselined_t        = []



for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    alldata_grandmean.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_grandmean_betas-tfr.h5'))[0])
    alldata_neutral.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_neutral_betas-tfr.h5'))[0])
    alldata_cued.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_cued_betas-tfr.h5'))[0])
    alldata_psideneutral.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_psideneutral_betas-tfr.h5'))[0])
    alldata_psidecued.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_psidecued_betas-tfr.h5'))[0])
    alldata_DT.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_DT_betas-tfr.h5'))[0])
    alldata_error.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_error_betas-tfr.h5'))[0])
    alldata_pside.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_pside_betas-tfr.h5'))[0])
    alldata_DTxpside.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpside_betas-tfr.h5'))[0])
    alldata_errorxpside.append(mne.time_frequency.read_tfrs(fname       = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpside_betas-tfr.h5'))[0])
    alldata_conf.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidence_betas-tfr.h5'))[0])
    alldata_confxpside.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidencexpside_betas-tfr.h5'))[0])
    alldata_confneut.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidenceneutral_betas-tfr.h5'))[0])
    alldata_confcued.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidencecued_betas-tfr.h5'))[0])
    
    alldata_grandmean_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_grandmean_tstats-tfr.h5'))[0])
    alldata_neutral_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_neutral_tstats-tfr.h5'))[0])
    alldata_cued_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_cued_tstats-tfr.h5'))[0])
    alldata_psideneutral_t.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_psideneutral_tstats-tfr.h5'))[0])
    alldata_psidecued_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_psidecued_tstats-tfr.h5'))[0])
    alldata_DT_t.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_DT_tstats-tfr.h5'))[0])
    alldata_error_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_error_tstats-tfr.h5'))[0])
    alldata_pside_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_pside_tstats-tfr.h5'))[0])
    alldata_DTxpside_t.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpside_tstats-tfr.h5'))[0])
    alldata_errorxpside_t.append(mne.time_frequency.read_tfrs(fname       = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpside_tstats-tfr.h5'))[0])
    alldata_conf_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidence_tstats-tfr.h5'))[0])
    alldata_confxpside_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidencexpside_tstats-tfr.h5'))[0])
    alldata_confneut_t.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidenceneutral_tstats-tfr.h5'))[0])
    alldata_confcued_t.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidencecued_tstats-tfr.h5'))[0])
    
    
    #and the baselined versions ...
    alldata_grandmean_baselined.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_grandmean_betas_baselined-tfr.h5'))[0])
    alldata_neutral_baselined.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_neutral_betas_baselined-tfr.h5'))[0])
    alldata_cued_baselined.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_cued_betas_baselined-tfr.h5'))[0])
    alldata_psideneutral_baselined.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_psideneutral_betas_baselined-tfr.h5'))[0])
    alldata_psidecued_baselined.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_psidecued_betas_baselined-tfr.h5'))[0])
    alldata_DT_baselined.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_DT_betas_baselined-tfr.h5'))[0])
    alldata_error_baselined.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_error_betas_baselined-tfr.h5'))[0])
    alldata_pside_baselined.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_pside_betas_baselined-tfr.h5'))[0])
    alldata_DTxpside_baselined.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpside_betas_baselined-tfr.h5'))[0])
    alldata_errorxpside_baselined.append(mne.time_frequency.read_tfrs(fname       = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpside_betas_baselined-tfr.h5'))[0])
    alldata_conf_baselined.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidence_betas_baselined-tfr.h5'))[0])
    alldata_confxpside_baselined.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidencexpside_betas_baselined-tfr.h5'))[0])
    alldata_confneut_baselined.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidenceneutral_betas_baselined-tfr.h5'))[0])
    alldata_confcued_baselined.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidencecued_betas_baselined-tfr.h5'))[0])
    
    alldata_grandmean_baselined_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_grandmean_tstats_baselined-tfr.h5'))[0])
    alldata_neutral_baselined_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_neutral_tstats_baselined-tfr.h5'))[0])
    alldata_cued_baselined_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_cued_tstats_baselined-tfr.h5'))[0])
    alldata_psideneutral_baselined_t.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_psideneutral_tstats_baselined-tfr.h5'))[0])
    alldata_psidecued_baselined_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_psidecued_tstats_baselined-tfr.h5'))[0])
    alldata_DT_baselined_t.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_DT_tstats_baselined-tfr.h5'))[0])
    alldata_error_baselined_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_error_tstats_baselined-tfr.h5'))[0])
    alldata_pside_baselined_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_pside_tstats_baselined-tfr.h5'))[0])
    alldata_DTxpside_baselined_t.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpside_tstats_baselined-tfr.h5'))[0])
    alldata_errorxpside_baselined_t.append(mne.time_frequency.read_tfrs(fname       = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpside_tstats_baselined-tfr.h5'))[0])
    alldata_conf_baselined_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidence_tstats_baselined-tfr.h5'))[0])
    alldata_confxpside_baselined_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidencexpside_tstats_baselined-tfr.h5'))[0])
    alldata_confneut_baselined_t.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidenceneutral_tstats_baselined-tfr.h5'))[0])
    alldata_confcued_baselined_t.append(mne.time_frequency.read_tfrs(fname          = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_confidencecued_tstats_baselined-tfr.h5'))[0])
    
    
    
#%%
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#visualise some regressors
#we're not going to plot the grandmean, this shouldn't be in the glm as it doesnt make sense anyways.
#we will visualise the regressors for neutral and cued trials though as these should be different (no lateralisation in in neutral trials)
timefreqs_all = {(.4, 10):(.4, 4),
                 (.6, 10):(.4, 4),
                 (.8, 10):(.4, 4),
                 (.4, 22):(.4, 16),
                 (.6, 22):(.4, 16),
                 (.8, 22):(.4, 16)}

timefreqs_alpha = {(.4, 10):(.4, 4),
                   (.6, 10):(.4, 4),
                   (.8, 10):(.4, 4)} 

gave_gmean = mne.grand_average(alldata_grandmean_baselined); gave_gmean.drop_channels(['RM'])
gave_gmean.plot_joint(title = 'all trials, preglm baseline, average betas', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0))

gave_gmean = mne.grand_average(alldata_grandmean_baselined); gave_gmean.data = toverparam(alldata_grandmean_baselined)
gave_gmean.drop_channels(['RM'])
gave_gmean.plot_joint(title = 'all trials, preglm baseline, t over betas', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -4, vmax = 4))

gave_gmean_t = mne.grand_average(alldata_grandmean_baselined_t); gave_gmean_t.data = toverparam(alldata_grandmean_baselined_t)
gave_gmean_t.drop_channels(['RM'])
gave_gmean_t.plot_joint(title = 'all trials, preglm baseline, t over tstats', timefreqs = timefreqs_alpha,
                        topomap_args = dict(outlines = 'head', contours = 0, vmin = -3, vmax = 3))

#spacebar sensors (contra to it) = ['C2', 'C4', 'C6']
#mouse sensors (contra to it)    = ['C1', 'C3', 'C5'] 

#%%
# Question -- is probe related processing different if an item is already in mind vs not previously attended?


#neutral trials isn't a contrast, so need to take the baselined glm for this regressor
gave_neutral = mne.grand_average(alldata_neutral_baselined); gave_neutral.drop_channels(['RM'])
gave_neutral.plot_joint(title = 'neutral trials evoked response, average betas, preglm baselined', timefreqs = timefreqs_alpha,
                        topomap_args = dict(outlines = 'head', contours = 0))

#ttest over betas
gave_neutral = mne.grand_average(alldata_neutral_baselined); gave_neutral.data = toverparam(alldata_neutral_baselined); gave_neutral.drop_channels(['RM'])
gave_neutral.plot_joint(title = 'neutral trials evoked response, t over betas, preglm baselined', timefreqs = timefreqs_alpha,
                        topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

#ttest over tstats
gave_neutral_t = mne.grand_average(alldata_neutral_baselined_t); gave_neutral_t.data = toverparam(alldata_neutral_baselined_t); gave_neutral_t.drop_channels(['RM'])
gave_neutral_t.plot_joint(title = 'neutral trials evoked response, t over tstats, preglm baselined', timefreqs = timefreqs_alpha,
                          topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))


#cued trials also aren't a contrast, so take baselined glm for this regressor
gave_cued = mne.grand_average(alldata_cued_baselined); gave_cued.drop_channels(['RM'])
gave_cued.plot_joint(title = 'cued trials evoked response, average betas, preglm baseline', timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0))

#ttest over betas
gave_cued = mne.grand_average(alldata_cued_baselined); gave_cued.data = toverparam(alldata_cued_baselined); gave_cued.drop_channels(['RM'])
gave_cued.plot_joint(title = 'cued trials evoked response, t over betas, preglm baseline', timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

#ttest over tstats
gave_cued_t = mne.grand_average(alldata_cued_baselined_t); gave_cued_t.data = toverparam(alldata_cued_baselined_t); gave_cued_t.drop_channels(['RM'])
gave_cued_t.plot_joint(title = 'cued trials evoked reponse, t over tstats, preglm baseline', timefreqs = timefreqs_alpha,
                       topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))


#%%
#Question -- is probe processing different 























#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#look at lateralisation of probe evoked response in cued trials (maybe will explain we'rd overall cued trial response to probe)
gave_psidecued = mne.grand_average(alldata_psidecued); gave_psidecued.drop_channels(['RM'])
gave_psidecued.plot_joint(title = 'probed left vs right cued trials only, average betas',
                             timefreqs = timefreqs_alpha, topomap_args = dict(outlines = 'head', contours = 0))

gave_psidecued = mne.grand_average(alldata_psidecued); gave_psidecued.data = toverparam(alldata_psidecued);
gave_psidecued.drop_channels(['RM'])
gave_psidecued.plot_joint(title = 'probed left vs right cued trials only, t over betas',
                             timefreqs = timefreqs_alpha, topomap_args = dict(outlines = 'head', contours = 0, vmin=-2, vmax=2))


gave_pside_cued_t = mne.grand_average(alldata_psidecued_t);
gave_pside_cued_t.data = toverparam(alldata_psidecued_t); gave_pside_cued_t.drop_channels(['RM'])
gave_pside_cued_t.plot_joint(title = 'probed left vs right, cued trials only, t over tstats', timefreqs = timefreqs_alpha,
                                topomap_args = dict(outlines = 'head', contours = 0, vmin=-2, vmax=2))



gave_psideneutral = mne.grand_average(alldata_psideneutral); gave_psideneutral.drop_channels(['RM'])
gave_psideneutral.plot_joint(title = 'probed left vs right neutral trials only, average betas',
                             timefreqs = timefreqs_alpha, topomap_args = dict(outlines = 'head', contours = 0))

gave_psideneutral = mne.grand_average(alldata_psideneutral); gave_psideneutral.data = toverparam(alldata_psideneutral);
gave_psideneutral.drop_channels(['RM'])
gave_psideneutral.plot_joint(title = 'probed left vs right neutral trials only, t over betas',
                             timefreqs = timefreqs_alpha, topomap_args = dict(outlines = 'head', contours = 0, vmin=-2,vmax=2))

gave_pside_neutral_t = mne.grand_average(alldata_psideneutral_t);
gave_pside_neutral_t.data = toverparam(alldata_psideneutral_t); gave_pside_neutral_t.drop_channels(['RM'])
gave_pside_neutral_t.plot_joint(title = 'probed left vs right, neutral trials only, t over tstats', timefreqs = timefreqs_alpha,
                                topomap_args = dict(outlines = 'head', contours = 0, vmin=-2, vmax=2))



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#side of the probed item (left vs right) across all trials (neutral and cued)
gave_pside = mne.grand_average(alldata_pside); gave_pside.drop_channels(['RM'])
gave_pside.plot_joint(title = 'probed left vs right item, average betas', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0))

gave_pside_t = mne.grand_average(alldata_pside_t); gave_pside_t.drop_channels(['RM'])
gave_pside_t.plot_joint(title = 'probed left vs right item, average tstats', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0))


gave_pside = mne.grand_average(alldata_pside); gave_pside.data = toverparam(alldata_pside); gave_pside.drop_channels(['RM'])
gave_pside.plot_joint(title = 'probed left vs right item, t over betas', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_pside_t = mne.grand_average(alldata_pside_t); gave_pside_t.data = toverparam(alldata_pside_t); gave_pside_t.drop_channels(['RM'])
gave_pside_t.plot_joint(title = 'probed left vs right item, t over tstats', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))


#---------------------------------------------------------------------------------------------------

#main effect of error
#this isn't carried by a contrast at this point, so we need the baselined version of the glm for this regressor
gave_error = mne.grand_average(alldata_error_baselined); gave_error.drop_channels(['RM'])
gave_error.plot_joint(title = 'main effect of error, average betas, preglm baseline', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0))

#ttest over betas
gave_error = mne.grand_average(alldata_error_baselined); gave_error.data = toverparam(alldata_error_baselined); gave_error.drop_channels(['RM'])
gave_error.plot_joint(title = 'main effect of error, t over betas, preglm baseline', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

#ttest over tstats
gave_error_t = mne.grand_average(alldata_error_baselined_t); gave_error_t.data = toverparam(alldata_error_baselined_t); gave_error_t.drop_channels(['RM'])
gave_error_t.plot_joint(title = 'main effect of error, t over tstats, preglm baseline', timefreqs = timefreqs_alpha,
                        topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

#main effect of reaction time (time taken to press space bar and start response phase)

gave_dt = mne.grand_average(alldata_DT_baselined); gave_dt.drop_channels(['RM'])
gave_dt.plot_joint(title = 'main effect of decision time, average betas, preglm baseline', timefreqs = timefreqs_alpha,
                   topomap_args = dict(outlines = 'head', contours = 0))

gave_dt = mne.grand_average(alldata_DT_baselined); gave_dt.data = toverparam(alldata_DT_baselined); gave_dt.drop_channels(['RM'])
gave_dt.plot_joint(title = 'main effect of decision time, t over betas, preglm baseline', timefreqs = timefreqs_alpha,
                   topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_dt_t = mne.grand_average(alldata_DT_baselined_t); gave_dt_t.data = toverparam(alldata_DT_baselined_t); gave_dt_t.drop_channels(['RM'])
gave_dt_t.plot_joint(title = 'main effect of decision time, t over tstats, preglm baseline', timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))


#main effect of confidence
#also not carried by a contrast at this point, so baselined version of glm is used here

gave_conf = mne.grand_average(alldata_conf_baselined); gave_conf.drop_channels(['RM'])
gave_conf.plot_joint(title = 'main effect of confidence, average betas, preglm baseline', timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0))

gave_conf = mne.grand_average(alldata_conf_baselined); gave_conf.data = toverparam(alldata_conf_baselined); gave_conf.drop_channels(['RM'])
gave_conf.plot_joint(title = 'main effect of confidence, t over betas, preglm baseline', timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_conf_t = mne.grand_average(alldata_conf_baselined_t); gave_conf_t.data = toverparam(alldata_conf_baselined_t); gave_conf_t.drop_channels(['RM'])
gave_conf_t.plot_joint(title = 'main effect of confidence, t over tstats, preglm baseline', timefreqs = timefreqs_alpha,
                       topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))


#now look at confidence carried by left vs right probed contrast

gave_confpside = mne.grand_average(alldata_confxpside); gave_confpside.data = toverparam(alldata_confxpside); gave_confpside.drop_channels(['RM'])
gave_confpside.plot_joint(title = 'main effect of confidence x probed side, t over betas', timefreqs = timefreqs_alpha,
                          topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))




#---------------------------------------------------------------------------------------------------


#confidence separately for neutral and cued trials

gave_conf_neutral = mne.grand_average(alldata_confneut_baselined); gave_conf_neutral.drop_channels(['RM'])
gave_conf_neutral.plot_joint(title = 'main effect confidence, neutral trials only, average betas, preglm baseline', timefreqs = timefreqs_alpha,
                             topomap_args = dict(outlines = 'head', contours = 0))

gave_conf_neutral = mne.grand_average(alldata_confneut_baselined); gave_conf_neutral.data = toverparam(alldata_confneut_baselined); gave_conf_neutral.drop_channels(['RM'])
gave_conf_neutral.plot_joint(title = 'main effect confidence, neutral trials only, t over betas, preglm baseline', timefreqs = timefreqs_alpha,
                             topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_conf_neutral_t = mne.grand_average(alldata_confneut_baselined_t); gave_conf_neutral_t.data = toverparam(alldata_confneut_baselined_t); gave_conf_neutral_t.drop_channels(['RM'])
gave_conf_neutral_t.plot_joint(title = 'main effect confidence, neutral trials only, t over tstats, preglm baseline', timefreqs = timefreqs_alpha,
                               topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_conf_cued = mne.grand_average(alldata_confcued_baselined); gave_conf_cued.drop_channels(['RM'])
gave_conf_cued.plot_joint(title = 'main effect confidence, cued trials only, average betas, preglm baseline', timefreqs = timefreqs_alpha,
                             topomap_args = dict(outlines = 'head', contours = 0))

gave_conf_cued = mne.grand_average(alldata_confcued_baselined); gave_conf_cued.data = toverparam(alldata_confcued_baselined); gave_conf_cued.drop_channels(['RM'])
gave_conf_cued.plot_joint(title = 'main effect confidence, cued trials only, t over betas, preglm baseline', timefreqs = timefreqs_alpha,
                             topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_conf_cued_t = mne.grand_average(alldata_confcued_baselined_t); gave_conf_cued_t.data = toverparam(alldata_confcued_baselined_t); gave_conf_cued_t.drop_channels(['RM'])
gave_conf_cued_t.plot_joint(title = 'main effect confidence, cued trials only, t over tstats, preglm baseline', timefreqs = timefreqs_alpha,
                               topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#error interaction with probed side (left vs right)
gave_errorxpside = mne.grand_average(alldata_errorxpside); gave_errorxpside.drop_channels(['RM'])
gave_errorxpside.plot_joint(title = 'error x probed left vs right item, epoch demean, average betas', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0))

gave_errorxpside_t = mne.grand_average(alldata_errorxpside_t); gave_errorxpside_t.drop_channels(['RM'])
gave_errorxpside_t.plot_joint(title = 'error x probed left vs right item, average tstats', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0))


gave_errorxpside = mne.grand_average(alldata_errorxpside); gave_errorxpside.data = toverparam(alldata_errorxpside); gave_errorxpside.drop_channels(['RM'])
gave_errorxpside.plot_joint(title = 'error x probed left vs right item, t over betas', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_errorxpside_t = mne.grand_average(alldata_errorxpside_t); gave_errorxpside_t.data = toverparam(alldata_errorxpside_t); gave_errorxpside_t.drop_channels(['RM'])
gave_errorxpside_t.plot_joint(title = 'error x probed left vs right item, t over tstats', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

#decision time interaction with probed side (left vs right)
gave_dtxpside = mne.grand_average(alldata_DTxpside); gave_dtxpside.drop_channels(['RM'])
gave_dtxpside.plot_joint(title = 'DT x probed side (left vs right), average betas', timefreqs = timefreqs_alpha,
                         topomap_args = dict(outlines = 'head', contours = 0))

gave_dtxpside = mne.grand_average(alldata_DTxpside); gave_dtxpside.data = toverparam(alldata_DTxpside); gave_dtxpside.drop_channels(['RM'])
gave_dtxpside.plot_joint(title = 'DT x probed side (lvsr), t over betas', timefreqs = timefreqs_alpha,
                         topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))


gave_dtxpside_t = mne.grand_average(alldata_DTxpside_t); gave_dtxpside_t.drop_channels(['RM'])
gave_dtxpside_t.plot_joint(title = 'DT x probed side (left vs right), average tstats', timefreqs = timefreqs_alpha,
                         topomap_args = dict(outlines = 'head', contours = 0))

gave_dtxpside_t = mne.grand_average(alldata_DTxpside_t); gave_dtxpside_t.data = toverparam(alldata_DTxpside_t); gave_dtxpside_t.drop_channels(['RM'])
gave_dtxpside_t.plot_joint(title = 'DT x probed side (left vs right), t over tstats', timefreqs = timefreqs_alpha,
                         topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

#confidence interaction with probed side (left vs right)
gave_confpside = mne.grand_average(alldata_confxpside); gave_confpside.drop_channels(['RM'])
gave_confpside.plot_joint(title = 'effect of confidence as a function of probed left vs right, average betas no baseline', timefreqs = timefreqs_alpha,
                          topomap_args = dict(outlines = 'head', contours = 0))


gave_confpside = mne.grand_average(alldata_confxpside); gave_confpside.data = toverparam(alldata_confxpside)
gave_confpside.drop_channels(['RM'])
gave_confpside.plot_joint(title = 'interaction between confidence and probed side (lvsr), t over betas, no baseline', timefreqs = timefreqs_alpha,
                          topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_confpside_t = mne.grand_average(alldata_confxpside_t); gave_confpside_t.data = toverparam(alldata_confxpside_t);
gave_confpside_t.drop_channels(['RM'])
gave_confpside_t.plot_joint(title= ' effect of confidence x probed side (lvsr), t over tstats no baseline', timefreqs = timefreqs_alpha,
                            topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))












