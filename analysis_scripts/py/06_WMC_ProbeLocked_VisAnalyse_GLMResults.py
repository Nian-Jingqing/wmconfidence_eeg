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

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
subs = np.array([       4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm


alldata_grandmean       = []
alldata_neutral         = []
alldata_cued            = []
alldata_psideneutral    = []
alldata_psidecued       = []
alldata_DT              = []
alldata_error           = []
alldata_conf            = []
alldata_pside           = []
alldata_psidexcvsn      = []
alldata_error_cvsn      = []
alldata_dt_cvsn         = []
alldata_conf_cvsn       = []
alldata_errorxpside_neut = []
alldata_errorxpside_cued = []
alldata_errorxpside_cvsn = []
alldata_dtxpside_neut = []
alldata_dtxpside_cued = []
alldata_dtxpside_cvsn = []
alldata_confxpside_neut = []
alldata_confxpside_cued = []
alldata_confxpside_cvsn = []


alldata_grandmean_t       = []
alldata_neutral_t         = []
alldata_cued_t            = []
alldata_psideneutral_t    = []
alldata_psidecued_t       = []
alldata_DT_t              = []
alldata_error_t           = []
alldata_conf_t            = []
alldata_pside_t           = []
alldata_psidexcvsn_t      = []
alldata_error_cvsn_t      = []
alldata_dt_cvsn_t         = []
alldata_conf_cvsn_t       = []
alldata_errorxpside_neut_t = []
alldata_errorxpside_cued_t = []
alldata_errorxpside_cvsn_t = []
alldata_dtxpside_neut_t = []
alldata_dtxpside_cued_t = []
alldata_dtxpside_cvsn_t = []
alldata_confxpside_neut_t = []
alldata_confxpside_cued_t = []
alldata_confxpside_cvsn_t = []

alldata_grandmean_baselined       = []
alldata_neutral_baselined         = []
alldata_cued_baselined            = []
alldata_psideneutral_baselined    = []
alldata_psidecued_baselined       = []
alldata_DT_baselined              = []
alldata_error_baselined           = []
alldata_conf_baselined            = []
alldata_pside_baselined           = []
alldata_psidexcvsn_baselined      = []
alldata_error_cvsn_baselined      = []
alldata_dt_cvsn_baselined         = []
alldata_conf_cvsn_baselined       = []
alldata_errorxpside_neut_baselined = []
alldata_errorxpside_cued_baselined = []
alldata_errorxpside_cvsn_baselined = []
alldata_dtxpside_neut_baselined = []
alldata_dtxpside_cued_baselined = []
alldata_dtxpside_cvsn_baselined = []
alldata_confxpside_neut_baselined = []
alldata_confxpside_cued_baselined = []
alldata_confxpside_cvsn_baselined = []


alldata_grandmean_baselined_t       = []
alldata_neutral_baselined_t         = []
alldata_cued_baselined_t            = []
alldata_psideneutral_baselined_t    = []
alldata_psidecued_baselined_t       = []
alldata_DT_baselined_t              = []
alldata_error_baselined_t           = []
alldata_conf_baselined_t            = []
alldata_pside_baselined_t           = []
alldata_psidexcvsn_baselined_t      = []
alldata_error_cvsn_baselined_t      = []
alldata_dt_cvsn_baselined_t         = []
alldata_conf_cvsn_baselined_t       = []
alldata_errorxpside_neut_baselined_t = []
alldata_errorxpside_cued_baselined_t = []
alldata_errorxpside_cvsn_baselined_t = []
alldata_dtxpside_neut_baselined_t = []
alldata_dtxpside_cued_baselined_t = []
alldata_dtxpside_cvsn_baselined_t = []
alldata_confxpside_neut_baselined_t = []
alldata_confxpside_cued_baselined_t = []
alldata_confxpside_cvsn_baselined_t = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
#    alldata_grandmean.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_grandmean_betas-tfr.h5'))[0])
#    alldata_neutral.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_neutral_betas-tfr.h5'))[0])
#    alldata_cued.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_cued_betas-tfr.h5'))[0])
#    alldata_psideneutral.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_psideneutral_betas-tfr.h5'))[0])
#    alldata_psidecued.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_psidecued_betas-tfr.h5'))[0])
#    alldata_DT.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DT_betas-tfr.h5'))[0])
#    alldata_error.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_error_betas-tfr.h5'))[0])
#    alldata_pside.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_pside_betas-tfr.h5'))[0])
#    alldata_conf.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confidence_betas-tfr.h5'))[0])
#    alldata_errorxpside_neut.append(mne.time_frequency.read_tfrs(fname  = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpside_neutral_betas-tfr.h5'))[0])
#    alldata_errorxpside_cued.append(mne.time_frequency.read_tfrs(fname  = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpside_cued_betas-tfr.h5'))[0])
#    alldata_errorxpside_cvsn.append(mne.time_frequency.read_tfrs(fname  = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpsidexcvsn_betas-tfr.h5'))[0])
#    alldata_dtxpside_neut.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpside_neutral_betas-tfr.h5'))[0])
#    alldata_dtxpside_cued.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpside_cued_betas-tfr.h5'))[0])
#    alldata_dtxpside_cvsn.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpsidexcvsn_betas-tfr.h5'))[0])
#    alldata_confxpside_neut.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confxpside_neutral_betas-tfr.h5'))[0])
#    alldata_confxpside_cued.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confxpside_cued_betas-tfr.h5'))[0])
#    alldata_confxpside_cvsn.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confxpsidexcvsn_betas-tfr.h5'))[0])
#    alldata_error_cvsn.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_error_cvsn_betas-tfr.h5'))[0])
#    alldata_dt_cvsn.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DT_cvsn_betas-tfr.h5'))[0])
#    alldata_conf_cvsn.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_conf_cvsn_betas-tfr.h5'))[0])
    
    alldata_grandmean_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_grandmean_tstats-tfr.h5'))[0])
    alldata_neutral_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_neutral_tstats-tfr.h5'))[0])
    alldata_cued_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_cued_tstats-tfr.h5'))[0])
    alldata_psideneutral_t.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_psideneutral_tstats-tfr.h5'))[0])
    alldata_psidecued_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_psidecued_tstats-tfr.h5'))[0])
    alldata_DT_t.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DT_tstats-tfr.h5'))[0])
    alldata_error_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_error_tstats-tfr.h5'))[0])
    alldata_pside_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_pside_tstats-tfr.h5'))[0])
    alldata_conf_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confidence_tstats-tfr.h5'))[0])
    alldata_errorxpside_neut_t.append(mne.time_frequency.read_tfrs(fname  = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpside_neutral_tstats-tfr.h5'))[0])
    alldata_errorxpside_cued_t.append(mne.time_frequency.read_tfrs(fname  = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpside_cued_tstats-tfr.h5'))[0])
    alldata_errorxpside_cvsn_t.append(mne.time_frequency.read_tfrs(fname  = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpsidexcvsn_tstats-tfr.h5'))[0])
    alldata_dtxpside_neut_t.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpside_neutral_tstats-tfr.h5'))[0])
    alldata_dtxpside_cued_t.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpside_cued_tstats-tfr.h5'))[0])
    alldata_dtxpside_cvsn_t.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpsidexcvsn_tstats-tfr.h5'))[0])
    alldata_confxpside_neut_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confxpside_neutral_tstats-tfr.h5'))[0])
    alldata_confxpside_cued_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confxpside_cued_tstats-tfr.h5'))[0])
    alldata_confxpside_cvsn_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confxpsidexcvsn_tstats-tfr.h5'))[0])
    alldata_error_cvsn_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_error_cvsn_tstats-tfr.h5'))[0])
    alldata_dt_cvsn_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DT_cvsn_tstats-tfr.h5'))[0])
    alldata_conf_cvsn_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_conf_cvsn_tstats-tfr.h5'))[0])
    
    
#    alldata_grandmean_baselined.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_grandmean_betas_baselined-tfr.h5'))[0])
#    alldata_neutral_baselined.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_neutral_betas_baselined-tfr.h5'))[0])
#    alldata_cued_baselined.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_cued_betas_baselined-tfr.h5'))[0])
#    alldata_psideneutral_baselined.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_psideneutral_betas_baselined-tfr.h5'))[0])
#    alldata_psidecued_baselined.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_psidecued_betas_baselined-tfr.h5'))[0])
#    alldata_DT_baselined.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DT_betas_baselined-tfr.h5'))[0])
#    alldata_error_baselined.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_error_betas_baselined-tfr.h5'))[0])
#    alldata_pside_baselined.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_pside_betas_baselined-tfr.h5'))[0])
#    alldata_conf_baselined.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confidence_betas_baselined-tfr.h5'))[0])
#    alldata_errorxpside_neut_baselined.append(mne.time_frequency.read_tfrs(fname  = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpside_neutral_betas_baselined-tfr.h5'))[0])
#    alldata_errorxpside_cued_baselined.append(mne.time_frequency.read_tfrs(fname  = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpside_cued_betas_baselined-tfr.h5'))[0])
#    alldata_errorxpside_cvsn_baselined.append(mne.time_frequency.read_tfrs(fname  = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpsidexcvsn_betas_baselined-tfr.h5'))[0])
#    alldata_dtxpside_neut_baselined.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpside_neutral_betas_baselined-tfr.h5'))[0])
#    alldata_dtxpside_cued_baselined.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpside_cued_betas_baselined-tfr.h5'))[0])
#    alldata_dtxpside_cvsn_baselined.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpsidexcvsn_betas_baselined-tfr.h5'))[0])
#    alldata_confxpside_neut_baselined.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confxpside_neutral_betas_baselined-tfr.h5'))[0])
#    alldata_confxpside_cued_baselined.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confxpside_cued_betas_baselined-tfr.h5'))[0])
#    alldata_confxpside_cvsn_baselined.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confxpsidexcvsn_betas_baselined-tfr.h5'))[0])
#    alldata_error_cvsn_baselined.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_error_cvsn_betas_baselined-tfr.h5'))[0])
#    alldata_dt_cvsn_baselined.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DT_cvsn_betas_baselined-tfr.h5'))[0])
#    alldata_conf_cvsn_baselined.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_conf_cvsn_betas_baselined-tfr.h5'))[0])
    
    alldata_grandmean_baselined_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_grandmean_tstats_baselined-tfr.h5'))[0])
    alldata_neutral_baselined_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_neutral_tstats_baselined-tfr.h5'))[0])
    alldata_cued_baselined_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_cued_tstats_baselined-tfr.h5'))[0])
    alldata_psideneutral_baselined_t.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_psideneutral_tstats_baselined-tfr.h5'))[0])
    alldata_psidecued_baselined_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_psidecued_tstats_baselined-tfr.h5'))[0])
    alldata_DT_baselined_t.append(mne.time_frequency.read_tfrs(fname                = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DT_tstats_baselined-tfr.h5'))[0])
    alldata_error_baselined_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_error_tstats_baselined-tfr.h5'))[0])
    alldata_pside_baselined_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_pside_tstats_baselined-tfr.h5'))[0])
    alldata_conf_baselined_t.append(mne.time_frequency.read_tfrs(fname              = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confidence_tstats_baselined-tfr.h5'))[0])
    alldata_errorxpside_neut_baselined_t.append(mne.time_frequency.read_tfrs(fname  = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpside_neutral_tstats_baselined-tfr.h5'))[0])
    alldata_errorxpside_cued_baselined_t.append(mne.time_frequency.read_tfrs(fname  = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpside_cued_tstats_baselined-tfr.h5'))[0])
    alldata_errorxpside_cvsn_baselined_t.append(mne.time_frequency.read_tfrs(fname  = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_errorxpsidexcvsn_tstats_baselined-tfr.h5'))[0])
    alldata_dtxpside_neut_baselined_t.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpside_neutral_tstats_baselined-tfr.h5'))[0])
    alldata_dtxpside_cued_baselined_t.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpside_cued_tstats_baselined-tfr.h5'))[0])
    alldata_dtxpside_cvsn_baselined_t.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DTxpsidexcvsn_tstats_baselined-tfr.h5'))[0])
    alldata_confxpside_neut_baselined_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confxpside_neutral_tstats_baselined-tfr.h5'))[0])
    alldata_confxpside_cued_baselined_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confxpside_cued_tstats_baselined-tfr.h5'))[0])
    alldata_confxpside_cvsn_baselined_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_confxpsidexcvsn_tstats_baselined-tfr.h5'))[0])
    alldata_error_cvsn_baselined_t.append(mne.time_frequency.read_tfrs(fname        = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_error_cvsn_tstats_baselined-tfr.h5'))[0])
    alldata_dt_cvsn_baselined_t.append(mne.time_frequency.read_tfrs(fname           = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_DT_cvsn_tstats_baselined-tfr.h5'))[0])
    alldata_conf_cvsn_baselined_t.append(mne.time_frequency.read_tfrs(fname         = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_conf_cvsn_tstats_baselined-tfr.h5'))[0])
    
    
    
    
#%%
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#visualise some regressors
#we're not going to plot the grandmean, this shouldn't be in the glm as it doesnt make sense anyways.
#we will visualise the regressors for neutral and cued trials though as these should be different (no lateralisation in in neutral trials)
from copy import deepcopy
timefreqs_all = {(.4, 10):(.4, 4),
                 (.6, 10):(.4, 4),
                 (.8, 10):(.4, 4),
                 (.4, 22):(.4, 16),
                 (.6, 22):(.4, 16),
                 (.8, 22):(.4, 16)}

timefreqs_alpha = {(.4, 10):(.4, 4),
                   (.6, 10):(.4, 4),
                   (.8, 10):(.4, 4)}

timefreqs_cue = {(-1.5, 10):(.4, 4),
                 (-1.3, 10):(.4, 4),
                 (-1.1, 10):(.4, 4),
                 (-0.9, 10):(.4, 4),
                 (-0.7, 10):(.4, 4),
                 (-0.5, 10):(.4, 4),
                 (-.3, 10):(.4, 4)}


vischans = np.array(['PO4', 'PO8', 'O2', 'O1', 'PO3', 'PO7'])
visrightchans = np.array(['PO4', 'PO8', 'O2'])
visleftchans  = np.array(['PO3', 'PO7', 'O1'])

motrightchans= ['C4', 'C2']
motleftchans = ['C3', 'C5']
#%%
#get some behavioural data that we should probably plot to make some sense of things
allbdata = []
for i in subs:
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    bdata = pd.read_csv(param['probelocked_tfr_meta'], index_col = None, header = 0) #read the metadata associated with this subject
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
#%%

#in this, probed side is left vs right, and cue is just cued or not
# so cued trials should have an evoked lateralisation, and neutral trials shouldn't.
# probed side cvsn is the difference in lateralisation due to cue vs random
alldata_pside_cvsn_t = []
for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    alldata_pside_cvsn_t.append(mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm','wmConfidence_' + param['subid'] + '_probelocked_tfr_psidexcvsn_tstats-tfr.h5'))[0])

gave_psidecvsn = mne.grand_average(alldata_pside_cvsn_t); gave_psidecvsn.data = toverparam(alldata_pside_cvsn_t); gave_psidecvsn.drop_channels(['RM'])
gave_pside = mne.grand_average(alldata_pside_t); gave_pside.data = toverparam(alldata_pside_t); gave_pside.drop_channels(['RM'])

gave_pside.plot_joint(timefreqs = timefreqs_cue, topomap_args = dict(outlines='head', contours=0, vmin=-2, vmax=2))
gave_psidecvsn.plot_joint(timefreqs = timefreqs_cue, topomap_args = dict(outlines='head', contours=0, vmin=-2, vmax=2))

#%%

#global state effects on error?

gave_error = mne.grand_average(alldata_error_baselined_t); gave_error.data = toverparam(alldata_error_baselined_t); gave_error.drop_channels(['RM'])
gave_error.plot_joint(timefreqs=timefreqs_cue, topomap_args = dict(outlines='head', contours=0, vmin=-2, vmax=2), vmin=-2, vmax=2)

times, freqs = gave_error.times, gave_error.freqs

fig = plt.figure()
ax = fig.subplots(1,1)
ax.contourf(times, freqs, np.nanmean(deepcopy(gave_error).pick_channels(vischans).data, 0), vmin=-2, vmax=2, levels=100, cmap ='RdBu_r', antialiased=False)
ax.vlines([-1.5, 0], linestyle='--', lw=.75, ymin=1, ymax=39)

x_error_vis = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_error_vis[i,:,:] = np.nanmean(deepcopy(alldata_error_baselined_t[i]).pick_channels(vischans).data,0)

t_error_vis, clusters_error_vis, clusters_pv_error_vis, _ = mne.stats.permutation_cluster_1samp_test(x_error_vis, n_permutations='all')


#same for decision time?
gave_dt = mne.grand_average(alldata_DT_baselined_t); gave_dt.data = toverparam(alldata_DT_baselined_t); gave_dt.drop_channels(['RM'])
gave_dt.plot_joint(timefreqs=timefreqs_cue, topomap_args = dict(outlines='head', contours=0, vmin=-2, vmax=2), vmin=-2, vmax=2)

times, freqs = gave_dt.times, gave_dt.freqs

fig = plt.figure()
ax = fig.subplots(1,1)
ax.contourf(times, freqs, np.nanmean(deepcopy(gave_dt).pick_channels(vischans).data, 0), vmin=-2, vmax=2, levels=100, cmap ='RdBu_r', antialiased=False)
ax.vlines([-1.5, 0], linestyle='--', lw=.75, ymin=1, ymax=39)

x_dt_vis = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_dt_vis[i,:,:] = np.nanmean(deepcopy(alldata_DT_baselined_t[i]).pick_channels(vischans).data,0)

t_dt_vis, clusters_dt_vis, clusters_pv_dt_vis, _ = mne.stats.permutation_cluster_1samp_test(x_dt_vis, n_permutations='all')
masks_dt_vis = np.asarray(clusters_dt_vis)[clusters_pv_dt_vis <0.05]

fig = plt.figure()
ax = fig.subplots(1,1)
ax.contourf(times, freqs, np.nanmean(deepcopy(gave_dt).pick_channels(vischans).data, 0), vmin=-2, vmax=2, levels=100, cmap ='RdBu_r', antialiased=False)
ax.vlines([-1.5, 0, gave_mean_dt], linestyle='--', lw=1, ymin=1, ymax=39)
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time relative to probe onset (s)')
for mask in masks_dt_vis:
    bigmask = np.kron(mask, np.ones((10,10)))
    ax.contour(bigmask, colors='black', lw=.75, extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), antialiased=False)



#same for confidence?
gave_conf = mne.grand_average(alldata_conf_baselined_t); gave_conf.data = toverparam(alldata_conf_baselined_t); gave_conf.drop_channels(['RM'])

tmin, tmax = None, None
gave_conf.crop(tmin=tmin, tmax = tmax)
gave_conf.plot_joint(timefreqs=timefreqs_cue, topomap_args = dict(outlines='head', contours=0, vmin=-2, vmax=2), vmin=-2, vmax=2)

times, freqs = gave_conf.times, gave_conf.freqs

fig = plt.figure()
ax = fig.subplots(1,1)
ax.contourf(times, freqs, np.nanmean(deepcopy(gave_conf).pick_channels(vischans).data, 0), vmin=-2, vmax=2, levels=100, cmap ='RdBu_r', antialiased=False)
ax.vlines([0, gave_mean_dt], linestyle='--', lw=.75, ymin=1, ymax=39)

x_conf_vis = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_conf_vis[i,:,:] = np.nanmean(deepcopy(alldata_conf_baselined_t[i]).crop(tmin=tmin,tmax=tmax).pick_channels(vischans).data,0)

t_conf_vis, clusters_conf_vis, clusters_pv_conf_vis, _ = mne.stats.permutation_cluster_1samp_test(x_conf_vis, n_permutations='all')
masks_conf_vis = np.asarray(clusters_conf_vis)[clusters_pv_conf_vis <0.05]

fig = plt.figure()
ax = fig.subplots(1,1)
ax.contourf(times, freqs, np.nanmean(deepcopy(gave_conf).pick_channels(vischans).data, 0), vmin=-2, vmax=2, levels=100, cmap ='RdBu_r', antialiased=False)
ax.vlines([-1.5, 0, gave_mean_dt], linestyle='--', lw=1, ymin=1, ymax=39)
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time relative to probe onset (s)')
for mask in masks_conf_vis:
    bigmask = np.kron(mask, np.ones((10,10)))
    ax.contour(bigmask, colors='black', lw=.75, extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), antialiased=False)

#%%
 
gave_errorpside_cued = mne.grand_average(alldata_errorxpside_cued_t); gave_errorpside_cued.data = toverparam(alldata_errorxpside_cued_t)
gave_errorpside_cued.drop_channels(['RM'])
gave_errorpside_cued.plot_joint(timefreqs=timefreqs_cue, topomap_args = dict(outlines='head', contours=0, vmin=-2, vmax=2), vmin=-2, vmax=2)

gave_errorpside_cued.plot(picks = ['PO4', 'PO8', 'O2', 'P8', 'P6', 'P4','P2'], combine='mean', vmin=-2, vmax=2)

fig = plt.figure()
ax = fig.subplots(1,1)
ax.contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_cued).pick_channels(['PO4', 'PO8', 'O2', 'P8', 'P6', 'P4','P2']).data, 0), vmin=-2, vmax=2, levels=100, cmap ='RdBu_r', antialiased=False)
ax.vlines([-1.5, 0, gave_mean_dt], linestyle='--', lw=1, ymin=1, ymax=39)
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time relative to probe onset (s)')

x_errorpside_cued = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_errorpside_cued[i,:,:] = np.nanmean(deepcopy(alldata_errorxpside_cued_t[i]).pick_channels(['PO4', 'PO8', 'O2', 'P8', 'P6', 'P4','P2']).data,0)

t_errorpside_cued_vis, clusters_errorpside_cued_vis, clusters_pv_errorpside_cued_vis, _ = mne.stats.permutation_cluster_1samp_test(x_errorpside_cued, n_permutations='all')
masks_errorpside_cued_vis = np.asarray(clusters_errorpside_cued_vis)[clusters_pv_errorpside_cued_vis <0.05]

fig = plt.figure()
ax = fig.subplots(1,1)
ax.contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_cued).pick_channels(vischans).data, 0), vmin=-2, vmax=2, levels=100, cmap ='RdBu_r', antialiased=False)
ax.vlines([-1.5, 0, gave_mean_dt], linestyle='--', lw=1, ymin=1, ymax=39)
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time relative to probe onset (s)')
for mask in masks_errorpside_cued_vis:
    bigmask = np.kron(mask, np.ones((10,10)))
    ax.contour(bigmask, colors='black', lw=.75, extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), antialiased=False)



#DT and cued side, rel to left hand (right motor channels needed)
gave_dtpside_cued = mne.grand_average(alldata_dtxpside_cued_t); gave_dtpside_cued.data = toverparam(alldata_dtxpside_cued_t)
gave_dtpside_cued.drop_channels(['RM'])
gave_dtpside_cued.plot_joint(timefreqs=timefreqs_cue, topomap_args = dict(outlines='head', contours=0, vmin=-2, vmax=2), vmin=-2, vmax=2)

motleftchans = ['C3', 'C1']

fig = plt.figure()
ax = fig.subplots(1,1)
ax.contourf(times, freqs, np.nanmean(deepcopy(gave_dtpside_cued).pick_channels(motleftchans).data, 0), vmin=-2, vmax=2, levels=100, cmap ='RdBu_r', antialiased=False)
ax.vlines([-1.5, 0, gave_mean_dt], linestyle='--', lw=1, ymin=1, ymax=39)
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time relative to probe onset (s)')

x_errorpside_cued = np.empty(shape = (subs.size, freqs.size, times.size))
for i in range(subs.size):
    x_errorpside_cued[i,:,:] = np.nanmean(deepcopy(alldata_errorxpside_cued_t[i]).pick_channels(['PO4', 'PO8', 'O2', 'P8', 'P6', 'P4','P2']).data,0)

t_errorpside_cued_vis, clusters_errorpside_cued_vis, clusters_pv_errorpside_cued_vis, _ = mne.stats.permutation_cluster_1samp_test(x_errorpside_cued, n_permutations='all')
masks_errorpside_cued_vis = np.asarray(clusters_errorpside_cued_vis)[clusters_pv_errorpside_cued_vis <0.05]

fig = plt.figure()
ax = fig.subplots(1,1)
ax.contourf(times, freqs, np.nanmean(deepcopy(gave_errorpside_cued).pick_channels(vischans).data, 0), vmin=-2, vmax=2, levels=100, cmap ='RdBu_r', antialiased=False)
ax.vlines([-1.5, 0, gave_mean_dt], linestyle='--', lw=1, ymin=1, ymax=39)
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time relative to probe onset (s)')
for mask in masks_errorpside_cued_vis:
    bigmask = np.kron(mask, np.ones((10,10)))
    ax.contour(bigmask, colors='black', lw=.75, extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), antialiased=False)



















#%%
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












