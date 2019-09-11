#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:13:04 2019

@author: sammirc
"""
import numpy as np
import scipy as sp
import pandas as pd
import mne
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri/analysis_scripts')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import toverparam

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/glm')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

subs = np.array([1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
subs = np.array([1,    4, 5, 6, 7, 8, 9, 10,     12, 13, 14]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm


alldata_grandmean                = []
alldata_DT                       = []
alldata_error                    = []
alldata_conf                     = []
alldata_prevtrlconfdiff          = []
alldata_prevtrlunderconfconfdiff = []
alldata_prevtrloverconfconfdiff  = []

alldata_grandmean_t                = []
alldata_DT_t                       = []
alldata_error_t                    = []
alldata_conf_t                     = []
alldata_prevtrlconfdiff_t          = []
alldata_prevtrlunderconfconfdiff_t = []
alldata_prevtrloverconfconfdiff_t  = []




for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    #there are no contrasts here, so all we are reading in are baselined
    alldata_grandmean.append(mne.time_frequency.read_tfrs(fname                     = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_grandmean_betas_baselined-tfr.h5'))[0])
    alldata_DT.append(mne.time_frequency.read_tfrs(fname                            = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_DT_betas_baselined-tfr.h5'))[0])
    alldata_error.append(mne.time_frequency.read_tfrs(fname                         = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_error_betas_baselined-tfr.h5'))[0])
    alldata_conf.append(mne.time_frequency.read_tfrs(fname                          = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_confidence_betas_baselined-tfr.h5'))[0])
    #alldata_prevtrlconfdiff.append(mne.time_frequency.read_tfrs(fname               = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_prevtrialconfdiff_betas_baselined-tfr.h5'))[0])
    alldata_prevtrlunderconfconfdiff.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_prevtrialconfdiffunderconf_betas_baselined-tfr.h5'))[0])
    alldata_prevtrloverconfconfdiff.append(mne.time_frequency.read_tfrs(fname       = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_prevtrialconfdiffoverconf_betas_baselined-tfr.h5'))[0])
    
    alldata_grandmean_t.append(mne.time_frequency.read_tfrs(fname                   = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_grandmean_tstats_baselined-tfr.h5'))[0])
    alldata_DT_t.append(mne.time_frequency.read_tfrs(fname                          = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_DT_tstats_baselined-tfr.h5'))[0])
    alldata_error_t.append(mne.time_frequency.read_tfrs(fname                       = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_error_tstats_baselined-tfr.h5'))[0])
    alldata_conf_t.append(mne.time_frequency.read_tfrs(fname                        = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_confidence_tstats_baselined-tfr.h5'))[0])
    #alldata_prevtrlconfdiff_t.append(mne.time_frequency.read_tfrs(fname             = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_prevtrialconfdiff_tstats_baselined-tfr.h5'))[0])
    alldata_prevtrlunderconfconfdiff_t.append(mne.time_frequency.read_tfrs(fname    = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_prevtrialconfdiffunderconf_tstats_baselined-tfr.h5'))[0])
    alldata_prevtrloverconfconfdiff_t.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_prevtrialconfdiffoverconf_tstats_baselined-tfr.h5'))[0])
    
   


#%%
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#visualise some regressors
#we're not going to plot the grandmean, this shouldn't be in the glm as it doesnt make sense anyways.
#we will visualise the regressors for neutral and cued trials though as these should be different (no lateralisation in in neutral trials)
timefreqs = {(-.2, 10):(.2, 4),
             (.4, 10) :(.4, 4),
             (.6, 10) :(.4, 4),
             (.8, 10) :(.4, 4)}

gave_gmean = mne.grand_average(alldata_grandmean); gave_gmean.drop_channels(['RM'])
gave_gmean.plot_joint(title = 'all trials response to array, average betas, preglm baseline', timefreqs = timefreqs,
                      topomap_args = dict(outlines = 'head', contours = 0))
plot_singlesubs = False
from copy import deepcopy
if plot_singlesubs:
    for i in range(len(alldata_grandmean)):
        tmp = deepcopy(alldata_grandmean[i])
        tmp = tmp.drop_channels(['RM'])
        tmp.plot_joint(title = 'ave response, betas, subject %d'%subs[i], timefreqs = timefreqs,
                       topomap_args = dict(outlines = 'head', contours = 0))
        
        


gave_gmean = mne.grand_average(alldata_grandmean); gave_gmean.data = toverparam(alldata_grandmean); gave_gmean.drop_channels(['RM'])
gave_gmean.plot_joint(title = 'all trials response to array, t over betas, preglm baseline', timefreqs = timefreqs,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -4, vmax = +4))

gave_gmean_t = mne.grand_average(alldata_grandmean_t); gave_gmean_t.data = toverparam(alldata_grandmean_t); gave_gmean_t.drop_channels(['RM'])
gave_gmean_t.plot_joint(title = 'all trials response to array, t over tstats, preglm baseline', timefreqs = timefreqs,
                        topomap_args = dict(outlines = 'head', contours = 0, vmin = -4, vmax = 4))


gave_error_t = mne.grand_average(alldata_error_t); gave_error_t.data = toverparam(alldata_error_t); gave_error_t.drop_channels(['RM'])
gave_error_t.plot_joint(title = 'main effect of error, t over tstats, preglm baseline', timefreqs = timefreqs,
                        topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_conf_t = mne.grand_average(alldata_conf_t); gave_conf_t.data = toverparam(alldata_conf_t); gave_conf_t.drop_channels(['RM'])
gave_conf_t.plot_joint(title = 'main effect of confidence, t over tstats, preglm baseline', timefreqs = timefreqs,
                       topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_dt_t = mne.grand_average(alldata_DT_t); gave_dt_t.data = toverparam(alldata_DT_t); gave_dt_t.drop_channels(['RM'])
gave_dt_t.plot_joint(title = 'main effect of DT, t over tstats, preglm baseline', timefreqs = timefreqs,
                       topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_prevtrlunder_t = mne.grand_average(alldata_prevtrlunderconfconfdiff_t); gave_prevtrlunder_t.data = toverparam(alldata_prevtrlunderconfconfdiff_t)
gave_prevtrlunder_t.drop_channels(['RM'])
gave_prevtrlunder_t.plot_joint(title = 'effect of previous trial error awareness when underconfident, t over tstats', timefreqs = timefreqs,
                               topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_prevtrlover_t = mne.grand_average(alldata_prevtrloverconfconfdiff_t); gave_prevtrlover_t.data = toverparam(alldata_prevtrloverconfconfdiff_t)
gave_prevtrlover_t.drop_channels(['RM'])
gave_prevtrlover_t.plot_joint(title = 'effect of previous trial error awareness when overconfident, t over tstats', timefreqs = timefreqs,
                              topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))


