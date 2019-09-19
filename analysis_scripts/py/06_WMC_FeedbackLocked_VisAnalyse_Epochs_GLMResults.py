#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:56:51 2019

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

alldata_grandmean       = []
alldata_neutral         = []
alldata_cued            = []
alldata_error           = []
alldata_conf            = []
alldata_targinconf      = []
alldata_targoutsideconf = []
alldata_underconf       = []
alldata_overconf        = []
alldata_cvsn            = []
alldata_badvsgood       = []

alldata_grandmean_t       = []
alldata_neutral_t         = []
alldata_cued_t            = []
alldata_error_t           = []
alldata_conf_t            = []
alldata_targinconf_t      = []
alldata_targoutsideconf_t = []
alldata_underconf_t       = []
alldata_overconf_t        = []
alldata_cvsn_t            = []
alldata_badvsgood_t       = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    alldata_grandmean.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_grandmean_betas-ave.fif'))[0])
    alldata_neutral.append(mne.read_evokeds(fname         = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_neutral_betas-ave.fif'))[0])
    alldata_cued.append(mne.read_evokeds(fname            = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_cued_betas-ave.fif'))[0])
    alldata_error.append(mne.read_evokeds(fname           = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_error_betas-ave.fif'))[0])
    alldata_conf.append(mne.read_evokeds(fname            = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confidence_betas-ave.fif'))[0])
    alldata_targinconf.append(mne.read_evokeds(fname      = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_targinconf_betas-ave.fif'))[0])
    alldata_targoutsideconf.append(mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_targoutsideconf_betas-ave.fif'))[0])
    alldata_underconf.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confdiffunderconf_betas-ave.fif'))[0])
    alldata_overconf.append(mne.read_evokeds(fname        = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confdiffoverconf_betas-ave.fif'))[0])
    alldata_cvsn.append(mne.read_evokeds(fname            = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_cuedvsneutral_betas-ave.fif'))[0])
    alldata_badvsgood.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_redvsgreen_betas-ave.fif'))[0])
    

    alldata_grandmean_t.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_grandmean_tstats-ave.fif'))[0])
    alldata_neutral_t.append(mne.read_evokeds(fname         = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_neutral_tstats-ave.fif'))[0])
    alldata_cued_t.append(mne.read_evokeds(fname            = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_cued_tstats-ave.fif'))[0])
    alldata_error_t.append(mne.read_evokeds(fname           = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_error_tstats-ave.fif'))[0])
    alldata_conf_t.append(mne.read_evokeds(fname            = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confidence_tstats-ave.fif'))[0])
    alldata_targinconf_t.append(mne.read_evokeds(fname      = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_targinconf_tstats-ave.fif'))[0])
    alldata_targoutsideconf_t.append(mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_targoutsideconf_tstats-ave.fif'))[0])
    alldata_underconf_t.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confdiffunderconf_tstats-ave.fif'))[0])
    alldata_overconf_t.append(mne.read_evokeds(fname        = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_confdiffoverconf_tstats-ave.fif'))[0])
    alldata_cvsn_t.append(mne.read_evokeds(fname            = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_cuedvsneutral_tstats-ave.fif'))[0])
    alldata_badvsgood_t.append(mne.read_evokeds(fname       = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_redvsgreen_tstats-ave.fif'))[0])
    
#%%
# Question 1 - is there an error related signal in the scalp voltage, coarsely on whether there was green or red feedback?
    
    
gave_green = mne.grand_average(alldata_targinconf); gave_green.drop_channels(['RM'])
gave_green.plot_joint(picks = 'eeg')
#gave_green.plot(picks = ['FCZ'])

gave_red = mne.grand_average(alldata_targoutsideconf); gave_red.drop_channels(['RM'])
gave_red.plot_joint(picks = 'eeg')
#gave_red.plot(picks = ['FCZ'])


gave_redvsgreen = mne.grand_average(alldata_badvsgood); gave_redvsgreen.drop_channels(['RM'])
gave_redvsgreen.plot_joint(picks = 'eeg')


gave_underconf = mne.grand_average(alldata_underconf); gave_underconf.drop_channels(['RM'])
gave_underconf.plot_joint(picks = 'eeg')
gave_underconf.plot(picks = 'FCZ')

gave_overconf = mne.grand_average(alldata_overconf); gave_overconf.drop_channels(['RM'])
gave_


gave_error = mne.grand_average(alldata_error); gave_error.drop_channels(['RM'])
gave_error.plot_joint(picks = 'eeg')
gave_error.plot(picks = 'FCZ')



mne.viz.plot_compare_evokeds(evokeds = dict(error = gave_red,
                                            correct = gave_green,
                                            diff = gave_redvsgreen,
                                            erroreffect = gave_error),
                             invert_y = False,
                             picks = 'FCZ',
                             colors = ['green', 'blue', 'red', 'purple'])























