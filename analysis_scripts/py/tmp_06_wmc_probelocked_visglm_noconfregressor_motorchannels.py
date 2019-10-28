#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:28:55 2019

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
from scipy import ndimage


sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

figpath = op.join(wd, 'figures', 'eeg_figs', 'probelocked', 'tfr_glm2_results')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm



contrasts = ['grandmean',
             'pleft_neutral','pleft_cued','pright_neutral','pright_cued',
             'dt_pleft_neutral','dt_pleft_cued','dt_pright_neutral','dt_pright_cued',
             'error_pleft_neutral','error_pleft_cued','error_pright_neutral','error_pright_cued',
             'conf_pleft_neutral','conf_pleft_cued','conf_pright_neutral','conf_pright_cued',
             'pleft_cvsn','pright_cvsn',
             'neutral','cued',
             'dt_pleft_cvsn','dt_pright_cvsn','dt_neutral','dt_cued','dt_cued_lvsr',
             'error_pleft_cvsn','error_pright_cvsn','error_neutral','error_cued','error_cued_lvsr',
             'conf_pleft_cvsn','conf_pright_cvsn','conf_neutral','conf_cued','conf_cued_lvsr',
             'plvsr_cvsn','plvsr_cued',
             'cuedvsneutral', 'dt_cuedvsneutral', 'error_cuedvsneutral', 'conf_cuedvsneutral']
data = dict()
data_baselined = dict()
data_t = dict()
data_baselined_t = dict()
for i in contrasts:
    data[i] = []
    data_baselined[i] = []
    data_t[i] = []
    data_baselined_t[i] = []
for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    for name in contrasts:
        #data[name].append(mne.time_frequency.read_tfrs(            fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm_2_conf', 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_betas-tfr.h5'))[0])
        data_t[name].append(mne.time_frequency.read_tfrs(          fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm_2_conf', 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_tstats-tfr.h5'))[0])
        #data_baselined[name].append(mne.time_frequency.read_tfrs(  fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm_2_conf', 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_betas_baselined-tfr.h5'))[0])
        data_baselined_t[name].append(mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm_2_conf', 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_tstats_baselined-tfr.h5'))[0])
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
                  (.8, 10):(.4, 4)}
timefreqs_cue = {(-1.2, 10):(.4, 4),
                 (-1.0, 10):(.4, 4),
                 (-0.8, 10):(.4, 4),
                 (-0.6, 10):(.4, 4)}


timefreqs_cue_rel2cue = {
        (.3, 10):(.4, 4),
        (.5, 10):(.4, 4),
        (.7, 10):(.5, 4),
        (.9, 10):(.4, 4)}

visleftchans  = ['PO3', 'PO7', 'O1']

visrightchans = ['PO4','PO8','O2']

motrightchans = ['C2', 'C4']  #channels contra to the left hand (space bar)
motleftchans = ['C1', 'C3']   #channels contra to the right hand (mouse)

topoargs = dict(outlines= 'head', contours = 0)
topoargs_t = dict(outlines = 'head', contours = 0, vmin=-2, vmax = 2)



#%%
gave_gmean = mne.grand_average(data_baselined_t['grandmean']); gave_gmean.data = toverparam(data_baselined_t['grandmean']); gave_gmean.drop_channels(['RM'])
#gave_gmean.plot_joint(title = 'grandmean, t over tstats', timefreqs = timefreqs_cue,
#                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -5, vmax = 5))

times = gave_gmean.times
timesrel2cue = np.add(times, 1.5) #this sets zero to be the cue onset time
allfreqs = gave_gmean.freqs

#%%

for i in data_baselined_t['cued']:
    i.times = timesrel2cue
gave_cued = mne.grand_average(data_baselined_t['cued']); gave_cued.data = toverparam(data_baselined_t['cued']); gave_cued.drop_channels(['RM'])


tmp = deepcopy(gave_cued).pick_channels(['C2', 'C4', 'C1', 'C3'])

cued_lefthand  = deepcopy(gave_cued).pick_channels(motrightchans)
cued_righthand = deepcopy(gave_cued).pick_channels(motleftchans)

cued_lh = np.nanmean(deepcopy(cued_lefthand).data, 0) #average across these channels but preserve frequencies

fig = plt.figure()
fig.suptitle('tfr for left hand motor channels (C2, C4)')
ax = fig.add_subplot(111)
tfplot = ax.imshow(cued_lh, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
ax.set_xlabel('Time rel. to cue onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0, 1.5], linestyle = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)


cued_rh = np.nanmean(deepcopy(cued_righthand).data, 0) #average across these channels but preserve frequencies
fig = plt.figure()
fig.suptitle('tfr for right hand motor channels (C1, C3)')
ax = fig.add_subplot(111)
tfplot = ax.imshow(cued_rh, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
ax.set_xlabel('Time rel. to cue onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0, 1.5], linestyle = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)

#%% just for interest in cued left vs neutral (because i have this!) is there a beta thing? if so, might be worth re-running the glm to get the cued vs neutral contrast out

for i in data_t['pleft_cvsn']:
    i.times = timesrel2cue
gave_clvsn = mne.grand_average(data_t['pleft_cvsn']); gave_clvsn.data = toverparam(data_t['pleft_cvsn']); gave_clvsn.drop_channels(['RM'])

clvsn_lefthand = deepcopy(gave_clvsn).pick_channels(motrightchans)
clvsn_righthand = deepcopy(gave_clvsn).pick_channels(motleftchans)

clvsn_lh = np.nanmean(deepcopy(clvsn_lefthand).data , 0)
clvsn_rh = np.nanmean(deepcopy(clvsn_righthand).data, 0)

#plot left hand for cued left vs neutral
fig = plt.figure()
fig.suptitle('tfr for left hand motor channels (C2, C4) - cued left vs neutral')
ax = fig.add_subplot(111)
tfplot = ax.imshow(clvsn_lh, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
ax.set_xlabel('Time rel. to cue onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0, 1.5], linestyle = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)

#plot left hand for cued left vs neutral
fig = plt.figure()
fig.suptitle('tfr for right hand motor channels (C1, C3) - cued left vs neutral')
ax = fig.add_subplot(111)
tfplot = ax.imshow(clvsn_rh, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
ax.set_xlabel('Time rel. to cue onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0, 1.5], linestyle = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)

#%%

#is error reflected in something related to the motor channels? either hand? can we separate contributions to effector selection and action preparation

#maybe something for decision time in cued vs neutral but its more alpha than beta really (just in motor channels thats all)
#beta maybe something ~ 20hz for confidence in cued vs neutral but nothing really stands out like it would be meaningful in either set of electrodes
for i in data_t['dt_cued']:
    i.times = timesrel2cue
gave_err_cued = mne.grand_average(data_t['dt_cued']); gave_err_cued.data = toverparam(data_t['dt_cued']); gave_err_cued.drop_channels(['RM'])


tmp = deepcopy(gave_err_cued).pick_channels(['C2', 'C4', 'C1', 'C3'])

errcued_lefthand  = deepcopy(gave_err_cued).pick_channels(motrightchans)
errcued_righthand = deepcopy(gave_err_cued).pick_channels(motleftchans)

errcued_lh = np.nanmean(deepcopy(errcued_lefthand).data , 0)
errcued_rh = np.nanmean(deepcopy(errcued_righthand).data, 0)


#plot left hand for cued left vs neutral
fig = plt.figure()
fig.suptitle('tfr for left hand motor channels (C2, C4)')
ax = fig.add_subplot(111)
tfplot = ax.imshow(errcued_lh, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
ax.set_xlabel('Time rel. to cue onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0, 1.5], linestyle = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)

#plot left hand for cued left vs neutral
fig = plt.figure()
fig.suptitle('tfr for right hand motor channels (C1, C3)')
ax = fig.add_subplot(111)
tfplot = ax.imshow(errcued_rh, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
ax.set_xlabel('Time rel. to cue onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0, 1.5], linestyle = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)

# see if there's anything lateralised in motor maybe?
lvsr_motor = np.empty(shape = (subs.size, allfreqs.size, timesrel2cue.size))
for i in range(subs.size):
    data_t['dt_cued'][i].times = timesrel2cue
    tmp = deepcopy(data_t['dt_cued'][i]).pick_channels(motleftchans+motrightchans)
    tmp_lvsr = np.subtract(np.nanmean(deepcopy(tmp).pick_channels(motleftchans).data,0),
                           np.nanmean(deepcopy(tmp).pick_channels(motrightchans).data,0))
    lvsr_motor[i,:,:] = tmp_lvsr
    
lvsr_mot = sp.stats.ttest_1samp(lvsr_motor, popmean = 0, axis = 0)[0]

fig = plt.figure()
fig.suptitle('tfr for right hand - left hand motor channels')
ax = fig.add_subplot(111)
tfplot = ax.imshow(lvsr_mot, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
ax.set_xlabel('Time rel. to cue onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0, 1.5], linestyle = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)















