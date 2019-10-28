#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:58:11 2019

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
from wmConfidence_funcs import gesd, plot_AR, toverparam, runclustertest_epochs

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

figpath = op.join(wd, 'figures', 'eeg_figs', 'feedbacklocked', 'epochs_glm4')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm



contrasts = ['grandmean', 'neutral', 'cued', 'cuedvsneutral', 'pside',
             'incorrvscorr', 'error', 'conferror', 'confidence',
             'error_corr', 'error_incorr', 'error_incorrvscorr',
             'conferr_corr', 'conferr_incorr', 'conferror_incorrvscorr',
             'confidence_corr', 'confidence_incorr', 'confidence_incorrvscorr']

data = dict()
data_b = dict()
for i in contrasts:
    data[i] = []
    data_b[i] = []
for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    for name in contrasts:
        data[name].append(   mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_glm4', 'wmc_' + param['subid'] + '_fblocked_tfr_'+ name + '_tstats-tfr.h5'))[0])
        data_b[name].append( mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_glm4', 'wmc_' + param['subid'] + '_fblocked_tfr_'+ name + '_tstats_baselined-tfr.h5'))[0])
#%%#visualise some regressors
#we're not going to plot the grandmean, this shouldn't be in the glm as it doesnt make sense anyways.
#we will visualise the regressors for neutral and cued trials though as these should be different (no lateralisation in in neutral trials)
contrasts = data.keys()

timefreqs = {(.4, 12):(.4, 6),
             (.6, 12):(.4, 6),
             (.8, 12):(.4, 6),
             (.4, 6):(.4, 4),
             (.6, 6):(.4, 4),
             (.8, 6):(.4, 4)}

timefreqs_alpha = {(.4, 12):(.4, 6), #set alpha to 9-15Hz
                   (.6, 12):(.4, 6),
                   (.8, 12):(.4, 6)}
timefreqs_beta  = {(.4, 22):(.4, 14), #set alpha to 9-15Hz
                   (.6, 22):(.4, 14),
                   (.8, 22):(.4, 14)}
timefreqs_theta = {(.4, 6):(.4, 4),
                   (.6, 6):(.4, 4),
                   (.8, 6):(.4, 4)}

visleftchans  = ['PO3', 'PO7', 'O1']
visrightchans = ['PO4','PO8','O2']
motrightchans = ['C2', 'C4']  #channels contra to the left hand (space bar)
motleftchans = ['C1', 'C3']   #channels contra to the right hand (mouse)
frontal_chans = ['CZ', 'FCZ', 'FZ', 'FC1', 'FC2']
ftheta_chans  = ['FPZ','AFZ','FZ','AF3','AF4']

topoargs = dict(outlines= 'head', contours = 0)
topoargs_t = dict(outlines = 'head', contours = 0, vmin=-2, vmax = 2)
#%%
gave_gmean = mne.grand_average(data['grandmean']); gave_gmean.data = toverparam(data['grandmean']); gave_gmean.drop_channels(['RM'])
#gave_gmean.plot_joint(title = 'grandmean, t over tstats', timefreqs = timefreqs_cue,
#                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -5, vmax = 5))

times = gave_gmean.times
allfreqs = gave_gmean.freqs
del(gave_gmean)

#%%

for cope in contrasts:
    for i in range(subs.size):
        data[cope][i].drop_channels(['RM'])
        data_b[cope][i].drop_channels(['RM'])

#%%
        
#is there anything lateralised in the visual response, relative to the location of the probed item?

gave_pside = mne.grand_average(data['pside']); gave_pside.data = toverparam(data['pside'])
gave_pside.plot_joint(title = 'probed item left vs right', timefreqs = timefreqs, topomap_args = topoargs_t, picks = 'eeg')



#%%
#firstly lets just visualise the ERN if we can: this is incorr vs corr
#this is a contrast so we can take the non-baselined data

gave_ern = mne.grand_average(data['incorrvscorr']); gave_ern.data = toverparam(data['incorrvscorr']); #gave_ern.drop_channels(['RM'])

gave_ern.plot_joint(title = 'incorrect vs correct', picks = 'eeg',
                    topomap_args = topoargs_t, timefreqs = timefreqs)

#lets plot just FCz now, as we know thats where we get the ERN effect

fcz_ern = deepcopy(gave_ern).pick_channels(['FCZ']).data[0]
fcz_ern = deepcopy(gave_ern).pick_channels(frontal_chans).data[0]



fig = plt.figure()
ax = fig.add_subplot(111)
tfplot = ax.imshow(fcz_ern, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (times.min(), times.max(), allfreqs.min(), allfreqs.max()))
ax.set_xlabel('Time rel. to feedback onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0], linestyles = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
ax.hlines([4,8], linestyles = 'dashed', lw = 2, color = '#000000', xmin = times.min(), xmax = times.max())
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)

#%%
gave_errorcorr = mne.grand_average(data_b['error_corr']); gave_errorcorr.data = toverparam(data_b['error_corr']); #gave_errorcorr.drop_channels(['RM'])

gave_errorcorr.plot_joint(title = 'error ~ correct trials', picks = 'eeg',
                    topomap_args = topoargs_t, timefreqs = timefreqs)

fcz_errcorr = deepcopy(gave_errorcorr).pick_channels(['FCZ']).data[0]
fcz_errcorr = deepcopy(gave_errorcorr).pick_channels(frontal_chans).data[0]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('frontal channels: Error ~ correct trials')
tfplot = ax.imshow(fcz_errcorr, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (times.min(), times.max(), allfreqs.min(), allfreqs.max()))
ax.set_xlabel('Time rel. to feedback onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0], linestyles = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)

gave_errorincorr = mne.grand_average(data_b['error_incorr']); gave_errorincorr.data = toverparam(data_b['error_incorr']); #gave_errorcorr.drop_channels(['RM'])

gave_errorincorr.plot_joint(title = 'error ~ incorrect trials', picks = 'eeg',
                    topomap_args = topoargs_t, timefreqs = timefreqs)

fcz_errincorr = deepcopy(gave_errorincorr).pick_channels(['FCZ']).data[0]
fcz_errincorr = deepcopy(gave_errorincorr).pick_channels(frontal_chans).data[0]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('frontal channels: Error ~ incorrect trials')
tfplot = ax.imshow(fcz_errincorr, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (times.min(), times.max(), allfreqs.min(), allfreqs.max()))
ax.set_xlabel('Time rel. to feedback onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0], linestyles = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)


gave_errorern = mne.grand_average(data_b['error_incorrvscorr']); gave_errorern.data = toverparam(data_b['error_incorrvscorr'])
gave_errorern.plot_joint(title = 'error ~ incorrect vs correct trials', picks = 'eeg',
                         topomap_args = topoargs_t, timefreqs = timefreqs)

fcz_errorern = deepcopy(gave_errorern).pick_channels(['FCZ']).data[0]
fcz_errorern = deepcopy(gave_errorern).pick_channels(frontal_chans).data[0]

fig = plt.figure()
ax = fig.add_subplot(111)
tfplot = ax.imshow(fcz_errorern, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (times.min(), times.max(), allfreqs.min(), allfreqs.max()))
ax.set_xlabel('Time rel. to feedback onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0], linestyles = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)


gave_error = mne.grand_average(data_b['error']); gave_error.data = toverparam(data_b['error'])
gave_error.plot_joint(title = 'main effect error', picks = 'eeg', timefreqs=timefreqs, topomap_args = topoargs_t)

fcz_error = deepcopy(gave_error).pick_channels(['FCZ']).data[0]
fcz_error = deepcopy(gave_error).pick_channels(frontal_chans).data[0]

fig = plt.figure()
ax = fig.add_subplot(111)
tfplot = ax.imshow(fcz_error, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (times.min(), times.max(), allfreqs.min(), allfreqs.max()))
ax.set_xlabel('Time rel. to feedback onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0], linestyles = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)

#%%

#lets quickly just look at cued and neutral trials, and the contrast

gave_cued = mne.grand_average(data_b['cued']); gave_cued.data = toverparam(data_b['cued'])
gave_cued.plot_joint(title = 'cued trials', picks = 'eeg', timefreqs = timefreqs, topomap_args = topoargs_t)

gave_neut = mne.grand_average(data_b['neutral']); gave_neut.data = toverparam(data_b['neutral'])
gave_neut.plot_joint(title = 'neutral trials', picks = 'eeg', timefreqs = timefreqs, topomap_args = topoargs_t)

gave_cvsn = mne.grand_average(data['cuedvsneutral']); gave_cvsn.data = toverparam(data['cuedvsneutral'])
gave_cvsn.plot_joint(title = 'cued vs neutral trials', picks = 'eeg', fmin=None, fmax=None, timefreqs = timefreqs_theta, topomap_args = topoargs_t)

#%%
gave_confcorr = mne.grand_average(data_b['confidence_corr']); gave_confcorr.data = toverparam(data_b['confidence_corr']); #gave_errorcorr.drop_channels(['RM'])

gave_confcorr.plot_joint(title = 'confidence ~ correct trials', picks = 'eeg',
                    topomap_args = topoargs_t, timefreqs = timefreqs)

fcz_confcorr = deepcopy(gave_confcorr).pick_channels(['FCZ']).data[0]
fcz_confcorr = deepcopy(gave_confcorr).pick_channels(frontal_chans).data[0]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('frontal channels: Error ~ correct trials')
tfplot = ax.imshow(fcz_confcorr, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (times.min(), times.max(), allfreqs.min(), allfreqs.max()))
ax.set_xlabel('Time rel. to feedback onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0], linestyles = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)

gave_confincorr = mne.grand_average(data_b['confidence_incorr']); gave_confincorr.data = toverparam(data_b['confidence_incorr']); #gave_errorcorr.drop_channels(['RM'])

gave_confincorr.plot_joint(title = 'confidence ~ incorrect trials', picks = 'eeg',
                    topomap_args = topoargs_t, timefreqs = timefreqs)

fcz_confincorr = deepcopy(gave_confincorr).pick_channels(['FCZ']).data[0]
fcz_confincorr = deepcopy(gave_confincorr).pick_channels(frontal_chans).data[0]


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('frontal channels: confidence ~ incorrect trials')
tfplot = ax.imshow(fcz_confincorr, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (times.min(), times.max(), allfreqs.min(), allfreqs.max()))
ax.set_xlabel('Time rel. to feedback onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0], linestyles = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)


gave_confern = mne.grand_average(data_b['confidence_incorrvscorr']); gave_confern.data = toverparam(data_b['confidence_incorrvscorr'])
gave_confern.plot_joint(title = 'confidence ~ incorrect vs correct trials', picks = 'eeg',
                         topomap_args = topoargs_t, timefreqs = timefreqs)

fcz_confern = deepcopy(gave_confern).pick_channels(['FCZ']).data[0]
fcz_confern = deepcopy(gave_confern).pick_channels(frontal_chans).data[0]

fig = plt.figure()
ax = fig.add_subplot(111)
tfplot = ax.imshow(fcz_confern, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (times.min(), times.max(), allfreqs.min(), allfreqs.max()))
ax.set_xlabel('Time rel. to feedback onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0], linestyles = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)


gave_error = mne.grand_average(data_b['confidence']); gave_error.data = toverparam(data_b['confidence'])
gave_error.plot_joint(title = 'main effect confidence', picks = 'eeg', timefreqs=timefreqs, topomap_args = topoargs_t)

fcz_error = deepcopy(gave_error).pick_channels(['FCZ']).data[0]
fcz_error = deepcopy(gave_error).pick_channels(frontal_chans).data[0]

fig = plt.figure()
ax = fig.add_subplot(111)
tfplot = ax.imshow(fcz_error, cmap = 'RdBu_r', aspect = 'auto', vmin=-2, vmax=2, interpolation = 'gaussian',
          origin = 'lower', extent = (times.min(), times.max(), allfreqs.min(), allfreqs.max()))
ax.set_xlabel('Time rel. to feedback onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0], linestyles = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = fig.add_axes([.95, .15, .02, .35])
fig.colorbar(tfplot, cax = cbaxes)

