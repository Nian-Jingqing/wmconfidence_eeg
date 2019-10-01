#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 18:16:53 2019

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

contrasts = ['grandmean',
             'pleft_neutral','pleft_cued','pright_neutral','pright_cued',
             'plvsr_cued',
             'error', 'confidence', 'DT']
        

data = dict()
data_baselined = dict()
data_t = dict()
data_baselined_t = dict()

for i in contrasts:
    data[i] = []
    data_baselined[i] = []
    data_t[i] = []
    data_baselined_t[i] = []

model = 'simple'# 'simple_nogmean', 'simple_nogmean_withmaineffects', 'simple_withmaineffects'
model = 'simple_nogmean'
model = 'simple_withmaineffects'
#model = 'simple_nogmean_withmaineffects'
for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    for name in contrasts:
        data[name].append(mne.time_frequency.read_tfrs(            fname = op.join(param['path'], 'glms', 'probe', model, 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_betas-tfr.h5'))[0])
        data_t[name].append(mne.time_frequency.read_tfrs(          fname = op.join(param['path'], 'glms', 'probe', model, 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_tstats-tfr.h5'))[0])
        data_baselined[name].append(mne.time_frequency.read_tfrs(  fname = op.join(param['path'], 'glms', 'probe', model, 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_betas_baselined-tfr.h5'))[0])
        data_baselined_t[name].append(mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'probe', model, 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_tstats_baselined-tfr.h5'))[0])
#%%
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

visleftchans  = ['PO3', 'PO7', 'O1']

visrightchans = ['PO4','PO8','O2']

motrightchans = ['C2', 'C4']  #channels contra to the left hand (space bar)
motleftchans = ['C1', 'C3']   #channels contra to the right hand (mouse)

topoargs = dict(outlines= 'head', contours = 0)
topoargs_t = dict(outlines = 'head', contours = 0, vmin=-2, vmax = 2)
#%%

gave_cleft = mne.grand_average(data_baselined_t['pleft_cued'])
gave_cleft.data = toverparam(data_baselined_t['pleft_cued']); gave_cleft.drop_channels(['RM'])
gave_cleft.plot_joint(topomap_args = topoargs_t, timefreqs = timefreqs_cue, title = 'cued left '+ model+ ' t over tstats')

cleft_vrchans = np.subtract(np.nanmean(deepcopy(gave_cleft).pick_channels(visrightchans).data,0),0) #np.nanmean(deepcopy(gave_cleft).pick_channels(visleftchans).data,0))

times, freqs = deepcopy(gave_cleft).crop(tmin=-2, tmax=0).times, gave_cleft.freqs

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.pcolormesh(gave_dtlvsr.times, gave_dtlvsr.freqs, cvsi_dtlvsr, cmap = 'RdBu_r', antialiased = False, vmin = -2, vmax = 2)
ax.imshow(cleft_vrchans, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'none', origin = 'lower', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)))
ax.vlines([0, -1.5], linestyle = 'dashed', lw = .75, ymin=1, ymax = 39)
ax.set_ylim(1,39)
ax.set_title('contra-ipsi to cued left')


gave_cright = mne.grand_average(data_baselined_t['pright_cued'])
gave_cright.data = toverparam(data_baselined_t['pright_cued']); gave_cright.drop_channels(['RM'])
gave_cright.plot_joint(topomap_args = topoargs_t, timefreqs = timefreqs_cue, title = 'cued right ' + model +' t over tstats')

cright_vrchans = np.subtract(np.nanmean(deepcopy(gave_cright).pick_channels(visleftchans).data,0), 0)#np.nanmean(deepcopy(gave_cright).pick_channels(visrightchans).data,0))

times, freqs = deepcopy(gave_cright).crop(tmin=-2, tmax=0).times, gave_cright.freqs

fig = plt.figure()
ax = fig.add_subplot(111)
#ax.pcolormesh(gave_dtlvsr.times, gave_dtlvsr.freqs, cvsi_dtlvsr, cmap = 'RdBu_r', antialiased = False, vmin = -2, vmax = 2)
ax.imshow(cright_vrchans, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'none', origin = 'lower', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)))
ax.vlines([0, -1.5], linestyle = 'dashed', lw = .75, ymin=1, ymax = 39)
ax.set_ylim(1,39)
ax.set_title('contra-ipsi to cued right')
#%%
        
gave_lvsr = mne.grand_average(data_t['plvsr_cued']); gave_lvsr.data = toverparam(data_t['plvsr_cued']); gave_lvsr.drop_channels(['RM'])
gave_lvsr.plot_joint(topomap_args = topoargs_t, timefreqs = timefreqs_cue, title = 'cued lvsr ' + model + ' t over tstats')

cvsi_clvsr = np.subtract(np.nanmean(deepcopy(gave_lvsr).crop(tmin=-2,tmax=0).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(gave_lvsr).crop(tmin=-2,tmax=0).pick_channels(visleftchans).data, 0))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(deepcopy(gave_lvsr).crop(tmin=-2,tmax=0).times, deepcopy(gave_lvsr).crop(tmin=-2,tmax=0).freqs, cvsi_clvsr, levels = 100, cmap = 'RdBu_r', antialiased = False, vmin = -2, vmax = 2)
ax.vlines([0, -1.5], linestyle = 'dashed', lw = .75, ymin=1, ymax = 39)

#%%

gave_gmean = mne.grand_average(data_baselined_t['grandmean']); gave_gmean.data = toverparam(data_baselined_t['grandmean'])
gave_gmean.drop_channels(['RM'])
gave_gmean.plot_joint(timefreqs = timefreqs_cue, topomap_args = dict(outlines='head', contours=0, vmin=-3, vmax=3), vmin=-3, vmax=3 )


#%%

gave_error = mne.grand_average(data_baselined_t['error']); gave_error.data = toverparam(data_baselined_t['error'])
gave_error.drop_channels(['RM'])
gave_error.plot_joint(topomap_args = topoargs_t, timefreqs = timefreqs_cue, title = 'error ' +model+' t over tstats')

gave_dt   = mne.grand_average(data_baselined_t['DT']); gave_dt.data =toverparam(data_baselined_t['DT']);
gave_dt.drop_channels(['RM'])
gave_dt.plot_joint(topomap_args = topoargs_t, timefreqs = timefreqs_cue, title = 'DT '+model+' t over tstats')

gave_conf = mne.grand_average(data_baselined_t['confidence']); gave_conf.data = toverparam(data_baselined_t['confidence'])
gave_conf.drop_channels(['RM'])
gave_conf.plot_joint(topomap_args = topoargs_t, timefreqs = timefreqs_cue, title = 'confidence '+model+' t over tstats')




