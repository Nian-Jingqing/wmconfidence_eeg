#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 11:52:50 2019

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
             'dt_pleft_neutral','dt_pleft_cued','dt_pright_neutral','dt_pright_cued',
             'error_pleft_neutral','error_pleft_cued','error_pright_neutral','error_pright_cued',
             'conf_pleft_neutral','conf_pleft_cued','conf_pright_neutral','conf_pright_cued',
             'pleft_cvsn','pright_cvsn',
             'neutral','cued',
             'dt_pleft_cvsn','dt_pright_cvsn','dt_neutral','dt_cued','dt_cued_lvsr',
             'error_pleft_cvsn','error_pright_cvsn','error_neutral','error_cued','error_cued_lvsr',
             'conf_pleft_cvsn','conf_pright_cvsn','conf_neutral','conf_cued','conf_cued_lvsr',
             'plvsr_cvsn','plvsr_cued']
        




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
        data[name].append(mne.time_frequency.read_tfrs(            fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm_2_conf', 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_betas-tfr.h5'))[0])
        data_t[name].append(mne.time_frequency.read_tfrs(          fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm_2_conf', 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_tstats-tfr.h5'))[0])
        data_baselined[name].append(mne.time_frequency.read_tfrs(  fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm_2_conf', 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_betas_baselined-tfr.h5'))[0])
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
                  (.8, 10):(.4, 4),
                  (1., 10):(.4, 4)}
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

gave_cuedlvsr = mne.grand_average(data_t['plvsr_cued']); gave_cuedlvsr.data = toverparam(data_t['plvsr_cued']); gave_cuedlvsr.drop_channels(['RM'])

gave_cuedlvsr.plot_joint(title = 'cued left vs right, t over tstats',
                         timefreqs = timefreqs_cue, topomap_args = topoargs_t)


cvsi_clvsr = np.subtract(np.nanmean(deepcopy(gave_cuedlvsr).crop(tmin=-2,tmax=0).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(gave_cuedlvsr).crop(tmin=-2,tmax=0).pick_channels(visleftchans).data, 0))


fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(deepcopy(gave_cuedlvsr).crop(tmin=-2,tmax=0).times, deepcopy(gave_cuedlvsr).crop(tmin=-2,tmax=0).freqs, cvsi_clvsr, levels = 100, cmap = 'RdBu_r', antialiased = False, vmin = -2, vmax = 2)
ax.vlines([0, -1.5], linestyle = 'dashed', lw = .75, ymin=1, ymax = 39)


x_clvsr = np.empty(shape = (subs.size, deepcopy(gave_cuedlvsr).crop(tmin=-2,tmax=0).freqs.size, deepcopy(gave_cuedlvsr).crop(tmin=-2,tmax=0).times.size))
for i in range(subs.size):
    iclvsr = np.subtract(np.nanmean(deepcopy(data_t['plvsr_cued'][i]).crop(tmin=-2,tmax=0).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(data_t['plvsr_cued'][i]).crop(tmin=-2,tmax=0).pick_channels(visleftchans).data,0))
    x_clvsr[i,:,:] = iclvsr

t_lvsr, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(x_clvsr, n_permutations='all')
masks = np.asarray(clusters)[cluster_pv<0.15]


times, freqs =deepcopy(gave_cuedlvsr).crop(tmin=-2,tmax=0).times,  deepcopy(gave_cuedlvsr).crop(tmin=-2,tmax=0).freqs
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(deepcopy(gave_cuedlvsr).crop(tmin=-2,tmax=0).times, deepcopy(gave_cuedlvsr).crop(tmin=-2,tmax=0).freqs, cvsi_clvsr, levels = 100, cmap = 'RdBu_r', antialiased = False, vmin = -2, vmax = 2)
ax.vlines([0, -1.5], linestyle = 'dashed', lw = .75, ymin=1, ymax = 39)
ax.set_title('cued left vs right')
for mask in masks:
    bigmask = np.kron(mask, np.ones((10,10)))
    ax.contour(bigmask, extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), antialiased = False, colors = 'black', linewidths = .5)

#%%
    
gave_plvsr_cvsn = mne.grand_average(data_t['plvsr_cvsn']); gave_plvsr_cvsn.data = toverparam(data_t['plvsr_cvsn']); gave_plvsr_cvsn.drop_channels(['RM'])
gave_plvsr_cvsn.plot_joint(title = 'cued leftvsneutral - cued rightvsneutral, t over betas',
                           topomap_args = topoargs_t, timefreqs = timefreqs_cue)
    
    
cvsi_plvsr_cvsn = np.subtract(np.nanmean(deepcopy(gave_plvsr_cvsn).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(gave_plvsr_cvsn).pick_channels(visleftchans).data, 0))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(gave_plvsr_cvsn.times, gave_plvsr_cvsn.freqs, cvsi_plvsr_cvsn, levels = 100, cmap = 'RdBu_r', antialiased = False, vmin = -2, vmax = 2)
ax.vlines([0, -1.5], linestyle = 'dashed', lw = .75, ymin=1, ymax = 39)

   
x_clvsrvsn = np.empty(shape = (subs.size, gave_plvsr_cvsn.freqs.size, gave_plvsr_cvsn.times.size))
for i in range(subs.size):
    iclvsrvsn = np.subtract(np.nanmean(deepcopy(data_t['plvsr_cvsn'][i]).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(data_t['plvsr_cvsn'][i]).pick_channels(visleftchans).data,0))
    x_clvsrvsn[i,:,:] = iclvsrvsn

t_lvsr, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(x_clvsrvsn, n_permutations='all')
masks = np.asarray(clusters)[cluster_pv<0.2]
 
    
times, freqs = gave_plvsr_cvsn.times, gave_plvsr_cvsn.freqs
fig = plt.figure()
ax = fig.add_subplot(111)
ax.contourf(gave_plvsr_cvsn.times, gave_plvsr_cvsn.freqs, cvsi_plvsr_cvsn, levels = 100, cmap = 'RdBu_r', antialiased = False, vmin = -2, vmax = 2)
ax.vlines([0, -1.5], linestyle = 'dashed', lw = .75, ymin=1, ymax = 39)
ax.set_title('(cued left vs neutral) - (cued right vs neutral)')
for mask in masks:
    bigmask = np.kron(mask, np.ones((10,10)))
    ax.contour(bigmask, extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), antialiased = False, colors = 'black', linewidths = .5)
    
#%%

gave_dtlvsr = mne.grand_average(data_t['dt_cued_lvsr']); gave_dtlvsr.data = toverparam(data_t['dt_cued_lvsr']); gave_dtlvsr.drop_channels(['RM'])
gave_dtlvsr.plot_joint(title = 'DT ~ cued left vs right, t over tstats', topomap_args = topoargs_t, timefreqs = timefreqs_cue)

cvsi_dtlvsr = np.subtract(np.nanmean(deepcopy(gave_dtlvsr).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(gave_dtlvsr).pick_channels(visleftchans).data, 0))
    

times, freqs = gave_dtlvsr.times, gave_dtlvsr.freqs


fig = plt.figure()
ax = fig.add_subplot(111)
#ax.pcolormesh(gave_dtlvsr.times, gave_dtlvsr.freqs, cvsi_dtlvsr, cmap = 'RdBu_r', antialiased = False, vmin = -2, vmax = 2)
ax.imshow(cvsi_dtlvsr, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'hanning', origin = 'lower', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)))
ax.vlines([0, -1.5], linestyle = 'dashed', lw = .75, ymin=1, ymax = 39)
ax.set_ylim(1,39)
ax.set_title('DT ~ contra-ipside to cued lvsr')

x_dtlvsr = np.empty(shape = (subs.size, gave_dtlvsr.freqs.size, gave_dtlvsr.times.size))
for i in range(subs.size):
    idtlvsr = np.subtract(np.nanmean(deepcopy(data_t['dt_cued_lvsr'][i]).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(data_t['dt_cued_lvsr'][i]).pick_channels(visleftchans).data,0))
    x_dtlvsr[i,:,:] = idtlvsr
    
t_lvsr, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(x_dtlvsr, n_permutations='all')
masks = np.asarray(clusters)[cluster_pv<0.05]
     
times, freqs = gave_dtlvsr.times, gave_dtlvsr.freqs
fig = plt.figure()
ax = fig.add_subplot(111)
contour = ax.imshow(cvsi_dtlvsr, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'hanning', origin = 'lower', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)))
ax.vlines([0, -1.5], linestyle = 'dashed', lw = .75, ymin=1, ymax = 39)
ax.set_title('DT ~ contra - ipsi for cued left vs right')
fig.colorbar(contour, ax = ax)
for mask in masks:
    bigmask = np.kron(mask, np.ones((10,10)))
    ax.contour(bigmask, extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), antialiased = False, colors = 'black', linewidths = .5)
    
#%%
    
    
gave_errorlvsr = mne.grand_average(data_t['error_cued_lvsr']); gave_errorlvsr.data = toverparam(data_t['error_cued_lvsr']); gave_errorlvsr.drop_channels(['RM'])
gave_errorlvsr.plot_joint(title = 'error ~ cued left vs right, t over betas', topomap_args = topoargs_t, timefreqs = timefreqs_cue)

cvsi_errorlvsr = np.subtract(np.nanmean(deepcopy(gave_errorlvsr).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(gave_errorlvsr).pick_channels(visleftchans).data, 0))
    
fig = plt.figure()
ax = fig.add_subplot(111)
contour = ax.imshow(cvsi_errorlvsr, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2,
                    interpolation = 'none', origin = 'lower', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)))
ax.vlines([0, -1.5], linestyle = 'dashed', lw = .75, ymin=1, ymax = 39)
ax.set_title('Error ~ contra - ipsi for cued left vs right')
fig.colorbar(contour, ax = ax)

#%%
#the confidence regressor hasn't been changed, so it reflects confidence widths
#higher values = larger confidence widths = less confident
#smaller values = narrower confidence width = more confident


gave_conflvsr = mne.grand_average(data_t['conf_cued_lvsr']); gave_conflvsr.data = toverparam(data_t['conf_cued_lvsr']); gave_conflvsr.drop_channels(['RM'])
gave_conflvsr.plot_joint(title = 'confidence ~ cued left vs right, t over betas', topomap_args = topoargs_t, timefreqs = timefreqs_cue)


cvsi_conflvsr = np.subtract(np.nanmean(deepcopy(gave_conflvsr).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(gave_conflvsr).pick_channels(visleftchans).data, 0))

    
fig = plt.figure()
ax = fig.add_subplot(111)
contour = ax.imshow(cvsi_conflvsr, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2,
                    interpolation = 'none', origin = 'lower', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)))
ax.vlines([0, -1.5], linestyle = 'dashed', lw = .75, ymin=1, ymax = 39)
ax.set_title('Confidence width ~ contra - ipsi for cued left vs right')
fig.colorbar(contour, ax = ax)

x_conflvsr = np.empty(shape = (subs.size, gave_dtlvsr.freqs.size, gave_dtlvsr.times.size))
for i in range(subs.size):
    iconflvsr = np.subtract(np.nanmean(deepcopy(data_t['conf_cued_lvsr'][i]).pick_channels(visrightchans).data,0), np.nanmean(deepcopy(data_t['conf_cued_lvsr'][i]).pick_channels(visleftchans).data,0))
    x_conflvsr[i,:,:] = iconflvsr
    
t_lvsr, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(x_conflvsr, n_permutations='all')
masks = np.asarray(clusters)[cluster_pv<0.25]
     


times, freqs = gave_conflvsr.times, gave_conflvsr.freqs
fig = plt.figure()
ax = fig.add_subplot(111)
contour = ax.imshow(cvsi_conflvsr, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'hanning', origin = 'lower', extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)))
ax.vlines([0, -1.5], linestyle = 'dashed', lw = .75, ymin=1, ymax = 39)
ax.set_title('Confidence width ~ contra - ipsi for cued left vs right')
fig.colorbar(contour, ax = ax)
for mask in masks:
    bigmask = np.kron(mask, np.ones((10,10)))
    ax.contour(bigmask, extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), antialiased = False, colors = 'black', linewidths = .5)
    






