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
#for i in data_baselined_t['pleft_cued']:
#    i.times = timesrel2cue

gave_pleft_cued = mne.grand_average(data_baselined_t['pleft_cued']); gave_pleft_cued.data = toverparam(data_baselined_t['pleft_cued'])
gave_pleft_cued.drop_channels(['RM'])
gave_pleft_cued.times = timesrel2cue

gave_pleft_cvsn = mne.grand_average(data_t['pleft_cvsn']); gave_pleft_cvsn.data = toverparam(data_t['pleft_cvsn'])
gave_pleft_cvsn.times = timesrel2cue

#contra vs ipsi should be done within subject and then averaged
alldata_clvsn = np.empty(shape = (subs.size, allfreqs.size, timesrel2cue.size))
for i in range(subs.size):
    alldata_clvsn[i,:,:] = np.subtract(np.nanmean( deepcopy(data_t['pleft_cvsn'][i]).pick_channels(visrightchans).data,0),
                                       np.nanmean( deepcopy(data_t['pleft_cvsn'][i]).pick_channels(visleftchans).data, 0))
#ttest over this
cvsi_clvsn = sp.stats.ttest_1samp(alldata_clvsn, popmean = 0, axis = 0)[0]


#cvsi_clvsn = np.subtract(np.nanmean(deepcopy(gave_pleft_cvsn).pick_channels(visrightchans).data,0),
#                         0)#np.nanmean(deepcopy(gave_pleft_cvsn).pick_channels(visleftchans).data, 0))

pleft_pjoint = gave_pleft_cvsn.plot_joint(title = 'probed left item cued vs neutral, t over tstats', timefreqs = timefreqs_cue_rel2cue, vmin=-2,vmax=2,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -3, vmax = 3))
#six axes in pleft_pjoint.axes
axes = pleft_pjoint.axes
#the first axis is the tfr image, so lets replace it with the actual image that we want (i.e. just the right visual channels)
axes[0].clear()
tfrplot = axes[0].imshow(cvsi_clvsn, cmap = 'RdBu_r', aspect = 'auto', vmin = -3, vmax = 3, interpolation = 'gaussian',
                 origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
axes[0].set_xlabel('Time rel. to cue onset (s)')
axes[0].set_ylabel('Frequency (Hz)')
axes[0].vlines([0, 1.5], linestyle = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = pleft_pjoint.add_axes([0.95, 0.15, .02, .35])
pleft_pjoint.colorbar(tfrplot, cax = cbaxes)

pleft_pjoint.savefig(fname = op.join(figpath, 'cuedlvsn_jointplot_cvsivis-tfr.eps'), dpi = 300, format = 'eps', bbox_inches = 'tight')
pleft_pjoint.savefig(fname = op.join(figpath, 'cuedlvsn_jointplot_cvsivis-tfr.pdf'), dpi = 300, format = 'pdf', bbox_inches = 'tight')



#%%

#for i in data_baselined_t['pright_cued']:
#    i.times = timesrel2cue
    
gave_pright_cued = mne.grand_average(data_baselined_t['pright_cued']); gave_pright_cued.data = toverparam(data_baselined_t['pright_cued'])
gave_pright_cued.drop_channels(['RM'])
gave_pright_cued.times = timesrel2cue

gave_pright_cvsn = mne.grand_average(data_t['pright_cvsn']); gave_pright_cvsn.data = toverparam(data_t['pright_cvsn'])
gave_pright_cvsn.times = timesrel2cue

alldata_crvsn = np.empty(shape = (subs.size, allfreqs.size, timesrel2cue.size))
for i in range(subs.size):
    alldata_crvsn[i,:,:] = np.subtract(np.nanmean(deepcopy(data_t['pright_cvsn'][i]).pick_channels(visleftchans).data,0),
                                       np.nanmean(deepcopy(data_t['pright_cvsn'][i]).pick_channels(visrightchans).data,0))
cvsi_crvsn = sp.stats.ttest_1samp(alldata_crvsn, popmean = 0, axis = 0)[0]


#cvsi_crvsn = np.subtract(np.nanmean(deepcopy(gave_pright_cvsn).pick_channels(visleftchans).data,0),
#                         0)#np.nanmean(deepcopy(gave_pright_cvsn).pick_channels(visrightchans).data, 0))

pright_pjoint = gave_pright_cvsn.plot_joint(title = 'probed right item cued vs neutral, t over tstats', timefreqs = timefreqs_cue_rel2cue, vmin=-2,vmax=2,
                            topomap_args = dict(outlines = 'head', contours = 0, vmin = -3, vmax = 3))
axes = pright_pjoint.axes
axes[0].clear()
tfrplot = axes[0].imshow(cvsi_crvsn, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'gaussian',
                 origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
axes[0].set_xlabel('Time rel. to cue onset (s)')
axes[0].set_ylabel('Frequency (Hz)')
axes[0].vlines([0, 1.5], linestyle = 'dashed', lw = 2, color = '#000000', ymin = 1, ymax = 39)
cbaxes = pright_pjoint.add_axes([.95, .15, .02, .35])
pright_pjoint.colorbar(tfrplot, cax = cbaxes)

pright_pjoint.savefig(fname = op.join(figpath, 'cuedrvsn_jointplot_cvsivis-tfr.eps'), dpi = 300, format = 'eps', bbox_inches = 'tight')
pright_pjoint.savefig(fname = op.join(figpath, 'cuedrvsn_jointplot_cvsivis-tfr.pdf'), dpi = 300, format = 'pdf', bbox_inches = 'tight')





#%%
for i in data_t['plvsr_cued']:
    i.times = timesrel2cue
gave_cuedlvsr = mne.grand_average(data_t['plvsr_cued']); gave_cuedlvsr.data = toverparam(data_t['plvsr_cued']); gave_cuedlvsr.drop_channels(['RM'])


plot_single_sub = True
isub = 18
sub = np.squeeze(np.where(np.isin(subs, isub)))
if plot_single_sub:
    tmp = deepcopy(data_t['plvsr_cued'][sub])
    tmp = tmp.drop_channels(['RM'])
    tmp.times = timesrel2cue
    tmp.plot_joint(
            picks = 'eeg',
            timefreqs = timefreqs_cue_rel2cue,
            topomap_args = topoargs_t)



#gave_cuedlvsr.plot_joint(title = 'cued left vs right, t over tstats',
#                         timefreqs = timefreqs_cue, topomap_args = topoargs_t)


cvsi_clvsr = np.empty(shape = (subs.size, allfreqs.size, timesrel2cue.size))
for i in range(subs.size):
    cvsi_clvsr[i,:,:] = np.subtract(np.nanmean( deepcopy(data_t['plvsr_cued'][i]).pick_channels(visrightchans).data,0),
                                    np.nanmean( deepcopy(data_t['plvsr_cued'][i]).pick_channels(visleftchans).data,0))
    
cvsi_clvsr = sp.stats.ttest_1samp(cvsi_clvsr, popmean=0, axis = 0)[0]

clvsr_joint = gave_cuedlvsr.plot_joint(title = 'cued left vs right, t over tstats', timefreqs = timefreqs_cue_rel2cue,
                                       topomap_args = topoargs_t)
axes = clvsr_joint.axes
axes[0].clear()
tfrplot = axes[0].imshow(cvsi_clvsr, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'gaussian',
                 origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
axes[0].set_xlabel('Time rel. to cue onset (s)')
axes[0].set_ylabel('Frequency (Hz)')
axes[0].vlines([0, 1.5], linestyle = 'dashed', lw = 2, ymin = 1, ymax = 39, color = '#000000')
cbaxes = clvsr_joint.add_axes([.95, .15, .02, .35])
clvsr_joint.colorbar(tfrplot, cax = cbaxes)

#
#clvsr_joint.savefig(fname = op.join(figpath, 'cuedlvsr_jointplot_cvsivis-tfr.eps'), dpi = 300, format = 'eps', bbox_inches = 'tight')
#clvsr_joint.savefig(fname = op.join(figpath, 'cuedlvsr_jointplot_cvsivis-tfr.pdf'), dpi = 300, format = 'pdf', bbox_inches = 'tight')
#the figure produced above has a tfr plot that shows t over tstats for the contra-ipsi of this calculated per subject first


#plot topographies for the two things that look like clusters of lateralisation for cued left vs right (one during delay, one pre-probe)
#this is the topography for that cluster during the memory delay for cued left vs right
fig = gave_cuedlvsr.plot_topomap(tmin = .7 , tmax = 1.0, fmin = 8, fmax = 12, vmin = -2, vmax = 2, outlines = 'head', contours = 0, cbar_fmt  = '%.1f', unit = 't-value')

#this is the topo for the bit just pre-probe that comes out at p=.19
fig = gave_cuedlvsr.plot_topomap(tmin = 1.2, tmax = 1.5, fmin = 8, fmax = 12, vmin = -2, vmax = 2, outlines = 'head', contours = 0, cbar_fmt  = '%.1f', unit = 't-value')
fig.savefig(fname = op.join(figpath, 'cuedlvsr_topo_preprobecluster_alpha.eps'), format = 'eps', dpi = 300, bbox_inches = 'tight')
fig.savefig(fname = op.join(figpath, 'cuedlvsr_topo_preprobecluster_alpha.pdf'), format = 'pdf', dpi = 300, bbox_inches = 'tight')



#which is different to the plot below (the above is better) where we t over the tstats per subject and then do cvsi on the output of that (i.e. combine subjects then cvsi)
clvsr_cvsi = np.subtract(np.nanmean(deepcopy(gave_cuedlvsr).pick_channels(visrightchans).data,0),
                         np.nanmean(deepcopy(gave_cuedlvsr).pick_channels(visleftchans).data,0))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(clvsr_cvsi, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'gaussian',
                 origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
ax.set_xlabel('Time rel. to cue onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0, 1.5], linestyle = 'dashed', lw = 2, ymin = 1, ymax = 39, color = '#000000')

#%%

tmin, tmax = None, None
x_clvsr = np.empty(shape = (subs.size, deepcopy(gave_cuedlvsr).crop(tmin=tmin,tmax=tmax).freqs.size, deepcopy(gave_cuedlvsr).crop(tmin=tmin,tmax=tmax).times.size))
for i in range(subs.size):
    iclvsr = np.subtract(np.nanmean(deepcopy(data_t['plvsr_cued'][i]).crop(tmin=tmin,tmax=tmax).pick_channels(visrightchans).data,0),
                         np.nanmean(deepcopy(data_t['plvsr_cued'][i]).crop(tmin=tmin,tmax=tmax).pick_channels(visleftchans).data,0))
    x_clvsr[i,:,:] = iclvsr

t_lvsr, clusters, cluster_pv, _ = mne.stats.permutation_cluster_1samp_test(x_clvsr, n_permutations='all')
masks = np.asarray(clusters)[cluster_pv<0.1]

times, freqs =deepcopy(gave_cuedlvsr).crop(tmin=tmin,tmax=tmax).times,  deepcopy(gave_cuedlvsr).crop(tmin=tmin,tmax=tmax).freqs
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.contourf(deepcopy(gave_cuedlvsr).crop(tmin=tmin,tmax=tmax).times, deepcopy(gave_cuedlvsr).crop(tmin=tmin,tmax=tmax).freqs, cvsi_clvsr, levels = 100, cmap = 'RdBu_r', antialiased = False, vmin = -2, vmax = 2)
ax.imshow(cvsi_clvsr, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'gaussian',
          origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
ax.vlines([0, 1.5], linestyle = 'dashed', lw = .75, ymin=1, ymax = 39)
ax.set_title('cued left vs right')
for mask in masks:
    bigmask = np.kron(mask, np.ones((10,10)))
    ax.contour(bigmask, extent = (np.min(times), np.max(times), np.min(freqs), np.max(freqs)), antialiased = False, colors = 'black', linewidths = .5)

#%%
for i in data_t['plvsr_cvsn']:
    i.times = timesrel2cue
    
gave_plvsr_cvsn = mne.grand_average(data_t['plvsr_cvsn']); gave_plvsr_cvsn.data = toverparam(data_t['plvsr_cvsn']); gave_plvsr_cvsn.drop_channels(['RM'])
gave_plvsr_cvsn.plot_joint(title = 'cued leftvsneutral - cued rightvsneutral, t over betas',
                           topomap_args = topoargs_t, timefreqs = timefreqs_cue_rel2cue)
    
    

cvsi_clvsrvsn = np.empty(shape = (subs.size, allfreqs.size, timesrel2cue.size))
for i in range(subs.size):
    cvsi_clvsrvsn[i,:,:] = np.subtract(np.nanmean( deepcopy(data_t['plvsr_cvsn'][i]).pick_channels(visrightchans).data,0),
                                       np.nanmean( deepcopy(data_t['plvsr_cvsn'][i]).pick_channels(visleftchans).data,0))
    
cvsi_clvsrvsn = sp.stats.ttest_1samp(cvsi_clvsrvsn, popmean=0, axis = 0)[0]

#this plot isnt as easy to interpret as it's the interaction
clvsrvsn_joint = gave_plvsr_cvsn.plot_joint(title = 'cued left vs right vs neutral, t over tstats', timefreqs = timefreqs_cue_rel2cue,
                                       topomap_args = topoargs_t)
axes = clvsrvsn_joint.axes
axes[0].clear()
tfrplot = axes[0].imshow(cvsi_clvsrvsn, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'gaussian',
                 origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
axes[0].set_xlabel('Time rel. to cue onset (s)')
axes[0].set_ylabel('Frequency (Hz)')
axes[0].vlines([0, 1.5], linestyle = 'dashed', lw = 2, ymin = 1, ymax = 39, color = '#000000')
cbaxes = clvsrvsn_joint.add_axes([.95, .15, .02, .25])
clvsrvsn_joint.colorbar(tfrplot, cax = cbaxes)
#the figure produced above has a tfr plot that shows t over tstats for the contra-ipsi of this calculated per subject first
#%%

for i in data_t['dt_cued_lvsr']:
    i.times = timesrel2cue

gave_dtlvsr = mne.grand_average(data_t['dt_cued_lvsr']); gave_dtlvsr.data = toverparam(data_t['dt_cued_lvsr']); gave_dtlvsr.drop_channels(['RM'])
#gave_dtlvsr.plot_joint(title = 'DT ~ cued left vs right, t over tstats', topomap_args = topoargs_t, timefreqs = timefreqs_cue_rel2cue)



cvsi_dtlvsr = np.empty(shape = (subs.size, allfreqs.size, timesrel2cue.size))
for i in range(subs.size):
    cvsi_dtlvsr[i,:,:] = np.subtract(np.nanmean( deepcopy(data_t['dt_cued_lvsr'][i]).pick_channels(visrightchans).data,0),
                                     np.nanmean( deepcopy(data_t['dt_cued_lvsr'][i]).pick_channels(visleftchans).data,0))
    
cvsi_dtlvsr = sp.stats.ttest_1samp(cvsi_dtlvsr, popmean=0, axis = 0)[0]

dtlvsr_joint = gave_dtlvsr.plot_joint(title = 'DT ~ cued left vs right, t over tstats', timefreqs = timefreqs_cue_rel2cue,
                                       topomap_args = topoargs_t)
axes = dtlvsr_joint.axes
axes[0].clear()
tfrplot = axes[0].imshow(cvsi_dtlvsr, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'gaussian',
                 origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
axes[0].set_xlabel('Time rel. to cue onset (s)')
axes[0].set_ylabel('Frequency (Hz)')
axes[0].vlines([0, 1.5], linestyle = 'dashed', lw = 2, ymin = 1, ymax = 39, color = '#000000')
cbaxes = dtlvsr_joint.add_axes([0.95,.15, .02, .35])
dtlvsr_joint.colorbar(tfrplot, cax = cbaxes)
#the figure produced above has a tfr plot that shows t over tstats for the contra-ipsi of this calculated per subject first

dtlvsr_joint.savefig(fname = op.join(figpath, 'dtlvsr_jointplot_cvsivis-tfr.eps'), dpi = 300, format = 'eps', bbox_inches = 'tight')
dtlvsr_joint.savefig(fname = op.join(figpath, 'dtlvsr_jointplot_cvsivis-tfr.pdf'), dpi = 300, format = 'pdf', bbox_inches = 'tight')



#this is the topography for that cluster just pre-probe in DT
fig = gave_dtlvsr.plot_topomap(tmin=.9, tmax =1.5, fmin = 8, fmax = 12, vmin = -2, vmax = 2, outlines = 'head', contours = 0, cbar_fmt  = '%.1f', unit = 't-value')
fig.savefig(fname = op.join(figpath, 'dtlvsr_topo_preprobecluster_alpha.eps'), format = 'eps', dpi = 300, bbox_inches = 'tight')
fig.savefig(fname = op.join(figpath, 'dtlvsr_topo_preprobecluster_alpha.pdf'), format = 'pdf', dpi = 300, bbox_inches = 'tight')


#which is different to the plot below (the above is better) where we t over the tstats per subject and then do cvsi on the output of that (i.e. combine subjects then cvsi)
dtlvsr_cvsi = np.subtract(np.nanmean(deepcopy(gave_dtlvsr).pick_channels(visrightchans).data,0),
                          np.nanmean(deepcopy(gave_dtlvsr).pick_channels(visleftchans).data,0))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(dtlvsr_cvsi, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'gaussian',
                 origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
ax.set_xlabel('Time rel. to cue onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0, 1.5], linestyle = 'dashed', lw = 2, ymin = 1, ymax = 39, color = '#000000')

    
#%%
    
for i in data_t['error_cued_lvsr']:
    i.times = timesrel2cue
    
gave_errorlvsr = mne.grand_average(data_t['error_cued_lvsr']); gave_errorlvsr.data = toverparam(data_t['error_cued_lvsr']); gave_errorlvsr.drop_channels(['RM'])
#gave_errorlvsr.plot_joint(title = 'error ~ cued left vs right, t over betas', topomap_args = topoargs_t, timefreqs = timefreqs_cue_rel2cue)


cvsi_errlvsr = np.empty(shape = (subs.size, allfreqs.size, timesrel2cue.size))
for i in range(subs.size):
    cvsi_errlvsr[i,:,:] = np.subtract(np.nanmean( deepcopy(data_t['error_cued_lvsr'][i]).pick_channels(visrightchans).data,0),
                                      np.nanmean( deepcopy(data_t['error_cued_lvsr'][i]).pick_channels(visleftchans).data,0))
    
cvsi_errlvsr = sp.stats.ttest_1samp(cvsi_errlvsr, popmean=0, axis = 0)[0]

errlvsr_joint = gave_errorlvsr.plot_joint(title = 'Error ~ cued left vs right, t over tstats', timefreqs = timefreqs_cue_rel2cue,
                                       topomap_args = topoargs_t)
axes = errlvsr_joint.axes
axes[0].clear()
tfrplot = axes[0].imshow(cvsi_errlvsr, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'gaussian',
                 origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
axes[0].set_xlabel('Time rel. to cue onset (s)')
axes[0].set_ylabel('Frequency (Hz)')
axes[0].vlines([0, 1.5], linestyle = 'dashed', lw = 2, ymin = 1, ymax = 39, color = '#000000')
cbaxes = errlvsr_joint.add_axes([0.95, .15, .03, .35])
errlvsr_joint.colorbar(tfrplot, cax = cbaxes)
#the figure produced above has a tfr plot that shows t over tstats for the contra-ipsi of this calculated per subject first


errlvsr_joint.savefig(fname = op.join(figpath, 'errlvsr_jointplot_cvsivis-tfr.eps'), dpi = 300, format = 'eps', bbox_inches = 'tight')
errlvsr_joint.savefig(fname = op.join(figpath, 'errlvsr_jointplot_cvsivis-tfr.pdf'), dpi = 300, format = 'pdf', bbox_inches = 'tight')



#this is the topography for that cluster just during the delay in error
fig = gave_errorlvsr.plot_topomap(tmin=.6, tmax =.9, fmin = 8, fmax = 12, vmin = -2, vmax = 2, outlines = 'head', contours = 0, cbar_fmt  = '%.1f', unit = 't-value')
fig.savefig(fname = op.join(figpath, 'errorlvsr_topo_delaycluster_alpha.eps'), format = 'eps', dpi = 300, bbox_inches = 'tight')
fig.savefig(fname = op.join(figpath, 'errorlvsr_topo_delaycluster_alpha.pdf'), format = 'pdf', dpi = 300, bbox_inches = 'tight')


#which is different to the plot below (the above is better) where we t over the tstats per subject and then do cvsi on the output of that (i.e. combine subjects then cvsi)
errlvsr_cvsi = np.subtract(np.nanmean(deepcopy(gave_errorlvsr).pick_channels(visrightchans).data,0),
                           np.nanmean(deepcopy(gave_errorlvsr).pick_channels(visleftchans).data,0))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(errlvsr_cvsi, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'gaussian',
                 origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
ax.set_xlabel('Time rel. to cue onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0, 1.5], linestyle = 'dashed', lw = 2, ymin = 1, ymax = 39, color = '#000000')





#%%
#the confidence regressor hasn't been changed, so it reflects confidence widths
#higher values = larger confidence widths = less confident
#smaller values = narrower confidence width = more confident


for i in data_t['conf_cued_lvsr']:
    i.times = timesrel2cue
    
gave_conflvsr = mne.grand_average(data_t['conf_cued_lvsr']); gave_conflvsr.data = toverparam(data_t['conf_cued_lvsr']); gave_conflvsr.drop_channels(['RM'])
#gave_conflvsr.plot_joint(title = 'confidence ~ cued left vs right, t over betas', topomap_args = topoargs_t, timefreqs = timefreqs_cue)



cvsi_cwlvsr = np.empty(shape = (subs.size, allfreqs.size, timesrel2cue.size))
for i in range(subs.size):
    cvsi_cwlvsr[i,:,:] = np.subtract(np.nanmean( deepcopy(data_t['conf_cued_lvsr'][i]).pick_channels(visrightchans).data,0),
                                     np.nanmean( deepcopy(data_t['conf_cued_lvsr'][i]).pick_channels(visleftchans).data,0))
    
cvsi_cwlvsr = sp.stats.ttest_1samp(cvsi_cwlvsr, popmean=0, axis = 0)[0]

cwlvsr_joint = gave_conflvsr.plot_joint(title = 'confidence width ~ cued left vs right, t over tstats', timefreqs = timefreqs_cue_rel2cue,
                                       topomap_args = topoargs_t)
axes = cwlvsr_joint.axes
axes[0].clear()
tfrplot = axes[0].imshow(cvsi_cwlvsr, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'gaussian',
                 origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
axes[0].set_xlabel('Time rel. to cue onset (s)')
axes[0].set_ylabel('Frequency (Hz)')
axes[0].vlines([0, 1.5], linestyle = 'dashed', lw = 2, ymin = 1, ymax = 39, color = '#000000')
cbaxes = cwlvsr_joint.add_axes([0.95, .15, .03, .35])
cwlvsr_joint.colorbar(tfrplot, cax = cbaxes)
#the figure produced above has a tfr plot that shows t over tstats for the contra-ipsi of this calculated per subject first


cwlvsr_joint.savefig(fname = op.join(figpath, 'confwidthlvsr_jointplot_cvsivis-tfr.eps'), dpi = 300, format = 'eps', bbox_inches = 'tight')
cwlvsr_joint.savefig(fname = op.join(figpath, 'confwidthlvsr_jointplot_cvsivis-tfr.pdf'), dpi = 300, format = 'pdf', bbox_inches = 'tight')


#this is the topography for that cluster just pre-probe in confidence
fig = gave_conflvsr.plot_topomap(tmin=1, tmax =1.5, fmin = 8, fmax = 12, vmin = -2, vmax = 2, outlines = 'head', contours = 0, cbar_fmt  = '%.1f', unit = 't-value')
fig.savefig(fname = op.join(figpath, 'confwidthlvsr_topo_preprobecluster_alpha.eps'), format = 'eps', dpi = 300, bbox_inches = 'tight')
fig.savefig(fname = op.join(figpath, 'confwidthlvsr_topo_preprobecluster_alpha.pdf'), format = 'pdf', dpi = 300, bbox_inches = 'tight')


#which is different to the plot below (the above is better) where we t over the tstats per subject and then do cvsi on the output of that (i.e. combine subjects then cvsi)
errlvsr_cvsi = np.subtract(np.nanmean(deepcopy(gave_errorlvsr).pick_channels(visrightchans).data,0),
                           np.nanmean(deepcopy(gave_errorlvsr).pick_channels(visleftchans).data,0))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(errlvsr_cvsi, cmap = 'RdBu_r', aspect = 'auto', vmin = -2, vmax = 2, interpolation = 'gaussian',
                 origin = 'lower', extent = (np.min(timesrel2cue), np.max(timesrel2cue), np.min(allfreqs), np.max(allfreqs)))
ax.set_xlabel('Time rel. to cue onset (s)')
ax.set_ylabel('Frequency (Hz)')
ax.vlines([0, 1.5], linestyle = 'dashed', lw = 2, ymin = 1, ymax = 39, color = '#000000')
#%%

#can we condense visual alpha lateralisation (selection) down to one trace 
#and plot how it relates to error, confidence and reaction time so we can see their relative time courses

for sigma in [1, 2, 3, 4, 5]:
    err_lvsr = np.empty(shape = (subs.size, timesrel2cue.size))
    for i in range(subs.size):
        tmp = deepcopy(data_t['error_cued_lvsr'][i])
        tmp.drop_channels(['RM']).crop(fmin=8, fmax=12)
        tmp_visright = np.nanmean(np.nanmean(deepcopy(tmp).pick_channels(visrightchans).data, 1), 0) #this is now just one trace
        tmp_visleft  = np.nanmean(np.nanmean(deepcopy(tmp).pick_channels(visleftchans).data,  1), 0)
        err_lvsr[i,:] = sp.ndimage.gaussian_filter1d(np.subtract(tmp_visright, tmp_visleft), sigma = sigma)
    #extract just the right vs left visual channels now so we have one trace to plot:
    
    #do ttest over this array now
    groupt_err_lvsr = sp.stats.ttest_1samp(err_lvsr, popmean=0, axis=0)[0]
    # - - - - - - - - - - - - - - - - - - - -  - - - - - - - 
    #same for confidence
    conf_lvsr = np.empty(shape = (subs.size, timesrel2cue.size))
    for i in range(subs.size):
        tmp = deepcopy(data_t['conf_cued_lvsr'][i])
        tmp.drop_channels(['RM']).crop(fmin=8, fmax=12)
        tmp_visright = np.nanmean(np.nanmean(deepcopy(tmp).pick_channels(visrightchans).data, 1), 0) #this is now just one trace
        tmp_visleft  = np.nanmean(np.nanmean(deepcopy(tmp).pick_channels(visleftchans).data,  1), 0)
        conf_lvsr[i,:] = sp.ndimage.gaussian_filter1d(np.subtract(tmp_visright, tmp_visleft), sigma = sigma)
    #extract just the right vs left visual channels now so we have one trace to plot:
    
    #do ttest over this array now
    groupt_conf_lvsr = sp.stats.ttest_1samp(conf_lvsr, popmean=0, axis=0)[0]
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    #same for decision time now
    dt_lvsr = np.empty(shape = (subs.size, timesrel2cue.size))
    for i in range(subs.size):
        tmp = deepcopy(data_t['dt_cued_lvsr'][i])
        tmp.drop_channels(['RM']).crop(fmin=8, fmax=12)
        tmp_visright = np.nanmean(np.nanmean(deepcopy(tmp).pick_channels(visrightchans).data, 1), 0) #this is now just one trace
        tmp_visleft  = np.nanmean(np.nanmean(deepcopy(tmp).pick_channels(visleftchans).data,  1), 0)
        dt_lvsr[i,:] = sp.ndimage.gaussian_filter1d(np.subtract(tmp_visright, tmp_visleft), sigma = sigma)
    #extract just the right vs left visual channels now so we have one trace to plot:
    
    #do ttest over this array now
    groupt_dt_lvsr = sp.stats.ttest_1samp(dt_lvsr, popmean=0, axis=0)[0]
    
    fig = plt.figure()
    fig.suptitle('sigma = %d'%sigma)
    ax = fig.add_subplot(111)
    ax.plot(timesrel2cue, groupt_err_lvsr,  label = 'error', color = '#66c2a5')
    ax.plot(timesrel2cue, groupt_conf_lvsr, label = 'confidence', color = '#fc8d62')
    ax.plot(timesrel2cue, groupt_dt_lvsr,   label = 'decision time', color = '#8da0cb')
    ax.hlines(y = 0, linestyles = 'dashed', lw = .75, xmin=timesrel2cue.min(), xmax = timesrel2cue.max(), color = '#000000')
    ax.vlines([0, 1.5], ymin = -3, ymax = 3, linestyles = 'dashed', lw = .75, color = '#000000')
    fig.legend(loc = 'lower left')
#%%
err_lvsr = np.empty(shape = (subs.size, timesrel2cue.size))
for i in range(subs.size):
    tmp = deepcopy(data_t['error_cued_lvsr'][i])
    tmp.drop_channels(['RM']).crop(fmin=8, fmax=12)
    tmp_visright = np.nanmean(np.nanmean(deepcopy(tmp).pick_channels(visrightchans).data, 1), 0) #this is now just one trace
    tmp_visleft  = np.nanmean(np.nanmean(deepcopy(tmp).pick_channels(visleftchans).data,  1), 0)
    err_lvsr[i,:] = np.subtract(tmp_visright, tmp_visleft)
#extract just the right vs left visual channels now so we have one trace to plot:
#do ttest over this array now
groupt_err_lvsr = sp.stats.ttest_1samp(err_lvsr, popmean=0, axis=0)[0]
# - - - - - - - - - - - - - - - - - - - -  - - - - - - - 
#same for confidence
conf_lvsr = np.empty(shape = (subs.size, timesrel2cue.size))
for i in range(subs.size):
    tmp = deepcopy(data_t['conf_cued_lvsr'][i])
    tmp.drop_channels(['RM']).crop(fmin=8, fmax=12)
    tmp_visright = np.nanmean(np.nanmean(deepcopy(tmp).pick_channels(visrightchans).data, 1), 0) #this is now just one trace
    tmp_visleft  = np.nanmean(np.nanmean(deepcopy(tmp).pick_channels(visleftchans).data,  1), 0)
    conf_lvsr[i,:] = np.subtract(tmp_visright, tmp_visleft)
#extract just the right vs left visual channels now so we have one trace to plot:
#do ttest over this array now
groupt_conf_lvsr = sp.stats.ttest_1samp(conf_lvsr, popmean=0, axis=0)[0]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#same for decision time now
dt_lvsr = np.empty(shape = (subs.size, timesrel2cue.size))
for i in range(subs.size):
    tmp = deepcopy(data_t['dt_cued_lvsr'][i])
    tmp.drop_channels(['RM']).crop(fmin=8, fmax=12)
    tmp_visright = np.nanmean(np.nanmean(deepcopy(tmp).pick_channels(visrightchans).data, 1), 0) #this is now just one trace
    tmp_visleft  = np.nanmean(np.nanmean(deepcopy(tmp).pick_channels(visleftchans).data,  1), 0)
    dt_lvsr[i,:] = np.subtract(tmp_visright, tmp_visleft)
#extract just the right vs left visual channels now so we have one trace to plot:
#do ttest over this array now
groupt_dt_lvsr = sp.stats.ttest_1samp(dt_lvsr, popmean=0, axis=0)[0]


#fig = plt.figure()
#ax = fig.subplots(3,1)
#ax[0].plot(timesrel2cue, groupt_err_lvsr,  label = 'error', color = '#66c2a5')
#ax[0].plot(timesrel2cue, groupt_conf_lvsr, label = 'confidence', color = '#fc8d62')
#ax[0].plot(timesrel2cue, groupt_dt_lvsr,   label = 'decision time', color = '#8da0cb')
#  
#ax[1].plot(timesrel2cue, sp.ndimage.gaussian_filter1d(groupt_err_lvsr, sigma = 1),  label = 'error', color = '#66c2a5')
#ax[1].plot(timesrel2cue, sp.ndimage.gaussian_filter1d(groupt_conf_lvsr, sigma = 1), label = 'confidence', color = '#fc8d62')
#ax[1].plot(timesrel2cue, sp.ndimage.gaussian_filter1d(groupt_dt_lvsr, sigma = 1),   label = 'decision time', color = '#8da0cb')
#
#ax[2].plot(timesrel2cue, sp.ndimage.gaussian_filter1d(groupt_err_lvsr, sigma = 3),  label = 'error', color = '#66c2a5')
#ax[2].plot(timesrel2cue, sp.ndimage.gaussian_filter1d(groupt_conf_lvsr, sigma = 3),  label = 'error', color = '#fc8d62')
#ax[2].plot(timesrel2cue, sp.ndimage.gaussian_filter1d(groupt_dt_lvsr, sigma = 3),  label = 'error', color = '#8da0cb')
#for iax in ax: 
#    iax.hlines(y = 0, linestyles = 'dashed', lw = .75, xmin=timesrel2cue.min(), xmax = timesrel2cue.max(), color = '#000000')
#    iax.vlines([0, 1.5], ymin = -3, ymax = 3, linestyles = 'dashed', lw = .75, color = '#000000')
#fig.legend(loc = 'lower left')

fig = plt.figure(figsize = (10, 6))
fig.suptitle('timecourse of alpha lateralisation and its relationship to behaviour')
ax = fig.add_subplot(111)
ax.plot(timesrel2cue, sp.ndimage.gaussian_filter1d(groupt_err_lvsr, sigma = 3),  lw=2, label = 'error', color = '#66c2a5')
ax.plot(timesrel2cue, sp.ndimage.gaussian_filter1d(groupt_conf_lvsr, sigma = 3),  lw=2, label = 'confidence', color = '#fc8d62')
ax.plot(timesrel2cue, sp.ndimage.gaussian_filter1d(groupt_dt_lvsr, sigma = 3),  lw=2, label = 'reaction time', color = '#8da0cb')
ax.hlines(y = 0, linestyles = 'dashed', lw = 1, xmin=timesrel2cue.min(), xmax = timesrel2cue.max(), color = '#000000')
ax.vlines([0, 1.5], ymin = -3, ymax = 3, linestyles = 'dashed', lw = 1, color = '#000000')
ax.set_xlabel('Time rel. to cue onset (s)')
ax.set_ylabel('t-stat of contra vs ipsi lateralisation')
fig.legend(loc = 'upper right')
fig.savefig(fname = op.join(figpath, 'probelocked_alphalateralisation_rel2behaviour_timecourses.eps'), format = 'eps', dpi = 300, bbox_inches = 'tight')
fig.savefig(fname = op.join(figpath, 'probelocked_alphalateralisation_rel2behaviour_timecourses.pdf'), format = 'pdf', dpi = 300, bbox_inches = 'tight')






