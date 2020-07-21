#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 19:40:37 2019

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
from wmConfidence_funcs import gesd, plot_AR, toverparam, flip_tfrdata, runclustertest_tfr

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

figpath = op.join(wd, 'figures', 'eeg_figs', 'cuelocked', 'tfrglm1')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm
nsubs = subs.size

contrasts = ['pleft_cued', 'pleft_neutral', 'pright_cued', 'pright_neutral',
             'crvsn', 'clvsn','clvsr',
             'neutral', 'cued', 'cuedvsneut']


laplacian = False
if laplacian:
    lapstr = 'laplacian_'
else:
    lapstr = ''

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
    param = get_subject_info_wmConfidence(sub)
    
    for name in contrasts:
        data[name].append(mne.time_frequency.read_tfrs(            fname = op.join(param['path'], 'glms', 'cue','tfrglm1', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ lapstr + name + '_betas-tfr.h5'))[0])
        data_t[name].append(mne.time_frequency.read_tfrs(          fname = op.join(param['path'], 'glms', 'cue','tfrglm1', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ lapstr + name + '_tstats-tfr.h5'))[0])
        data_baselined[name].append(mne.time_frequency.read_tfrs(  fname = op.join(param['path'], 'glms', 'cue','tfrglm1', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ lapstr + name + '_betas_baselined-tfr.h5'))[0])
        data_baselined_t[name].append(mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'cue','tfrglm1', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ lapstr + name + '_tstats_baselined-tfr.h5'))[0])

#%% this is just going to snip out all the stuff prior to cue, so we have half a second before the cue, and the entire post cue period
for i in range(subs.size):
    for contrast in contrasts:
        for dat in [data, data_t, data_baselined, data_baselined_t]:
            dat[contrast][i].crop(tmin = -1, tmax = None)


#%%
alltimes = mne.grand_average(data['pleft_neutral']).times
allfreqs = mne.grand_average(data['pleft_neutral']).freqs
#%%
timefreqs       = {(.4, 10):(.4, 4),
                   (.6, 10):(.4, 4),
                   (.8, 10):(.4, 4),
                   (.4, 22):(.4, 16),
                   (.6, 22):(.4, 16),
                   (.8, 22):(.4, 16)}
timefreqs_alpha = {(.4, 10):(.4, 4),
                   (.6, 10):(.4, 4),
                   (.8, 10):(.4, 4),
                   (1., 10):(.4, 4),
                   (1.2, 10):(.4, 4)}
visleftchans  = ['PO3', 'PO7', 'O1']
visrightchans = ['PO4','PO8','O2']
motrightchans = ['C4']  #channels contra to the left hand (space bar)
motleftchans  = ['C3']   #channels contra to the right hand (mouse)
topoargs = dict(outlines= 'head', contours = 0)
topoargs_t = dict(outlines = 'head', contours = 0, vmin=-2, vmax = 2)
#%%
#here we're going to go through and plot the conditions in separate cells, and do stats there too

#first we will start with the basic trial stuff
stat = 'beta'
baselined = False
contrast = 'clvsr'

if stat == 'beta' and not baselined:
    dat2use = deepcopy(data[contrast])
elif stat == 'beta' and baselined:
    dat2use = deepcopy(data_baselined[contrast])
elif stat == 'tstat' and not baselined:
    dat2use = deepcopy(data_t[contrast])
elif stat == 'tstat' and baselined:
    dat2use = deepcopy(data_baselined_t[contrast])

gave = mne.grand_average(dat2use)

if stat == 'tstat':
    gave.data = toverparam(dat2use)

#if you're using betas, remember that the scale is still in the same scale as the initial data input (e.g. the values of time frequency)
if stat != 'tstat':
    if laplacian:
        vmin = -3e-6
        if baselined:
            vmin = -2e-5
    else:
        vmin = -1e-10
    vmax = np.multiply(vmin,-1)
elif stat == 'tstat':
    vmax = np.multiply(vmin, -1)
    
topovmin = dict()
topovmin['beta'] = -1e-10
topovmin['tstat'] = -3
    
#get the subject wise difference between right and left visual channels
lvsrdata = np.empty(shape = (subs.size, allfreqs.size, alltimes.size))
for i in range(subs.size):
    lvsrdata[i,:,:] = np.subtract(
            np.nanmean(deepcopy(dat2use[i]).pick_channels(visrightchans).data, axis = 0),
            np.nanmean(deepcopy(dat2use[i]).pick_channels(visleftchans).data,  axis = 0)
            )

plot_t = True
if plot_t:
    tfdata = sp.stats.ttest_1samp(lvsrdata, popmean=0, axis = 0)[0]
    plotvmin = -3
else:
    tfdata = np.nanmean(lvsrdata, axis = 0)
    plotvmin = -1e-10


fig = gave.plot_joint(title = contrast, timefreqs     = timefreqs_alpha, vmin = plotvmin, vmax = plotvmin*-1,
                      topomap_args = dict(outlines    = 'head', 
                                          extrapolate = 'head', image_interp='gaussian',
                                          contours    = 0,
                                          vmin        = topovmin[stat],
                                          vmax        = topovmin[stat]*-1))
if 'vs' in contrast:
    axes = fig.axes
    axes[0].clear()
    
    tfrplot = axes[0].imshow(tfdata,  cmap = 'RdBu_r', aspect = 'auto', interpolation = 'gaussian',
                             origin = 'lower', vmin=plotvmin, vmax = plotvmin*-1,
                             extent = ( np.min(alltimes), np.max(alltimes), np.min(allfreqs), np.max(allfreqs)))
    axes[0].set_xlabel('time rel. 2 cue onset')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].vlines([ 0, 0.25, 1.75], lw = 1, linestyles='dashed', color = '#000000', ymin=1, ymax=40)
#%%
#contrast = 'crvsn'


#this will not plot lateralised effects in topographies
#stat = 'beta'
topovmin = dict()
topovmin['beta'] = -1e-10
topovmin['tstat'] = -3
topovmin['laplacian'] = -4e-5

tfvmin = dict()
tfvmin['beta'] = -1e-10
tfvmin['tstat'] = -3

        
for stat in ['beta']:
    tmin, tmax = 0.25, 1.5
    for contrast in ['clvsr', 'clvsn', 'crvsn']:
        if 'vs' in contrast:
            if stat == 'beta':
                dat2use = deepcopy(data)
            elif stat == 'tstat':
                dat2use = deepcopy(data_t)
        else:
            if stat == 'beta':
                dat2use = deepcopy(data_baselined)
            elif stat == 'tstat':
                dat2use = deepcopy(data_baselined_t)
        
        dat2plot = deepcopy(dat2use)
        
        gave = mne.grand_average(dat2plot[contrast])
        
        plot_t = True
        if plot_t:
            gave.data = toverparam(dat2plot[contrast])
        
        gave.plot_joint(title = '%s, stat = %s'%(contrast, stat),
                        timefreqs = timefreqs_alpha,
                        vmin = topovmin['laplacian'], vmax = topovmin['laplacian']*-1,
                        topomap_args = dict(outlines='head',
                                            extrapolate='head',
                                            contours=0,
                                            vmin = topovmin['tstat'],
                                            vmax = topovmin['tstat']*-1))
        
        #run cluster test
        #get data into dataframe first, for the channels we want    
        if 'right' in contrast or contrast == 'crvsn':
            contrachans = visleftchans
            ipsichans   = visrightchans
        else:
            contrachans = visrightchans
            ipsichans = visleftchans
        
        t_cope, clu_cope, clupv_cope, _ = runclustertest_tfr(data = dat2use,
                                                             contrast_name   = contrast,
                                                             contra_channels = contrachans,
                                                             ipsi_channels   = ipsichans,
                                                             tmin = tmin, tmax = tmax, out_type = 'mask',
                                                             n_permutations = 5000)
        masks_cope = np.asarray(clu_cope)[clupv_cope <= 0.05]
        clutimes = deepcopy(data[contrast][0]).crop(tmin = tmin, tmax = tmax).times
        
        
        
        lvsrdata = np.empty(shape = (subs.size, allfreqs.size, alltimes.size))
        for i in range(subs.size):
            tmp = deepcopy(dat2plot[contrast][i])
            tmp_c = deepcopy(tmp).pick_channels(contrachans).data
            tmp_i = deepcopy(tmp).pick_channels(ipsichans).data
            tmp_c = np.nanmean(tmp_c, axis=0); tmp_i = np.nanmean(tmp_i, axis=0)
            tmp_cvsi = np.subtract(tmp_c, tmp_i)
            lvsrdata[i,:,:] = tmp_cvsi
        
        if plot_t:
            lvsr_plotdata = sp.stats.ttest_1samp(lvsrdata, popmean = 0, axis = 0)[0]
    
        
        
        fig = plt.figure(figsize = (8, 4))
        ax = fig.subplots(1)
        tfplot = ax.imshow(lvsr_plotdata,
                           cmap = 'RdBu_r', aspect = 'auto', vmin = vmin, vmax = vmax, interpolation = 'none',
                           origin = 'lower', extent =(alltimes.min(), alltimes.max(), allfreqs.min(), allfreqs.max()))
        ax.vlines([0, 1.75], ymin = allfreqs.min(), ymax = allfreqs.max(),linestyles = 'dashed', color = '#000000', lw = 3)
        fig.colorbar(tfplot, ax = ax)
        for mask in masks_cope:
            bigmask = np.kron(mask, np.ones((10,10)))
            ax.contour(bigmask, colors = '#000000', linewidths = .75, antialiased = False,
                       extent = (clutimes.min(), clutimes.max(), allfreqs.min(), allfreqs.max()))
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time relative to cue onset (s)')
        ax.set_title('%s contra vs ipsi lateralisation in visual channels, stat = %s'%(contrast, stat))
        plt.tight_layout()

#        for mask in masks_cope:
#            itmin = clutimes[np.where(mask==True)[1]].min()
#            itmax = clutimes[np.where(mask==True)[1]].max()
#            fig = mne.viz.plot_tfr_topomap(gave, tmin = itmin, tmax = itmax, fmin = 8, fmax = 12,contours=0, unit='t',
#                                     ch_type = 'eeg', vmin = -3, vmax = 3, res = 300,
#                                     colorbar = True, cmap = 'RdBu_r')
#%%