#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 08:53:31 2019

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
from wmConfidence_funcs import gesd, plot_AR, toverparam, runclustertest_tfr

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

figpath = op.join(wd, 'figures', 'eeg_figs', 'cuelocked', 'tfrglm1')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm
nsubs = subs.size

contrasts = ['pleft_neutral','pleft_cued','pright_neutral','pright_cued',
             'clvsn','crvsn','clvsr',
             'neutral','cued','cuedvsneut',
             'err_pleft_neutral','err_pleft_cued','err_pright_neutral','err_pright_cued',
             'err_clvsn','err_crvsn','err_clvsr',
             'err_neutral','err_cued','err_cuedvsneut']



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
        data[name].append(mne.time_frequency.read_tfrs(            fname = op.join(param['path'], 'glms', 'cue','tfrglm5_nogmean', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_betas-tfr.h5'))[0].crop(tmin=-0.5,tmax=None))
        data_t[name].append(mne.time_frequency.read_tfrs(          fname = op.join(param['path'], 'glms', 'cue','tfrglm5_nogmean', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_tstats-tfr.h5'))[0].crop(tmin=-0.5,tmax=None))
        data_baselined[name].append(mne.time_frequency.read_tfrs(  fname = op.join(param['path'], 'glms', 'cue','tfrglm5_nogmean', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_betas_baselined-tfr.h5'))[0].crop(tmin=-0.5,tmax=None))
        data_baselined_t[name].append(mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'cue','tfrglm5_nogmean', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_tstats_baselined-tfr.h5'))[0].crop(tmin=-0.5,tmax=None))

#%% this is just going to snip out all the stuff prior to cue, so we have half a second before the cue, and the entire post cue period
for i in range(subs.size):
    for contrast in contrasts:
        for dat in [data, data_t, data_baselined, data_baselined_t]:
            dat[contrast][i].crop(tmin = -0.5, tmax = None)
#%%
#align the channel names in all subjects (de-capitalise Z etc as montage needs it to be small)
for i in range(subs.size):
    for contrast in contrasts:
        for dat in [data, data_t, data_baselined, data_baselined_t]:
            chnames = np.asarray(dat[contrast][i].ch_names)
#            newchnames = [x.replace('Z', 'z').replace('FP','Fp') for x in chnames]
            chnamemapping = {}
            for x in range(len(chnames)):
                chnamemapping[chnames[x]] = chnames[x].replace('Z', 'z').replace('FP', 'Fp')
            mne.rename_channels(dat[contrast][i].info, chnamemapping)
            if 'RM' in dat[contrast][i].ch_names:
                dat[contrast][i].drop_channels(['RM'])

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
#this cell is a little dirty and will just plot a few things
stat = 'tstat'
baselined = False
contrast = 'err_pleft_cued'
for contrast in ['err_pleft_cued', 'err_pright_cued', 'err_cued', 'err_clvsn', 'err_crvsn', 'err_clvsr']:
    if contrast in ['err_pleft_cued', 'err_pright_cued', 'err_cued']:
        baselined = True
    else:
        baselined = False
    
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
        vmin = -2e-11
        vmax = np.multiply(vmin,-1)
    elif stat == 'tstat':
        vmin = -5
        vmax = np.multiply(vmin, -1)
        
        
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
    else:
        tfdata = np.nanmean(lvsrdata, axis = 0)
    
    if plot_t:
        vmin = -2
        vmax = np.multiply(vmin, -1)    
    
    fig = gave.plot_joint(title = contrast, timefreqs     = timefreqs_alpha, vmin = vmin, vmax = vmax,
                          topomap_args = dict(outlines    = 'head', 
                                              extrapolate = 'head', image_interp='gaussian',
                                              contours    = 0,
                                              vmin        = vmin,
                                              vmax        = vmax))
    
    if contrast in ['err_pleft_cued','err_clvsn','err_clvsr']:
        plot_lvsr = True
    else:
        plot_lvsr = True
    
    
    if plot_lvsr:
        axes = fig.axes
        axes[0].clear()
        
        tfrplot = axes[0].imshow(tfdata,  cmap = 'RdBu_r', aspect = 'auto', interpolation = 'gaussian',
                                 origin = 'lower', vmin=vmin, vmax = vmax,
                                 extent = ( np.min(alltimes), np.max(alltimes), np.min(allfreqs), np.max(allfreqs)))
        axes[0].set_xlabel('time rel. 2 cue onset')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].vlines([ 0, 0.25, 1.75], lw = 1, linestyles='dashed', color = '#000000', ymin=1, ymax=40)

#%%

#key contrast of interest here is error in cued left vs right
#a positive beta/tstat would mean that lower power is associated with lower error
#for the cued left vs right contrast, if there was item specific selection that related to error
#we would expect to see that the beta/tstat is higher over right visual channels than left visual channels

stat = 'beta'
baselined = True
contrast = 'err_clvsr'
for contrast in ['err_clvsn', 'err_crvsn', 'err_clvsr']:
    for baselined in [True, False]:

        if stat == 'beta' and not baselined:
            dat2use = deepcopy(data[contrast])
        elif stat == 'beta' and baselined:
            dat2use = deepcopy(data_baselined[contrast])
        elif stat == 'tstat' and not baselined:
            dat2use = deepcopy(data_t[contrast])
        elif stat == 'tstat' and baselined:
            dat2use = deepcopy(data_baselined_t[contrast])
        
        if baselined:
            bline = 'baselined'
        else:
            bline = 'not baselined'
        
        gave = mne.grand_average(dat2use)
        
        if stat == 'tstat':
            gave.data = toverparam(dat2use)
        
        #if you're using betas, remember that the scale is still in the same scale as the initial data input (e.g. the values of time frequency)
        if stat != 'tstat':
            vmin = -5e-11
            vmax = np.multiply(vmin,-1)
        elif stat == 'tstat':
            vmin = -3
            vmax = np.multiply(vmin, -1)
            
        
        if contrast in ['err_clvsr', 'err_clvsn']:
            contrachans = visrightchans
            ipsichans   = visleftchans
        elif contrast in ['err_crvsn']:
            contrachans = visleftchans
            ipsichans   = visrightchans
        
        
        #get the subject wise difference between right and left visual channels
        cvsidata = np.empty(shape = (subs.size, allfreqs.size, alltimes.size))
        for i in range(subs.size):
            cvsidata[i,:,:] = np.subtract(
                    np.nanmean(deepcopy(dat2use[i]).pick_channels(contrachans).data, axis = 0),
                    np.nanmean(deepcopy(dat2use[i]).pick_channels(ipsichans).data,  axis = 0)
                    )
               
        
        plot_t = False
        if plot_t:
            tfdata = sp.stats.ttest_1samp(cvsidata, popmean=0, axis = 0)[0]
        else:
            tfdata = np.nanmean(cvsidata, axis = 0)
        
        if plot_t:
            vmin = -2
            vmax = np.multiply(vmin, -1)    
        
        fig = gave.plot_joint(title = '%s %s'%(contrast, bline), timefreqs     = timefreqs_alpha, vmin = vmin, vmax = vmax,
                              topomap_args = dict(outlines    = 'head', 
                                                  extrapolate = 'head', image_interp='gaussian',
                                                  contours    = 0,
                                                  vmin        = vmin,
                                                  vmax        = vmax))
        plot_cvsi = True
        if plot_cvsi:
            axes = fig.axes
            axes[0].clear()
            
            tfrplot = axes[0].imshow(tfdata,  cmap = 'RdBu_r', aspect = 'auto', interpolation = 'gaussian',
                                     origin = 'lower', vmin=vmin, vmax = vmax,
                                     extent = ( np.min(alltimes), np.max(alltimes), np.min(allfreqs), np.max(allfreqs)))
            axes[0].set_xlabel('time rel. 2 cue onset')
            axes[0].set_ylabel('Frequency (Hz)')
            axes[0].vlines([ 0, 0.25, 1.75], lw = 1, linestyles='dashed', color = '#000000', ymin=1, ymax=40)
        
        t_, clu, clupv,_ = runclustertest_tfr(data = data,
                                              contrast_name = contrast,
                                              contra_channels = contrachans,
                                              ipsi_channels   = ipsichans,
                                              tmin = .25, tmax = 1.5,
                                              n_permutations = 5000)
        clutimes = deepcopy(gave).crop(tmin=0, tmax =  1.5).times
        masks = np.asarray(clu)[clupv <= 0.05]
#%%
  
stat = 'tstat'
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
    vmin = -1e-10
    vmax = np.multiply(vmin,-1)
elif stat == 'tstat':
    vmin = -5
    vmax = np.multiply(vmin, -1)
    
    
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
else:
    tfdata = np.nanmean(lvsrdata, axis = 0)

if plot_t:
    vmin = -4
    vmax = np.multiply(vmin, -1)    

fig = gave.plot_joint(title = contrast, timefreqs     = timefreqs_alpha, vmin = vmin, vmax = vmax,
                      topomap_args = dict(outlines    = 'head', 
                                          extrapolate = 'head', image_interp='gaussian',
                                          contours    = 0,
                                          vmin        = vmin,
                                          vmax        = vmax))
axes = fig.axes
axes[0].clear()

tfrplot = axes[0].imshow(tfdata,  cmap = 'RdBu_r', aspect = 'auto', interpolation = 'gaussian',
                         origin = 'lower', vmin=vmin, vmax = vmax,
                         extent = ( np.min(alltimes), np.max(alltimes), np.min(allfreqs), np.max(allfreqs)))
axes[0].set_xlabel('time rel. 2 cue onset')
axes[0].set_ylabel('Frequency (Hz)')
axes[0].vlines([ 0, 0.25, 1.75], lw = 1, linestyles='dashed', color = '#000000', ymin=1, ymax=40)

    
#%%

stat = 'tstat'
baselined = False
contrast = 'cuedvsneut'

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
    vmin = -1e-10
    vmax = np.multiply(vmin,-1)
elif stat == 'tstat':
    vmin = -5
    vmax = np.multiply(vmin, -1)
    
fig = gave.plot_joint(title = contrast, timefreqs     = timefreqs_alpha, vmin = vmin, vmax = vmax,
                      topomap_args = dict(outlines    = 'head', 
                                          extrapolate = 'head', image_interp='gaussian',
                                          contours    = 0,
                                          vmin        = vmin,
                                          vmax        = vmax))
tfdata = np.nanmean(deepcopy(gave).pick_channels(visrightchans+visleftchans).data, 0)

axes = fig.axes
axes[0].clear()
tfrplot = axes[0].imshow(tfdata,  cmap = 'RdBu_r', aspect = 'auto', interpolation = 'gaussian',
                         origin = 'lower', vmin=vmin, vmax = vmax,
                         extent = ( np.min(alltimes), np.max(alltimes), np.min(allfreqs), np.max(allfreqs)))
axes[0].set_xlabel('time rel. 2 cue onset')
axes[0].set_ylabel('Frequency (Hz)')
axes[0].vlines([ 0, 0.25, 1.75], lw = 1, linestyles='dashed', color = '#000000', ymin=1, ymax=40)
#fig.colorbar(tfrplot, ax = axes)
        

#plot motor stuff in cued vs neutral trials
c3dat = np.squeeze(deepcopy(gave).pick_channels(['C3']).data)
c4dat = np.squeeze(deepcopy(gave).pick_channels(['C4']).data)

rvslmot = np.empty(shape = (subs.size, allfreqs.size, alltimes.size))
for i in range(subs.size):
    rvslmot[i,:,:] = np.subtract(np.squeeze(deepcopy(dat2use[i]).pick_channels(['C3']).data),
                                 np.squeeze(deepcopy(dat2use[i]).pick_channels(['C4']).data))

if stat == 'tstat':
    rvslmotdat = sp.stats.ttest_1samp(rvslmot, axis = 0, popmean = 0)[0]
else:
    rvslmotdat =np.nanmean(rvslmot, axis = 0)


fig = plt.figure()
#ax1, ax2 = fig.subplots(1,2)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax1.imshow(c3dat, cmap = 'RdBu_r', aspect = 'auto', interpolation = 'gaussian',
           origin = 'lower', vmin=-4, vmax = 4,
           extent = (np.min(alltimes), np.max(alltimes), np.min(allfreqs), np.max(allfreqs)))

ax2.imshow(c4dat, cmap = 'RdBu_r', aspect = 'auto', interpolation = 'gaussian',
           origin = 'lower', vmin=-4, vmax = 4,
           extent = (np.min(alltimes), np.max(alltimes), np.min(allfreqs), np.max(allfreqs)) )

for ax in [ax1, ax2]:
    ax.set_xlabel('Time relative to cue onset (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.vlines([ 0, 0.25, 1.75], lw = 1, linestyles='dashed', color = '#000000', ymin=1, ymax=40)
ax1.set_title('electrode C3 - contra to right hand')
ax2.set_title('electrode C4 - contra to left hand')
fig.suptitle('cued vs neutral trials')

ax3 = fig.add_subplot(212)

ax3.imshow(rvslmotdat,
           cmap = 'RdBu_r', aspect = 'auto', interpolation = 'gaussian',
           origin = 'lower', vmin=-4, vmax = 4,
           extent = (np.min(alltimes), np.max(alltimes), np.min(allfreqs), np.max(allfreqs)))
ax3.set_xlabel('Time relative to cue onset (s)')
ax3.set_ylabel('Frequency (Hz)')
ax3.vlines([ 0, 0.25, 1.75], lw = 1, linestyles='dashed', color = '#000000', ymin=1, ymax=40)
ax3.set_title('contra vs ipsi to right hand (mouse)')

plt.tight_layout()
