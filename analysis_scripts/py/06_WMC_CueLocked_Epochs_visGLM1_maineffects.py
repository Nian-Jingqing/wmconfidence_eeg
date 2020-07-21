#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:09:27 2019

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
from scipy import stats

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, smooth, runclustertest_epochs

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)
figpath = op.join(wd,'figures', 'eeg_figs', 'cuelocked', 'epochsglm1')


subs = np.array([4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22,24, 25, 26])

contrasts = ['pleft_cued', 'pleft_neutral', 'pright_cued', 'pright_neutral',
             'crvsn', 'clvsn','clvsr',
             'neutral', 'cued', 'cuedvsneut']
laplacian = False

data = dict()
data_t = dict()
for i in contrasts:
    data[i] = []
    data_t[i] = []

for i in subs:
    for i in subs:
        print('\n\ngetting subject ' + str(i) +'\n\n')
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub) #_baselined
        if laplacian:
            lapstr = 'laplacian_'
        else:
            lapstr = ''
        
        for name in contrasts:
            data[name].append( mne.read_evokeds(  fname = op.join(param['path'], 'glms', 'cue', 'epochsglm1', 'wmc_' + param['subid'] + '_cuelocked_tl_' + lapstr + name + '_betas-ave.fif'))[0])        
            data_t[name].append( mne.read_evokeds(fname = op.join(param['path'], 'glms', 'cue', 'epochsglm1', 'wmc_' + param['subid'] + '_cuelocked_tl_' + lapstr + name + '_tstats-ave.fif'))[0])        
#%%    




#%%     
    for i in range(subs.size):
        for contrast in contrasts:
            for dat in [data, data_t]:
                chnames = np.asarray(dat[contrast][i].ch_names)
    #            newchnames = [x.replace('Z', 'z').replace('FP','Fp') for x in chnames]
                chnamemapping = {}
                for x in range(len(chnames)):
                    chnamemapping[chnames[x]] = chnames[x].replace('z', 'Z').replace('Fp', 'FP')
                mne.rename_channels(dat[contrast][i].info, chnamemapping)
                if 'RM' in dat[contrast][i].ch_names:
                    dat[contrast][i].drop_channels(['RM'])
    
            
    stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
    if stat == 'beta':
        dat2use = deepcopy(data)
    elif stat == 'tstat':
        dat2use = deepcopy(data_t)
    
    contrast = 'clvsr' #choose the cope you want to look at #print(contrasts) shows what there is
    channels  = ['PO7', 'O1'] #choose the channel you want to run the cluster permutation tests on
    
    for contrast in ['clvsr']:#['pright_cued', 'pleft_cued', 'clvsr']:
        gave = mne.grand_average(dat2use[contrast])
        
        gave.plot_joint(times = np.arange(0.05,0.7, .05), topomap_args = dict(outlines='skirt', contours = 0), title = contrast+laplacian)
#%%
stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
if stat == 'beta':
    dat2use = deepcopy(data)
elif stat == 'tstat':
    dat2use = deepcopy(data_t)

#first lets get all the single subject data into one dataframe:
plottimes   = deepcopy(dat2use['clvsn'][0]).times #this includes the baseline period for plotting
#contrast = 'pright_cued'
#for contrast in ['pleft_cued', 'pright_cued', 'cued', 'clvsr']:
for contrast in ['clvsr']:
    if 'left' in contrast:
        contrachans = ['PO8']#, 'O2']
        ipsichans   = ['PO7']#, 'O1']
    elif 'right' in contrast:
        contrachans   = ['PO7']#, 'O1']
        ipsichans     = ['PO8']#, 'O2']
    elif contrast == 'clvsr':
        contrachans = ['PO8']
        ipsichans   = ['PO7']
    
    plotdat_contra = np.empty(shape = (subs.size, plottimes.size))
    plotdat_ipsi   = np.empty(shape = (subs.size, plottimes.size))
    
    for i in range(subs.size):
        plotdat_contra[i,:] = np.nanmean(deepcopy(dat2use[contrast][i]).pick_channels(contrachans).data, axis=0)
        plotdat_ipsi[i,:] = np.nanmean(deepcopy(dat2use[contrast][i]).pick_channels(ipsichans).data, axis=0)
        
    plotdat_cvsi = np.subtract(plotdat_contra, plotdat_ipsi)
    
    plotdatmean_contra = np.nanmean(plotdat_contra, axis = 0)
    plotdatmean_ipsi   = np.nanmean(plotdat_ipsi,   axis = 0)
    plotdatmean_cvsi   = np.nanmean(plotdat_cvsi,   axis = 0)
    
    plotdatsem_contra  = sp.stats.sem(plotdat_contra, axis = 0)
    plotdatsem_ipsi    = sp.stats.sem(plotdat_ipsi,   axis = 0)
    plotdatsem_cvsi    = sp.stats.sem(plotdat_cvsi,   axis = 0)
    
    fig = plt.figure(figsize = (12, 6))
    ax  = fig.add_subplot(111)
    ax.plot(plottimes, plotdatmean_contra, color = '#fc8d59', lw = 1.5, label = 'contra visual chans')
    ax.fill_between(plottimes, plotdatmean_contra-plotdatsem_contra, plotdatmean_contra+plotdatsem_contra, alpha = .3, color = '#fc8d59')
                    
    ax.plot(plottimes, plotdatmean_ipsi, color = '#91bfdb', lw = 1.5, label = 'ipsi visual chans')
    ax.fill_between(plottimes, plotdatmean_ipsi-plotdatsem_ipsi, plotdatmean_ipsi+plotdatsem_ipsi, alpha = .3, color = '#91bfdb')
                    
    ax.plot(plottimes, plotdatmean_cvsi, color = '#636363', lw = 1.5, label = 'cvsi visual chans')
    ax.fill_between(plottimes, plotdatmean_cvsi-plotdatsem_cvsi, plotdatmean_cvsi+plotdatsem_cvsi, alpha = .3, color = '#636363')
    
    ax.hlines([0], lw = 1, xmin = plottimes.min(), xmax = plottimes.max(), linestyles = 'dashed')
    ax.vlines([0], lw = 1, ymin = plotdatmean_contra.min(), ymax = plotdatmean_contra.max(), linestyles = 'dashed')
    ax.set_title('grand average ERP: contra and ipsi visual channels for  ' + contrast)
    ax.set_ylabel('beta (AU)')
    ax.set_xlabel('Time relative to cue onset (s)')
    ax.legend(loc = 'upper left')
    
    gave = mne.grand_average(dat2use[contrast])
    gave.plot_joint(title = contrast,times=np.arange(.05,.7,.05), topomap_args = dict(outlines='skirt', contours = 0))
    
#%%
    
stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
if stat == 'beta':
    dat2use = deepcopy(data)
elif stat == 'tstat':
    dat2use = deepcopy(data_t)

#first lets get all the single subject data into one dataframe:
plottimes   = deepcopy(dat2use['clvsn'][0]).times #this includes the baseline period for plotting
contrast   = 'pright_cued' 
contrast2  = 'pleft_cued'
contrast3  = 'pleft_neutral'

if 'left' in contrast:
    contrachans = ['PO8', 'O2']
    ipsichans   = ['PO7', 'O1']
elif 'right' in contrast:
    contrachans   = ['PO7', 'O1']
    ipsichans     = ['PO8', 'O2']
elif contrast == 'clvsr':
    contrachans = ['PO8']
    ipsichans   = ['PO7']

plotdat_contra = np.empty(shape = (subs.size, plottimes.size))
plotdat_ipsi   = np.empty(shape = (subs.size, plottimes.size))

for i in range(subs.size):
    plotdat_contra[i,:] = np.nanmean(deepcopy(dat2use[contrast][i]).pick_channels(contrachans).data, axis=0)
    plotdat_ipsi[i,:] = np.nanmean(deepcopy(dat2use[contrast2][i]).pick_channels(contrachans).data, axis=0)
    
plotdat_cvsi = np.subtract(plotdat_contra, plotdat_ipsi)

plotdatmean_contra = np.nanmean(plotdat_contra, axis = 0)
plotdatmean_ipsi   = np.nanmean(plotdat_ipsi,   axis = 0)
plotdatmean_cvsi   = np.nanmean(plotdat_cvsi,   axis = 0)

plotdatsem_contra  = sp.stats.sem(plotdat_contra, axis = 0)
plotdatsem_ipsi    = sp.stats.sem(plotdat_ipsi,   axis = 0)
plotdatsem_cvsi    = sp.stats.sem(plotdat_cvsi,   axis = 0)

fig = plt.figure(figsize = (12, 6))
ax  = fig.add_subplot(111)
ax.plot(plottimes, plotdatmean_contra, color = '#fc8d59', lw = 1.5, label = 'contra visual target')
ax.fill_between(plottimes, plotdatmean_contra-plotdatsem_contra, plotdatmean_contra+plotdatsem_contra, alpha = .3, color = '#fc8d59')
                
ax.plot(plottimes, plotdatmean_ipsi, color = '#91bfdb', lw = 1.5, label = 'neutral')
ax.fill_between(plottimes, plotdatmean_ipsi-plotdatsem_ipsi, plotdatmean_ipsi+plotdatsem_ipsi, alpha = .3, color = '#91bfdb')
                
ax.plot(plottimes, plotdatmean_cvsi, color = '#636363', lw = 1.5, label = 'difference')
ax.fill_between(plottimes, plotdatmean_cvsi-plotdatsem_cvsi, plotdatmean_cvsi+plotdatsem_cvsi, alpha = .3, color = '#636363')

ax.hlines([0], lw = 1, xmin = plottimes.min(), xmax = plottimes.max(), linestyles = 'dashed')
ax.vlines([0], lw = 1, ymin = plotdatmean_contra.min(), ymax = plotdatmean_contra.max(), linestyles = 'dashed')
ax.set_title('grand average ERP: contra and ipsi visual target for  ' + contrachans[0] +' and ' +contrachans[1])
ax.set_ylabel('beta (AU)')
ax.set_xlabel('Time relative to cue onset (s)')
ax.legend(loc = 'upper left')

gave = mne.grand_average(dat2use[contrast])
gave.plot_joint(title = contrast,times=np.arange(.05,.7,.05), picks = 'eeg', topomap_args = dict(outlines='head', contours = 0))
    