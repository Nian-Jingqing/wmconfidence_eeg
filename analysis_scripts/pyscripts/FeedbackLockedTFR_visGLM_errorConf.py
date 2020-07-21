#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:46:21 2020

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

figpath = op.join(wd, 'figures', 'eeg_figs', 'fblocked', 'tfrglm1')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm
nsubs = subs.size



contrasts = ['grandmean', 'pside', 'error', 'conf',
             'correct',      'incorrect',      'incorrvscorr',
             'errorcorrect', 'errorincorrect', 'errorincorrvscorr',
             'confcorrect',  'confincorrect',  'confincorrvscorr']

laplacian = True
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
#        data[name].append(mne.time_frequency.read_tfrs(            fname = op.join(param['path'], 'glms', 'feedback','tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ lapstr + name + '_betas-tfr.h5'))[0])
        data_t[name].append(mne.time_frequency.read_tfrs(          fname = op.join(param['path'], 'glms', 'feedback','tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ lapstr + name + '_tstats-tfr.h5'))[0])
#        data_baselined[name].append(mne.time_frequency.read_tfrs(  fname = op.join(param['path'], 'glms', 'feedback','tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ lapstr + name + '_betas_baselined-tfr.h5'))[0])
        data_baselined_t[name].append(mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback','tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ lapstr + name + '_tstats_baselined-tfr.h5'))[0])


#%%
alltimes = mne.grand_average(data_t['grandmean']).times
allfreqs = mne.grand_average(data_t['grandmean']).freqs

timefreqs_alpha = {(.4, 10):(.4, 4),
                   (.6, 10):(.4, 4),
                   (.8, 10):(.4, 4),
                   (1., 10):(.4, 4),
                   (1.2, 10):(.4, 4)}
timefreqs_theta = {(.4, 6):(.4, 4),
                   (.6, 6):(.4, 4),
                   (.8, 6):(.4, 4),
                   (1., 6):(.4, 4),
                   (1.2, 6):(.4, 4)}

frontal_chans = ['Fz', 'AFz', 'F1', 'F2', 'FCz']
#%%

# we want to just look at the behavioural regressors here, so look at error and confidence
# and see if the error related frontal theta effect varies as a function of either of these aspects of behaviour
#or more broadly, as in the feedback ERP, if there are modulations of the feedback induced response by error or confidence
stat = 'tstat'
plot_t = True

vmin = dict()
vmin['beta'] = -5e-10
vmin['tstat'] = -3

for contrast in ['error', 'conf']:#, 'errorincorrvscorr', 'confincorrvscorr']:
    if 'vs' in contrast:
        baselined = False
    else:
        baselined = True
    
    if stat =='tstat' and baselined:
        dat2use = deepcopy(data_baselined_t)
    elif stat == 'tstat' and not baselined:
        dat2use = deepcopy(data_t)
        
    gave = mne.grand_average(dat2use[contrast])
    if plot_t:
        gave.data = toverparam(dat2use[contrast])
    
#    gave.plot_joint(timefreqs = timefreqs_alpha, topomap_args = dict(contours = 0, vmin = -2.8, vmax = 2.8), vmin = -2.8, vmax = 2.8, title = contrast)
    
    gave.plot(picks = 'FCz', title = contrast+' at FCz, stat ='+stat, vmin = -2.8, vmax = 2.8)


#%%

contrast = 'errorcorrect'
channel = ['FCz']
stat = 'tstat'
tmin = 0
tmax = 1.5
#for contrast in ['errorcorrect', 'errorincorrect', 'errorincorrvscorr']:
for contrast in ['conf', 'confcorrect', 'confincorrect', 'confincorrvscorr']:
    if stat == 'beta':
        if 'vs' in contrast:
            dat2use = deepcopy(data)
        else:
            dat2use = deepcopy(data_baselined)
    elif stat == 'tstat':
        if 'vs' in contrast:
            dat2use = deepcopy(data_t)
        else:
            dat2use = deepcopy(data_baselined_t)
    
    
    t_cope, clu_cope, clupv_cope, _ = runclustertest_tfr(data = dat2use,
                                                         contrast_name = contrast,
                                                         channels = ['FCz'],
                                                         contra_channels = None,
                                                         ipsi_channels   = None,
                                                         tmin = tmin, tmax = tmax, out_type = 'mask',
                                                         n_permutations = 5000)
    masks_cope = np.asarray(clu_cope)[clupv_cope <= 0.05]
    clutimes   = deepcopy(dat2use[contrast][0]).crop(tmin = tmin, tmax = tmax).times
    
    
    plotdata = mne.grand_average(deepcopy(dat2use[contrast]));
    if stat =='tstat':
        plot_t = True
        vmin = -3
        vmax = 3
    else:
        plot_t = False
        vmin = -5e-11
        vmax = 5e-11
    if plot_t:
        plotdata.data = toverparam(deepcopy(dat2use[contrast]))
    
    tfdata = np.squeeze(plotdata.pick_channels(['FCz']).data)
    
    fig = plt.figure(figsize = (8,4))
    ax = fig.subplots(1)
    tfplot = ax.imshow(tfdata,
                       cmap = 'RdBu_r', aspect = 'auto', vmin = vmin, vmax = vmax, interpolation = 'none',
                       origin = 'lower', extent = (alltimes.min(), alltimes.max(), allfreqs.min(), allfreqs.max()))
    ax.vlines([0, 0.5], ymin = allfreqs.min(), ymax = allfreqs.max(), linestyles = 'dashed', color = '#000000', lw = 3)
    fig.colorbar(tfplot, ax = ax)
    for mask in masks_cope:
        bigmask = np.kron(mask, np.ones((10,10)))
        ax.contour(bigmask, color = '#000000', linewidths = .75, antialiased = False,
                   extent = (clutimes.min(), clutimes.max(), allfreqs.min(), allfreqs.max()))
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time relative to feedback onset (s)')
    ax.set_title('feedback induced response for %s, stat = %s'%(contrast, stat))
    plt.tight_layout()
#fig.savefig(fname = op.join(figpath,'feedbackTFR_incorrvsCorr_channel_%s.eps'%(str(np.squeeze(channel)))), format = 'eps', dpi = 300)
#fig.savefig(fname = op.join(figpath,'feedbackTFR_incorrvsCorr_channel_%s.pdf'%(str(np.squeeze(channel)))), format = 'pdf', dpi = 300)

jplot = mne.grand_average(dat2use[contrast]);
if plot_t:
    jplot.data = toverparam(dat2use[contrast])
jplot.plot_joint(timefreqs = timefreqs_theta, topomap_args = dict(contours=0, vmin = vmin, vmax = vmax))

for mask in masks_cope:
    itmin = clutimes[np.where(mask==True)[1]].min()
    itmax = clutimes[np.where(mask==True)[1]].max()
    fig = mne.viz.plot_tfr_topomap(jplot, tmin = itmin, tmax = itmax, fmin = 4, fmax = 8,contours=0, unit='t',
                              vmin = -3, vmax = 3, res = 300,
                             colorbar = True, cmap = 'RdBu_r')
