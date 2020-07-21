#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:16:47 2020

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

np.random.seed(seed = 10)

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, runclustertest_epochs

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

figpath = op.join(wd, 'figures', 'eeg_figs', 'feedbacklocked')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm
nsubs = subs.size

contrasts = ['grandmean', 'trialupdate']

glmnum = 7

data = dict()
data_t = dict()
for i in contrasts:
    data[i] = []
    data_t[i] = []
for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    for name in contrasts:
        data[name].append(   mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm'+str(glmnum), 'wmc_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_betas-ave.fif'))[0])
        data_t[name].append( mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm'+str(glmnum), 'wmc_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_tstats-ave.fif'))[0])

gave = mne.grand_average(data['grandmean']); times = gave.times; del(gave)      
#%%

contrast = 'trialupdate'

#   this is just a quick and dirty visualisation cos easy

gave = mne.grand_average(data[contrast])
gave.plot_joint(times = np.arange(0.1, 0.6, 0.1))
#remember: trlupdate   =  np.subtract(nxttrlconf, conf) #positive values mean they became more confident, negative values they became less confident
#%%

stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
if stat == 'beta':
    dat2use = deepcopy(data)
elif stat == 'tstat':
    dat2use = deepcopy(data_t)

contrast = 'trialupdate'

tmin = -0.5
tmax = 1.5
for channel in ['FCz', 'Cz', 'CPz', 'Pz', 'POz']:
#for channel in ['FCz', 'Cz']:
    #first lets get all the single subject data into one dataframe:
    plottimes = deepcopy(dat2use[contrast][0]).crop(tmax=tmax).times
    plotdat   = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat[i,:] = np.squeeze(deepcopy(dat2use[contrast][i]).pick_channels([channel]).crop(tmax = tmax).data)
    
    plotdat_mean = np.nanmean(plotdat, axis = 0)
    plotdat_sem  = sp.stats.sem(plotdat, axis = 0)
    
    fig = plt.figure(figsize = (12, 6))
    ax  = fig.add_subplot(111)
    ax.plot(plottimes, plotdat_mean, color = '#e41a1c', lw = 1.5, label = 'trialupdate')
    ax.fill_between(plottimes, plotdat_mean - plotdat_sem, plotdat_mean + plotdat_sem, alpha = .3, color = '#e41a1c')
    
    
    ax.hlines([0], lw = 1, xmin = plottimes.min(), xmax = plottimes.max(), linestyles = 'dashed')
    ax.vlines([0], lw = 1, ymin = plotdat_mean.min(), ymax = plotdat_mean.max(), linestyles = 'dashed')
    ax.set_title('trial type  at channel '+channel)
    ax.set_ylabel('beta (AU)')
    ax.set_xlabel('Time relative to feedback onset (s)')
    ax.legend(loc = 'upper left')



        