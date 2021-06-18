#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:19:22 2021

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
# from wmConfidence_funcs import get_subject_info_wmConfidence
# from wmConfidence_funcs import gesd, plot_AR, toverparam, smooth, runclustertest_epochs
#sys.path.insert(0, 'C:\\Users\\sammi\\Desktop\\Experiments\\DPhil\\wmConfidence\\analysis_scripts')#because working from laptop to make this script
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, smooth, runclustertest_epochs

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
#wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd

os.chdir(wd)
figpath = op.join(wd,'figures', 'eeg_figs', 'fblocked', 'suppmodel2')


subs = np.array([4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22,24, 25, 26])

contrasts = ['neutralcorrect', 'cuedcorrect', 'neutralincorrect', 'cuedincorrect',
             'pside', 'cued', 'neutral', 'correct', 'incorrect', 'grandmean']

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
        data[name].append(   mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'suppmodel2', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'  + name + '_betas-ave.fif'))[0])        
        data_t[name].append( mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'suppmodel2', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'  + name + '_tstats-ave.fif'))[0])        

#%%
gave = mne.grand_average(data['grandmean']); times = gave.times;
#%%

#firstly, just look at the scalp topographies (joint plot) for the evoked response in the relevant conditions
mne.grand_average(deepcopy(data['cued'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'cued trials')
mne.grand_average(deepcopy(data['neutral'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'neutral trials')
#%%
 
# now for channels FCz, Fz, Cz, we want to plot the ERPs in the different trial types

for channel in ['FCz', 'Fz', 'Cz']:
    fig = plt.figure()
    ax = fig.subplots(1)
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    cued    = data['cuedcorrect'],
                    neutral  = data['neutralcorrect']),
            colors = dict(
                    cued = '#3182bd',
                    neutral = '#636363'),
            axes = ax, truncate_xaxis = False, vlines = [0, 0.5],
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'cued and neutral correct trials: electrode '+channel)
    ax.set_title('correct feedback ERP at electrode '+channel)
    ax.set_ylabel('beta (AU)')
    fig.savefig(fname = op.join(figpath, 'feedbackERP_CorrectTrials_byCue_electrode_%s_clustertimes_betas.eps'%(channel)), format = 'eps', dpi = 300 )
    fig.savefig(fname = op.join(figpath, 'feedbackERP_CorrectTrials_byCue_electrode_%s_clustertimes_betas.pdf'%(channel)), format = 'pdf', dpi = 300 )
    
for channel in ['FCz', 'Fz', 'Cz']:
    fig = plt.figure()
    ax = fig.subplots(1)
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    cued    = data['cuedincorrect'],
                    neutral  = data['neutralincorrect']),
            colors = dict(
                    cued = '#3182bd',
                    neutral = '#636363'),
            axes = ax, truncate_xaxis = False, vlines = [0, 0.5],
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'cued and neutral incorrect trials: electrode '+channel)
    ax.set_title('incorrect feedback ERP at electrode '+channel)
    ax.set_ylabel('beta (AU)')
    fig.savefig(fname = op.join(figpath, 'feedbackERP_IncorrectTrials_byCue_electrode_%s_clustertimes_betas.eps'%(channel)), format = 'eps', dpi = 300 )
    fig.savefig(fname = op.join(figpath, 'feedbackERP_IncorrectTrials_byCue_electrode_%s_clustertimes_betas.pdf'%(channel)), format = 'pdf', dpi = 300 )
    

for channel in ['FCz', 'Fz', 'Cz']:
    fig = plt.figure()
    ax = fig.subplots(1)
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    cued    = data['cued'],
                    neutral  = data['neutral']),
            colors = dict(
                    cued = '#3182bd',
                    neutral = '#636363'),
            axes = ax, truncate_xaxis = False, vlines = [0, 0.5],
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'cued and neutral trials: electrode '+channel)
    ax.set_title(' feedback ERP at electrode '+channel)
    ax.set_ylabel('beta (AU)')
    fig.savefig(fname = op.join(figpath, 'feedbackERP_byCue_electrode_%s_clustertimes_betas.eps'%(channel)), format = 'eps', dpi = 300 )
    fig.savefig(fname = op.join(figpath, 'feedbackERP_byCue_electrode_%s_clustertimes_betas.pdf'%(channel)), format = 'pdf', dpi = 300 )
    



