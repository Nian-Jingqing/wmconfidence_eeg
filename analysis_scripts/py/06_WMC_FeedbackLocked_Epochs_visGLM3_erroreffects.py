#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 18:15:03 2019

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
figpath = op.join(wd,'figures', 'eeg_figs', 'feedbacklocked', 'epochs_glm3')


subs = np.array([4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22,24, 25, 26])

contrasts = ['correct', 'incorrect', 'errorcorrect', 'errorincorrect', 'confcorrect', 'confincorrect', 'pside',
             'incorrvscorr', 'errorincorrvscorr', 'confincorrvscorr',
             'grandmean', 'error', 'conf']

laplacian = True
if laplacian:
    lapstr = 'laplacian_'
else:
    lapstr = ''


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
        data[name].append(   mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_' + lapstr + name + '_betas-ave.fif'))[0])        
        data_t[name].append( mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_' + lapstr + name + '_tstats-ave.fif'))[0])        

#%%
gave = mne.grand_average(data['grandmean']); times = gave.times; del(gave)      
#%%
for channel in ['FCz', 'Cz', 'CPz', 'Pz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(corr   = data['errorcorrect'],
                           incorr = data['errorincorrect'],
                           diff   = data['errorincorrvscorr'],
                           all    = data['error']),
            colors = dict(corr   = '#66c2a5',
                          incorr = '#fc8d62',
                          diff   = '#000000',
                          all    = '#377eb8'),
            legend = 'upper right', picks = channel,
            ci = .68, show_sensors = False, title = 'electrode ' + channel,
            truncate_xaxis=False)
gave = mne.grand_average(data['errorincorrvscorr'])

deepcopy(gave).plot_joint(times = np.arange(0.1,0.7,.1),
                title = 'error in incorrect vs correct trials')
        

tmin, tmax = 0, 1
for channel in ['FCz', 'Cz']:
    t_ern, clu_ern, clupv_ern, h0_ern = runclustertest_epochs(data = data,
                                                              contrast_name = 'errorincorrect',
                                                              channels = [channel],
                                                              tmin = tmin,
                                                              tmax = tmax,
                                                              gauss_smoothing = None,
                                                              out_type = 'indices', n_permutations = 'Default')
    clutimes = deepcopy(data['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_ern = np.asarray(clu_ern)[clupv_ern < 0.05]
            
    fig = plt.figure()
    ax = plt.axes()
    mne.viz.plot_compare_evokeds(
        evokeds = dict(corr   = data['errorcorrect'],
                       incorr = data['errorincorrect'],
                       #diff   = data['errorincorrvscorr'],
                       all    = data['error']),
        colors = dict(corr   = '#66c2a5',
                      incorr = '#fc8d62',
                      #diff   = '#000000',
                      all    = '#377eb8'),

        legend = 'upper right', picks = channel,
        ci = .68, show_sensors = False, title = 'electrode ' + channel,
        truncate_xaxis=False, axes = ax,
        vlines = [0, 0.5])
            
    ax.set_title('error regressor at electrode '+ channel)
    ax.set_ylabel('average beta (AU)')
    for mask in range(len(masks_ern)):
        ax.hlines(y = -2.5,
                  xmin = np.min(clutimes[masks_ern[mask][1]]),
                  xmax = np.max(clutimes[masks_ern[mask][1]]),
                  lw=5, color = '#fc8d62', alpha = .5) #plot significance timepoints for difference effect
              
              
#this will plot these significant times as an overlay onto the plot_joint image
#requires finding the right axis within the subplot thats drawn in order to draw them on properly              
fig1 = gave.plot_joint(topomap_args = dict(contours=0, outlines='head', vmin = -4, vmax = 4),
                       times = np.arange(0.1, 0.7, .1))
ax1 = fig1.axes[0]
for mask in range(len(masks_ern)):
    ax1.hlines(y = -3.5,
              xmin = np.min(clutimes[masks_ern[mask][1]]),
              xmax = np.max(clutimes[masks_ern[mask][1]]),
              lw=5, color = '#4292c6', alpha = .5) #plot significance timepoints for difference effect

#%% this is a MUCH better chunk of code to use than the stuff above, which is a bit shitty really and just coarse visualisations


stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
if stat == 'beta':
    dat2use = deepcopy(data)
elif stat == 'tstat':
    dat2use = deepcopy(data_t)

for channel in ['FCz', 'Cz']:
    tmin = 0
    tmax = 1
    
    t_cope, clu_cope, clupv_cope, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'errorincorrvscorr',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None,
                                                            out_type = 'indices', n_permutations = 10000,
                                                            n_jobs = 1) #specify number of cores to run in parallel
    clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_cope = np.asarray(clu_cope)[clupv_cope <= 0.05]
    print('\n%s significant clusters for the difference wave at channel %s\n'%(str(len(masks_cope)), channel))
    
    t_corr, clu_corr, clupv_corr, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'errorcorrect',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None,
                                                            out_type = 'indices', n_permutations = 10000,
                                                            n_jobs = 1) #specify number of cores to run in parallel
#    clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_corr = np.asarray(clu_corr)[clupv_corr <= 0.05]
    print('\n%s significant clusters for error in correct trials wave at channel %s\n'%(str(len(masks_corr)), channel))

    
    
    t_incorr, clu_incorr, clupv_incorr, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'errorincorrect',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None,
                                                            out_type = 'indices', n_permutations = 10000,
                                                            n_jobs = 1) #specify number of cores to run in parallel
#    clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_incorr = np.asarray(clu_incorr)[clupv_incorr <= 0.05]
    print('\n%s significant clusters for error in incorrect trials wave at channel %s\n'%(str(len(masks_incorr)), channel))

    
    
    # now we will plot the mean with std error ribbons around the signal shape, ideally using seaborn
    
    #first lets get all the single subject data into one dataframe:
    plottimes   = deepcopy(dat2use['grandmean'][0]).crop(tmax = tmax).times #this includes the baseline period for plotting
    plotdat_diff = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_diff[i,:] = np.squeeze(deepcopy(dat2use['errorincorrvscorr'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_diff = np.nanmean(plotdat_diff, axis = 0)
    plotdatsem_diff  = sp.stats.sem(plotdat_diff, axis = 0)
    
    plotdat_incorr = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_incorr[i,:] = np.squeeze(deepcopy(dat2use['errorincorrect'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_incorr = np.nanmean(plotdat_incorr, axis = 0)
    plotdatsem_incorr  = sp.stats.sem(plotdat_incorr, axis = 0)
    
    plotdat_corr = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_corr[i,:] = np.squeeze(deepcopy(dat2use['errorcorrect'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_corr = np.nanmean(plotdat_corr, axis = 0)
    plotdatsem_corr  = sp.stats.sem(plotdat_corr, axis = 0)
    
    
    #plotdatmean = np.multiply(plotdatmean, 10e5)
    fig = plt.figure(figsize = (12, 6))
    ax  = fig.add_subplot(111)
    ax.plot(plottimes, plotdatmean_diff, color = '#000000', lw = 1.5, label = 'difference')
    ax.fill_between(plottimes, plotdatmean_diff-plotdatsem_diff, plotdatmean_diff+plotdatsem_diff, alpha = .3, color = '#636363')
    
    ax.plot(plottimes, plotdatmean_incorr, color = '#e41a1c', lw = 1.5, label = 'incorrect')
    ax.fill_between(plottimes, plotdatmean_incorr - plotdatsem_incorr, plotdatmean_incorr + plotdatsem_incorr, alpha = .3, color = '#e41a1c')
    
    ax.plot(plottimes, plotdatmean_corr, color = '#4daf4a', lw = 1.5, label = 'correct')
    ax.fill_between(plottimes, plotdatmean_corr - plotdatsem_corr, plotdatmean_corr + plotdatsem_corr, alpha = .3, color = '#4daf4a')
    
    
    ax.hlines([0], lw = 1, xmin = plottimes.min(), xmax = plottimes.max(), linestyles = 'dashed')
    ax.vlines([0], lw = 1, ymin = plotdatmean_incorr.min(), ymax = plotdatmean_incorr.max(), linestyles = 'dashed')
    ax.set_title('error regressor at channel '+channel)
    ax.set_ylabel('beta (AU)')
    ax.set_xlabel('Time relative to feedback onset (s)')
    ax.legend(loc = 'upper left')
    for mask in masks_cope:
        ax.hlines(y = -5e-4,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#636363', alpha = .5) #plot significance timepoints for difference effect
                  
    for mask in masks_incorr:
        ax.hlines(y = -3.2e-4,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#e41a1c', alpha = .5) #plot significance timepoints for difference effect
    for mask in masks_corr:
        ax.hlines(y = -3.4e-4,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#4daf4a', alpha = .5) #plot significance timepoints for difference effect                  
#    fig.savefig(fname = op.join(figpath, 'error_x_diffwave_20subs_channel_'+channel+'_'+ stat +'.eps'), format = 'eps', dpi = 300)
#    fig.savefig(fname = op.join(figpath, 'error_x_diffwave_20subs_channel_'+channel+'_'+ stat +'.pdf'), format = 'pdf', dpi = 300)
#    
#    plt.close()
#%%    
plotvmin = dict(); plotvmin['beta'] = -3e-6; plotvmin['tstat'] = -6
gave = mne.grand_average(dat2use['errorcorrect'])
if stat == 'tstat':
    gave.data = toverparam(dat2use['errorcorrect'])
gave.plot_joint(topomap_args = dict(outlines='head', extrapolate = 'head'), times = np.arange(.1, .7, .1))                  
                  