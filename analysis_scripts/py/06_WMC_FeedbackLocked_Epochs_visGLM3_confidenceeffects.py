#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 23:46:19 2019

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
        data[name].append( mne.read_evokeds( fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_' + name + '_betas-ave.fif'))[0])        
        data_t[name].append( mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_' + name + '_tstats-ave.fif'))[0])        

for name in contrasts:
    for i in range(subs.size):
        data[name][i].drop_channels(['RM'])
        data_t[name][i].drop_channels(['RM'])
#%%
gave = mne.grand_average(data['grandmean']); times = gave.times; del(gave)      
#%%
for channel in ['FCz', 'Cz', 'CPz', 'Pz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(corr   = data['confcorrect'],
                           incorr = data['confincorrect'],
                           diff   = data['confincorrvscorr'],
                           all    = data['conf']),
            colors = dict(corr   = '#66c2a5',
                          incorr = '#fc8d62',
                          diff   = '#000000',
                          all    = '#377eb8'),
            show_legend = 'upper right', picks = channel,
            ci = .68, show_sensors = False, title = 'electrode ' + channel,
            truncate_xaxis=False)
gave = mne.grand_average(data['confincorrvscorr'])

deepcopy(gave).plot_joint(times = np.arange(0.1,0.7,.1),
                title = 'confidence regressor: incorrect vs correct trials',
                picks='eeg')
        

tmin, tmax = 0, .75
for channel in ['FCz', 'Cz', 'CPz', 'Pz']:
    t_ern, clu_ern, clupv_ern, h0_ern = runclustertest_epochs(data = data,
                                                              contrast_name = 'confincorrvscorr',
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
        evokeds = dict(corr   = data['confcorrect'],
                       incorr = data['confincorrect'],
                       diff   = data['confincorrvscorr'],
                       all    = data['conf']),
        colors = dict(corr   = '#66c2a5',
                      incorr = '#fc8d62',
                      diff   = '#000000',
                      all    = '#377eb8'),

        show_legend = 'upper right', picks = channel,
        ci = .68, show_sensors = False, title = 'electrode ' + channel,
        truncate_xaxis=False, axes = ax,
        vlines = [0, 0.5])
            
    ax.set_title('confidence regressor at electrode '+ channel)
    ax.set_ylabel('average beta (AU)')
    for mask in range(len(masks_ern)):
        ax.hlines(y = -2.5,
                  xmin = np.min(clutimes[masks_ern[mask][1]]),
                  xmax = np.max(clutimes[masks_ern[mask][1]]),
                  lw=5, color = '#000000', alpha = .5) #plot significance timepoints for difference effect
              
              
#this will plot these significant times as an overlay onto the plot_joint image
#requires finding the right axis within the subplot thats drawn in order to draw them on properly              
fig1 = gave.plot_joint(picks = 'eeg', topomap_args = dict(contours=0, outlines='head', vmin = -4, vmax = 4),
                       times = np.arange(0.1, 0.7, .1))
ax1 = fig1.axes[0]
for mask in range(len(masks_ern)):
    ax1.hlines(y = -3.5,
              xmin = np.min(clutimes[masks_ern[mask][1]]),
              xmax = np.max(clutimes[masks_ern[mask][1]]),
              lw=5, color = '#000000', alpha = .5) #plot significance timepoints for difference effect

#%% this is a MUCH better chunk of code to use than the stuff above, which is a bit shitty really and just coarse visualisations


stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
if stat == 'beta':
    dat2use = deepcopy(data)
elif stat == 'tstat':
    dat2use = deepcopy(data_t)

contrast = 'confincorrvscorr' #choose the cope you want to look at #print(contrasts) shows what there is
channel  = 'Cz' #choose the channel you want to run the cluster permutation tests on
#for channel in ['FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']:
#for channel in ['FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']:
for channel in ['FCz', 'Cz']:
    tmin = 0
    tmax = 1
    
    t_cope, clu_cope, clupv_cope, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = contrast,
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None,
                                                            out_type = 'indices', n_permutations = 10000,
                                                            n_jobs = 1) #specify number of cores to run in parallel
    clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_cope = np.asarray(clu_cope)[clupv_cope <= 0.05]
    print('%s significant clusters at channel %s for the difference wave'%(str(len(masks_cope)), channel))
    
    
    t_corr, clu_corr, clupv_corr, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'confcorrect',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None,
                                                            out_type = 'indices', n_permutations = 10000,
                                                            n_jobs = 1) #specify number of cores to run in parallel
#    clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_corr = np.asarray(clu_corr)[clupv_corr <= 0.05]
    
    
    
    t_incorr, clu_incorr, clupv_incorr, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'confincorrect',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None,
                                                            out_type = 'indices', n_permutations = 10000,
                                                            n_jobs = 1) #specify number of cores to run in parallel
#    clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_incorr = np.asarray(clu_incorr)[clupv_incorr <= 0.05]
    
    
    # now we will plot the mean with std error ribbons around the signal shape, ideally using seaborn
    
    #first lets get all the single subject data into one dataframe:
    plottimes   = deepcopy(dat2use['grandmean'][0]).crop(tmax = tmax).times #this includes the baseline period for plotting
    plotdat_diff = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_diff[i,:] = np.squeeze(deepcopy(dat2use[contrast][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_diff = np.nanmean(plotdat_diff, axis = 0)
    plotdatsem_diff  = sp.stats.sem(plotdat_diff, axis = 0)
    
    plotdat_incorr = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_incorr[i,:] = np.squeeze(deepcopy(dat2use['confincorrect'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_incorr = np.nanmean(plotdat_incorr, axis = 0)
    plotdatsem_incorr  = sp.stats.sem(plotdat_incorr, axis = 0)
    
    plotdat_corr = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_corr[i,:] = np.squeeze(deepcopy(dat2use['confcorrect'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
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
    ax.set_title('confidence regressor at channel '+channel)
    ax.set_ylabel('beta (AU)')
    ax.set_xlabel('Time relative to feedback onset (s)')
    ax.legend(loc = 'upper left')
    for mask in masks_cope:
        ax.hlines(y = -3e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#636363', alpha = .5) #plot significance timepoints for difference effect
                  
    for mask in masks_incorr:
        ax.hlines(y = -3.2e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#e41a1c', alpha = .5) #plot significance timepoints for difference effect
    for mask in masks_corr:
        ax.hlines(y = -3.4e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#4daf4a', alpha = .5) #plot significance timepoints for difference effect
        
#    fig.savefig(fname = op.join(figpath, 'confidence_x_diffwave_20subs_channel_'+channel+'_'+ stat +'.eps'), format = 'eps', dpi = 300)
#    fig.savefig(fname = op.join(figpath, 'confidence_x_diffwave_20subs_channel_'+channel+'_'+ stat +'.pdf'), format = 'pdf', dpi = 300)
##    
#    plt.close()
#%%
#%% this is just to see if the lack of statistical difference between correct and incorrect trials
#   is because of reduced trial numbers in some subjects
#   it looks like this isn't really the case    
    
    
    
incorrnaves = [x.nave for x in data['confincorrect']]
lowtrls = np.less(incorrnaves,40)
hightrls = np.equal(lowtrls, False) #subjects where there are more than 40 trials that were incorrect across the full task
hightrls = np.squeeze(np.where(np.greater(incorrnaves, 40)))

stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
if stat == 'beta':
    dat2use = deepcopy(data)
    for contrast in contrasts:
        dat2use[contrast] = [dat2use[contrast][i] for i in hightrls]
elif stat == 'tstat':
    dat2use = deepcopy(data_t)
    for contrast in contrasts:
        dat2use[contrast] = [dat2use[contrast][i] for i in hightrls]

contrast = 'confincorrvscorr' #choose the cope you want to look at #print(contrasts) shows what there is
channel  = 'Cz' #choose the channel you want to run the cluster permutation tests on
#for channel in ['FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']:
for channel in ['FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']:
    tmin = 0
    tmax = .75
    
    t_cope, clu_cope, clupv_cope, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = contrast,
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None,
                                                            out_type = 'indices', n_permutations = 10000,
                                                            n_jobs = 1) #specify number of cores to run in parallel
    clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_cope = np.asarray(clu_cope)[clupv_cope <= 0.05]
    
    
    t_corr, clu_corr, clupv_corr, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'confcorrect',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None,
                                                            out_type = 'indices', n_permutations = 10000,
                                                            n_jobs = 1) #specify number of cores to run in parallel
#    clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_corr = np.asarray(clu_corr)[clupv_corr <= 0.05]
    
    
    
    t_incorr, clu_incorr, clupv_incorr, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'confincorrect',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None,
                                                            out_type = 'indices', n_permutations = 10000,
                                                            n_jobs = 1) #specify number of cores to run in parallel
#    clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_incorr = np.asarray(clu_incorr)[clupv_incorr <= 0.05]
    
    
    
    
    
    
    # now we will plot the mean with std error ribbons around the signal shape, ideally using seaborn
    
    #first lets get all the single subject data into one dataframe:
    plottimes   = deepcopy(dat2use['grandmean'][0]).crop(tmax = tmax).times #this includes the baseline period for plotting
    plotdat_diff = np.empty(shape = (hightrls.size, plottimes.size))
    for i in range(hightrls.size):
        plotdat_diff[i,:] = np.squeeze(deepcopy(dat2use[contrast][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_diff = np.nanmean(plotdat_diff, axis = 0)
    plotdatsem_diff  = sp.stats.sem(plotdat_diff, axis = 0)
    
    plotdat_incorr = np.empty(shape = (hightrls.size, plottimes.size))
    for i in range(hightrls.size):
        plotdat_incorr[i,:] = np.squeeze(deepcopy(dat2use['confincorrect'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_incorr = np.nanmean(plotdat_incorr, axis = 0)
    plotdatsem_incorr  = sp.stats.sem(plotdat_incorr, axis = 0)
    
    plotdat_corr = np.empty(shape = (hightrls.size, plottimes.size))
    for i in range(hightrls.size):
        plotdat_corr[i,:] = np.squeeze(deepcopy(dat2use['confcorrect'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
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
    ax.set_title('confidence regressor at channel '+channel)
    ax.set_ylabel('beta (AU)')
    ax.set_xlabel('Time relative to feedback onset (s)')
    ax.legend(loc = 'upper left')
    for mask in masks_cope:
        ax.hlines(y = -3e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#636363', alpha = .5) #plot significance timepoints for difference effect
                  
    for mask in masks_incorr:
        ax.hlines(y = -3.2e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#e41a1c', alpha = .5) #plot significance timepoints for difference effect
    for mask in masks_corr:
        ax.hlines(y = -3.4e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#4daf4a', alpha = .5)
#%%
                  
#we know there is a significant effect within correct trials in both error and confidence
# so let's plot these two regressors alongside each other, and the ERP for the trial type
                
stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
if stat == 'beta':
    dat2use = deepcopy(data)
elif stat == 'tstat':
    dat2use = deepcopy(data_t)

#for channel in ['FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']:
for channel in ['Cz']:
    tmin = 0
    tmax = 1.0
    
    #we need three things: ERP, error regressor, confidence regressos
    
    
    t_err, clu_err, clupv_err, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'errorcorrect',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None,
                                                            out_type = 'indices', n_permutations = 5000,
                                                            n_jobs = 1) #specify number of cores to run in parallel
    clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_err = np.asarray(clu_err)[clupv_err <= 0.05]
    
    
    t_conf, clu_conf, clupv_conf, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'confcorrect',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None,
                                                            out_type = 'indices', n_permutations = 5000,
                                                            n_jobs = 1) #specify number of cores to run in parallel
#    clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_conf = np.asarray(clu_conf)[clupv_conf <= 0.05]

    
    
    # now we will plot the mean with std error ribbons around the signal shape, ideally using seaborn
    
    #first lets get all the single subject data into one dataframe:
    plottimes   = deepcopy(dat2use['grandmean'][0]).crop(tmax = tmax).times #this includes the baseline period for plotting
    plotdat_erp = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_erp[i,:] = np.squeeze(deepcopy(dat2use['correct'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    rescale_erp = False
    if rescale_erp:
        plotdat_erp = np.divide(plotdat_erp, 5)
    plotdatmean_erp = np.nanmean(plotdat_erp, axis = 0)
    plotdatsem_erp  = sp.stats.sem(plotdat_erp, axis = 0)
    
    plottimes   = deepcopy(dat2use['grandmean'][0]).crop(tmax = tmax).times #this includes the baseline period for plotting
    plotdat_erpincorr = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_erpincorr[i,:] = np.squeeze(deepcopy(dat2use['incorrect'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    if rescale_erp:
        plotdat_erpincorr = np.divide(plotdat_erpincorr, 5)
    plotdatmean_erpincorr = np.nanmean(plotdat_erpincorr, axis = 0)
    plotdatsem_erpincorr  = sp.stats.sem(plotdat_erpincorr, axis = 0)
    
    plottimes   = deepcopy(dat2use['grandmean'][0]).crop(tmax = tmax).times #this includes the baseline period for plotting
    plotdat_erpgmean = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_erpgmean[i,:] = np.squeeze(deepcopy(dat2use['grandmean'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    if rescale_erp:
        plotdat_erpgmean = np.divide(plotdat_erpgmean, 5)
    plotdatmean_erpgmean = np.nanmean(plotdat_erpgmean, axis = 0)
    plotdatsem_erpgmean  = sp.stats.sem(plotdat_erpgmean, axis = 0)
    
    plotdat_err = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_err[i,:] = np.squeeze(deepcopy(dat2use['errorcorrect'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_err = np.nanmean(plotdat_err, axis = 0)
    plotdatsem_err  = sp.stats.sem(plotdat_err, axis = 0)
    
    plotdat_conf = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_conf[i,:] = np.squeeze(deepcopy(dat2use['confcorrect'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_conf = np.nanmean(plotdat_conf, axis = 0)
    plotdatsem_conf  = sp.stats.sem(plotdat_conf, axis = 0)
    
    
    #plotdatmean = np.multiply(plotdatmean, 10e5)
    fig = plt.figure(figsize = (12, 6))
    ax  = fig.add_subplot(111)
    ax.plot(plottimes, plotdatmean_err, color = '#e41a1c', lw = 1.5, label = 'error regressor - correct')
    ax.fill_between(plottimes, plotdatmean_err - plotdatsem_err, plotdatmean_err + plotdatsem_err, alpha = .3, color = '#e41a1c')
    
    ax.plot(plottimes, plotdatmean_conf, color = '#4daf4a', lw = 1.5, label = 'confidence regressor - correct')
    ax.fill_between(plottimes, plotdatmean_conf - plotdatsem_conf, plotdatmean_conf + plotdatsem_conf, alpha = .3, color = '#4daf4a')
    
    ax.hlines([0], lw = 1, xmin = plottimes.min(), xmax = plottimes.max(), linestyles = 'dashed')
    ax.vlines([0], lw = 1, ymin = plotdatmean_incorr.min(), ymax = plotdatmean_incorr.max(), linestyles = 'dashed')
    ax.set_title('channel '+channel)
    ax.set_ylabel('beta (AU)')
    ax.set_xlabel('Time relative to feedback onset (s)')
    ax.legend(loc = 'upper left')
    for mask in masks_err:
        ax.hlines(y = -2.2e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#e41a1c', alpha = .5) #plot significance timepoints for difference effect
    for mask in masks_conf:
        ax.hlines(y = -2.4e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#4daf4a', alpha = .5) #plot significance timepoints for difference effect
#    fig.savefig(fname = op.join(figpath, 'error_confidence_correcttrials_channel_'+channel+'_20subs.eps'), format = 'eps', dpi = 300)
#    fig.savefig(fname = op.join(figpath, 'error_confidence_correcttrials_channel_'+channel+'_20subs.pdf'), format = 'pdf', dpi = 300)
    
    
    fig2 = plt.figure(figsize = (12, 6))
    ax2  = fig2.add_subplot(111)
    ax2.plot(plottimes, plotdatmean_erp, color = '#7570b3', lw = 1.5, label = 'ERP-correct')
    ax2.fill_between(plottimes, plotdatmean_erp-plotdatsem_erp, plotdatmean_erp+plotdatsem_erp, alpha = .3, color = '#636363')
                    
    ax2.plot(plottimes, plotdatmean_erpincorr, color = '#d95f02', lw = 1.5, label = 'ERP-incorrect')
    ax2.fill_between(plottimes, plotdatmean_erpincorr-plotdatsem_erpincorr, plotdatmean_erpincorr+plotdatsem_erpincorr, alpha = .3, color = '#636363')
    ax2.hlines([0], lw = 1, xmin = plottimes.min(), xmax = plottimes.max(), linestyles = 'dashed')
    ax2.vlines([0], lw = 1, ymin = plotdatmean_erpincorr.min(), ymax = plotdatmean_erpincorr.max(), linestyles = 'dashed')
    ax2.set_title('channel '+channel)
    ax2.set_ylabel('beta (AU)')
    ax2.set_xlabel('Time relative to feedback onset (s)')
    ax2.legend(loc = 'upper left')
#%%
#make a join plot of error and confidence in these trials, with the significant cluster times
#and plot the topographies of these clusters too, so we can see where this is represented in these time ranges
    
stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
if stat == 'beta':
    dat2use = deepcopy(data)
elif stat == 'tstat':
    dat2use = deepcopy(data_t)

contrast = 'errorcorrect' #choose the cope you want to look at #print(contrasts) shows what there is
channel  = 'Cz' #choose the channel you want to run the cluster permutation tests on
tmin = 0
tmax = 1

for contrast in ['errorcorrect', 'confcorrect']:
    
    #get the clusters again
    t_cope, clu_cope, clupv_cope, _ = runclustertest_epochs(data = dat2use,
                                                                contrast_name = contrast,
                                                                channels = [channel],
                                                                tmin = tmin,
                                                                tmax = tmax,
                                                                gauss_smoothing = None,
                                                                out_type = 'indices', n_permutations = 5000,
                                                                n_jobs = 1) #specify number of cores to run in parallel
    clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_cope = np.asarray(clu_cope)[clupv_cope <= 0.05]
        
    gave = mne.grand_average(dat2use[contrast]); #get the grand average data
    
    #worth remembering that the betas for these trial types (i.e. not parametric regressors) are on the same scale as the data (e.g. microvolts)
    vmin, vmax = -0.5, .5 #microvolts for the min/max values of the topographies we're going to plot
    
    
    fig = gave.plot_joint(topomap_args = dict(contours = 0, outlines = 'head', extrapolate = 'head', vmin = vmin, vmax = vmax),
                          times = np.arange(0.15, 0.5, .05)) #plot topos in this range
    ax = fig.axes[0] #get the axis of the erp plot
    tmins, tmaxs = [], [] #get the tmin and tmax of these clusters too as we're going to plot their topographies
    for mask in masks_cope:
        ax.hlines(y = -.7,
                  xmin = clutimes[mask[1]].min(),
                  xmax = clutimes[mask[1]].max(),
                  lw = 5, color = '#bdbdbd', alpha = .5)
        tmins.append(clutimes[mask[1]].min())
        tmaxs.append(clutimes[mask[1]].max())
#    fig.savefig(fname = op.join(figpath, '%s_jointplot_20subs_clusters_%s_.eps'%(contrast, channel)), format = 'eps', dpi = 300)
#    fig.savefig(fname = op.join(figpath, '%s_jointplot_20subs_clusters_%s_.pdf'%(contrast, channel)), format = 'pdf', dpi = 300)

    #now plot the topographies  
    for mask in range(len(tmins)):
        itmin = tmins[mask] #get the start time
        itmax = tmaxs[mask] #and the end time
        
        tcentre = np.add(itmin, np.divide(np.subtract(itmax, itmin),2)) #get the halfway point
        twidth  = np.subtract(itmax,itmin) #and get the width of the significant cluster
        fig = gave.plot_topomap(times = tcentre, #plot this time point
                                average = twidth, #and average over this time width around it (half of this value either side), so we plot the cluster time width
                                vmin = -.3, vmax = .3,
                                contours = 0,
                                #extrapolate = 'skirt',
                                res = 300, #resolution of the image
                                title = '%s to %sms'%(str(itmin), str(itmax)))
#        fig.savefig(fname = op.join(figpath, '%s_topomap_20subs_%s_to_%s_channel_%s_clusters.pdf'%(contrast, itmin, itmax, channel)), format = 'pdf', dpi = 300)
#        fig.savefig(fname = op.join(figpath, '%s_topomap_20subs_%s_to_%s_channel_%s_clusters.eps'%(contrast, itmin, itmax, channel)), format = 'eps', dpi = 300)
#     



#%%
#the cell above gives single channel plots of the ERP
#now we're going to make a joint plot of the grand average to show all channels, and mark out the significant time points from the cluster tests
#we will then get the tmin and tmax of each significant cluster and plot the topography of them too

stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
if stat == 'beta':
    dat2use = deepcopy(data)
elif stat == 'tstat':
    dat2use = deepcopy(data_t)

contrast = 'confincorrvscorr' #choose the cope you want to look at #print(contrasts) shows what there is
channel  = 'Cz' #choose the channel you want to run the cluster permutation tests on
tmin = 0
tmax = 1

#get the clusters again
t_cope, clu_cope, clupv_cope, _ = runclustertest_epochs(data = dat2use,
                                                        contrast_name = contrast,
                                                        channels = [channel],
                                                        tmin = tmin,
                                                        tmax = tmax,
                                                        gauss_smoothing = None,
                                                        out_type = 'indices', n_permutations = 'Default'
                                                        )
clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
masks_cope = np.asarray(clu_cope)[clupv_cope <= 0.05]
    
gave = mne.grand_average(dat2use[contrast]); #get the grand average data

#worth remembering that the betas for these trial types (i.e. not parametric regressors) are on the same scale as the data (e.g. microvolts)
vmin, vmax = -3, 3 #microvolts for the min/max values of the topographies we're going to plot


fig = gave.plot_joint(picks = 'eeg',
                      topomap_args = dict(contours = 0, outlines = 'skirt', vmin = vmin, vmax = vmax),
                      times = np.arange(0.15, 0.5, .05)) #plot topos in this range
ax = fig.axes[0] #get the axis of the erp plot
tmins, tmaxs = [], [] #get the tmin and tmax of these clusters too as we're going to plot their topographies
for mask in masks_cope:
    ax.hlines(y = -4,
              xmin = clutimes[mask[1]].min(),
              xmax = clutimes[mask[1]].max(),
              lw = 5, color = '#bdbdbd', alpha = .5)
    tmins.append(clutimes[mask[1]].min())
    tmaxs.append(clutimes[mask[1]].max())
fig.savefig(fname = op.join(figpath, 'incorrvscorr_jointplot_20subs_clusters_'+ channel + '_.eps'), format = 'eps', dpi = 300)
fig.savefig(fname = op.join(figpath, 'incorrvscorr_jointplot_20subs_clusters_'+ channel + '_.pdf'), format = 'pdf', dpi = 300)

#now plot the topographies  
for mask in range(len(tmins)):
    itmin = tmins[mask] #get the start time
    itmax = tmaxs[mask] #and the end time
    
    tcentre = np.add(itmin, np.divide(np.subtract(itmax, itmin),2)) #get the halfway point
    twidth  = np.subtract(itmax,itmin) #and get the width of the significant cluster
    fig = gave.plot_topomap(times = tcentre, #plot this time point
                            average = twidth, #and average over this time width around it (half of this value either side), so we plot the cluster time width
                            vmin = -2.5, vmax = 2.5, contours = 0,
                            extrapolate = 'head',
                            ch_type = 'eeg',
                            res = 300, #resolution of the image
                            title = '%s to %sms'%(str(itmin), str(itmax)))
    fig.savefig(fname = op.join(figpath, 'incorrvscorr_topomap_20subs_%s_to_%s_channel_%s_clusters.pdf'%(itmin, itmax, channel)), format = 'pdf', dpi = 300)
    fig.savefig(fname = op.join(figpath, 'incorrvscorr_topomap_20subs_%s_to_%s_channel_%s_clusters.eps'%(itmin, itmax, channel)), format = 'eps', dpi = 300)
     