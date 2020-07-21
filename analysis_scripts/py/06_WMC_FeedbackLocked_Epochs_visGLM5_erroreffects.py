#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 15:22:32 2019

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

figpath = op.join(wd, 'figures', 'eeg_figs', 'feedbacklocked', 'epochs_glm5')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm
nsubs = subs.size


contrasts = ['defcorrect', 'justcorrect', 'incorrect',
             'errdefcorrect', 'errjustcorrect', 'errincorrect',
             'confdefcorrect', 'confjustcorrect', 'confincorrect',
             'incorrvsdef', 'incorrvsjust', 'justvsdef',
             'errorincorrvsdef', 'errorincorrvsjust', 'errorjustvsdef',
             'confincorrvsdef', 'confincorrvsjust', 'confjustvsdef']

threshold = -10
if threshold == -10:
    glmnum = 6
else:
    glmnum = 5

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

#drop right mastoid from literally everything here lol its not useful anymore
for cope in data.keys():
    for i in range(subs.size):
        data[cope][i]   = data[cope][i].drop_channels(['RM'])#.set_eeg_reference(ref_channels='average')
        data_t[cope][i] = data_t[cope][i].drop_channels(['RM'])#.set_eeg_reference(ref_channels='average')
#%%
gave = mne.grand_average(data['incorrect']); times = gave.times; del(gave)      
#%%
#some quick and dirty visualisation of error regressor at different electrodes in the three trial groupings
for channel in ['FCz', 'Cz', 'CPz', 'Pz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(defcorr   = data['errdefcorrect'],
                           justcorr = data['errjustcorrect'],
                           incorr   = data['errincorrect']),
            colors = dict(defcorr   = '#31a354',
                          justcorr = '#a1d99b',
                          incorr   = '#de2d26'),
            linestyles = dict(justcorr = 'dashed'),
            show_legend = 'upper right', picks = channel,
            ci = .68, show_sensors = False, title = 'error regressor at electrode ' + channel,
            truncate_xaxis=False)

#%% and another quick and dirty of the differences ...
            
for channel in ['FCz', 'Cz', 'CPz', 'Pz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(incorrvsdef    = data['errorincorrvsdef'],
                           incorrvsjust   = data['errorincorrvsjust'],
                           justvsdef      = data['errorjustvsdef']),
            colors = dict(incorrvsjust    = '#8da0cb',
                          incorrvsdef     = '#fc8d62',
                          justvsdef       = '#66c2a5'),
#            linestyles = dict(justcorr    = 'dashed'),
            show_legend = 'upper right', picks = channel,
            ci = .68, show_sensors = False, title = 'difference in error regressor between trial types at electrode ' + channel,
            truncate_xaxis=False)            
            


#%% this is a MUCH better chunk of code to use than the stuff above, which is a bit shitty really and just coarse visualisations

stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
if stat == 'beta':
    dat2use = deepcopy(data)
elif stat == 'tstat':
    dat2use = deepcopy(data_t)

channel  = 'Cz' #choose the channel you want to run the cluster permutation tests on
#for channel in ['FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']:
#for channel in ['FCz', 'Cz', 'CPz', 'Pz', 'POz', 'Oz']:
for channel in ['FCz', 'Cz', 'Pz']:
    tmin = 0
    tmax = 1
    clutimes = deepcopy(dat2use['incorrect'][0]).crop(tmin=tmin,tmax=tmax).times
    
    #three contrasts of interest here:
    
    #incorrect vs definitely correct
    t_ivsd, clu_ivsd, clupv_ivsd, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'errorincorrvsdef',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None, out_type = 'indices',
                                                            n_permutations = 10000)
    masks_ivsd = np.asarray(clu_ivsd)[clupv_ivsd <= 0.05]
    
    #incorrect vs just correct
    t_ivsj, clu_ivsj, clupv_ivsj, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'errorincorrvsjust',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None, out_type = 'indices',
                                                            n_permutations = 10000)
    masks_ivsj = np.asarray(clu_ivsj)[clupv_ivsj <= 0.05]  
    
    #just vs definitely correct
    t_jvsd, clu_jvsd, clupv_jvsd, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'errorjustvsdef',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None, out_type = 'indices',
                                                            n_permutations = 10000)
    masks_jvsd = np.asarray(clu_jvsd)[clupv_jvsd <= 0.05]
    
    

    
    
    # now we will plot the mean with std error ribbons around the signal shape
    
    #first lets get all the single subject data into one dataframe:
    plottimes   = deepcopy(dat2use['incorrect'][0]).crop(tmax = tmax).times #this includes the baseline period for plotting
    
    #difference in error regressor between incorrect and definitely correct trials
    plotdat_ivsd = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_ivsd[i,:] = np.squeeze(deepcopy(dat2use['errorincorrvsdef'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_ivsd = np.nanmean(plotdat_ivsd, axis = 0)
    plotdatsem_ivsd  = sp.stats.sem(plotdat_ivsd, axis = 0)
    
    
    #difference in error regressor between incorrect trials and just correct trials
    plotdat_ivsj = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_ivsj[i,:] = np.squeeze(deepcopy(dat2use['errorincorrvsjust'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_ivsj = np.nanmean(plotdat_ivsj, axis = 0)
    plotdatsem_ivsj  = sp.stats.sem(plotdat_ivsj, axis = 0)
    
    #difference in error regressor between just correct and definitely correct trials
    plotdat_jvsd = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_jvsd[i,:] = np.squeeze(deepcopy(dat2use['errorjustvsdef'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_jvsd = np.nanmean(plotdat_jvsd, axis = 0)
    plotdatsem_jvsd  = sp.stats.sem(plotdat_jvsd, axis = 0)
    
    
    #plotdatmean = np.multiply(plotdatmean, 10e5)
    fig = plt.figure(figsize = (12, 6))
    ax  = fig.add_subplot(111)
    ax.plot(plottimes, plotdatmean_ivsd, color = '#de2d26', lw = 1.5, label = 'incorrect vs definitely correct')
    ax.fill_between(plottimes, plotdatmean_ivsd - plotdatsem_ivsd, plotdatmean_ivsd + plotdatsem_ivsd, alpha = .3, color = '#de2d26')
    
    ax.plot(plottimes, plotdatmean_ivsj, color = '#fc9272', lw = 1.5, label = 'incorrect vs just correct', ls = 'dashed')
    ax.fill_between(plottimes, plotdatmean_ivsj - plotdatsem_ivsj, plotdatmean_ivsj + plotdatsem_ivsj, alpha = .3, color = '#fc9272')
    
    ax.plot(plottimes, plotdatmean_jvsd, color = '#31a354', lw = 1.5, label = 'just vs definitely correct')
    ax.fill_between(plottimes, plotdatmean_jvsd - plotdatsem_jvsd, plotdatmean_jvsd + plotdatsem_jvsd, alpha = .3, color = '#31a354')
    
    
    ax.hlines([0], lw = 1, xmin = plottimes.min(), xmax = plottimes.max(), linestyles = 'dashed')
    ax.vlines([0], lw = 1, ymin = plotdatmean_jvsd.min(), ymax = plotdatmean_jvsd.max(), linestyles = 'dashed')
    ax.set_title('error regressor at channel '+channel)
    ax.set_ylabel('beta (AU)')
    ax.set_xlabel('Time relative to feedback onset (s)')
    ax.legend(loc = 'upper left')
    
    for mask in masks_ivsd:
        ax.hlines(y = -8e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#de2d26', alpha = .5) #plot significance timepoints for difference effect
    for mask in masks_ivsj:
        ax.hlines(y = -8.3e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#fc9272', alpha = .5) #plot significance timepoints for difference effect
    for mask in masks_jvsd:
        ax.hlines(y = -8.6e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#31a354', alpha = .5) #plot significance timepoints for difference effect
    fig.savefig(fname = op.join(figpath, 'error_x_diffwaves_20subs_threshold%d_channel_%s_%s.eps'%(abs(threshold),channel, stat)), format = 'eps', dpi = 300)
    fig.savefig(fname = op.join(figpath, 'error_x_diffwaves_20subs_threshold%d_channel_%s_%s.pdf'%(abs(threshold),channel, stat)), format = 'pdf', dpi = 300)
    
    plt.close()
#    fig.savefig(fname = op.join(figpath, 'trialtype_diffwaves_20subs_threshold%s_channel_%s_%s.eps'%(str(abs(threshold)), channel, stat)), format = 'eps', dpi = 300)

#%%
# we know there are differences in error across the trial types
# so lets loop over these trial types, focussing on the error regressor (this helps to describe these differences between trial types)

stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
if stat == 'beta':
    dat2use = deepcopy(data)
elif stat == 'tstat':
    dat2use = deepcopy(data_t)
    
#for channel in ['FCz', 'Cz', 'CPz', 'Pz']:
for channel in ['FCz', 'Cz', 'Pz']:
#for contrast in ['confdefcorrect', 'confjustcorrect', 'confincorrect']:
    tmin = 0
    tmax = 1
    clutimes = deepcopy(dat2use['incorrect'][0]).crop(tmin=tmin,tmax=tmax).times
    
    #three contrasts of interest here:
    
    #incorrect vs definitely correct
    t_def, clu_def, clupv_def, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'errdefcorrect',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None, out_type = 'indices',
                                                            n_permutations = 10000)
    masks_def = np.asarray(clu_def)[clupv_def <= 0.05]
    
    #incorrect vs just correct
    t_just, clu_just, clupv_just, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'errjustcorrect',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None, out_type = 'indices',
                                                            n_permutations = 10000)
    masks_just = np.asarray(clu_just)[clupv_just <= 0.05]  
    
    #just vs definitely correct
    t_incorr, clu_incorr, clupv_incorr, _ = runclustertest_epochs(data = dat2use,
                                                            contrast_name = 'errincorrect',
                                                            channels = [channel],
                                                            tmin = tmin,
                                                            tmax = tmax,
                                                            gauss_smoothing = None, out_type = 'indices',
                                                            n_permutations = 10000)
    masks_incorr = np.asarray(clu_incorr)[clupv_incorr <= 0.05]
    
    
    # now we will plot the mean with std error ribbons around the signal shape
    
    #first lets get all the single subject data into one dataframe:
    plottimes   = deepcopy(dat2use['incorrect'][0]).crop(tmax = tmax).times #this includes the baseline period for plotting
    
    #confidence regressor in definitely correct trials
    plotdat_def = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_def[i,:] = np.squeeze(deepcopy(dat2use['errdefcorrect'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_def = np.nanmean(plotdat_def, axis = 0)
    plotdatsem_def  = sp.stats.sem(plotdat_def, axis = 0)
    
    
    #confidence regressor in just correct trials
    plotdat_just = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_just[i,:] = np.squeeze(deepcopy(dat2use['errjustcorrect'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_just = np.nanmean(plotdat_just, axis = 0)
    plotdatsem_just  = sp.stats.sem(plotdat_just, axis = 0)
    
    #confidence regressor in incorrect trials
    plotdat_incorr = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_incorr[i,:] = np.squeeze(deepcopy(dat2use['errincorrect'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_incorr = np.nanmean(plotdat_incorr, axis = 0)
    plotdatsem_incorr  = sp.stats.sem(plotdat_incorr, axis = 0)
    
    fig = plt.figure(figsize = (12, 6))
    ax  = fig.add_subplot(111)
    ax.plot(plottimes, plotdatmean_def, color = '#31a354', lw = 1.5, label = 'error regressor: definitely correct')
    ax.fill_between(plottimes, plotdatmean_def - plotdatsem_def, plotdatmean_def + plotdatsem_def, alpha = .3, color = '#31a354')
    
    ax.plot(plottimes, plotdatmean_just, color = '#a1d99b', lw = 1.5, label = 'error regressor: just correct', ls = 'dashed')
    ax.fill_between(plottimes, plotdatmean_just - plotdatsem_just, plotdatmean_just + plotdatsem_just, alpha = .3, color = '#a1d99b')
    
    ax.plot(plottimes, plotdatmean_incorr, color = '#de2d26', lw = 1.5, label = 'error regressor: incorrect')
    ax.fill_between(plottimes, plotdatmean_incorr - plotdatsem_incorr, plotdatmean_incorr + plotdatsem_incorr, alpha = .3, color = '#de2d26')
    
    
    ax.hlines([0], lw = 1, xmin = plottimes.min(), xmax = plottimes.max(), linestyles = 'dashed')
    ax.vlines([0], lw = 1, ymin = plotdatmean_just.min(), ymax = plotdatmean_just.max(), linestyles = 'dashed')
    ax.set_title('error regressor at channel '+channel)
    ax.set_ylabel('beta (AU)')
    ax.set_xlabel('Time relative to feedback onset (s)')
    ax.legend(loc = 'upper left')
    
    for mask in masks_def:
        ax.hlines(y = -8e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#31a354', alpha = .5) #plot significance timepoints for difference effect
    for mask in masks_just:
        ax.hlines(y = -8.3e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#a1d99b', alpha = .5) #plot significance timepoints for difference effect
    for mask in masks_incorr:
        ax.hlines(y = -8.6e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#de2d26', alpha = .5)
    
    fig.savefig(fname = op.join(figpath, 'error_trialtypes_20subs_threshold%d_channel_%s_%s.eps'%(abs(threshold),channel, stat)), format = 'eps', dpi = 300)
    fig.savefig(fname = op.join(figpath, 'error_trialtypes_20subs_threshold%d_channel_%s_%s.pdf'%(abs(threshold),channel, stat)), format = 'pdf', dpi = 300)
#    plt.close()

#%%
#now we will construct jointplots for a few different copes, and plot topographies of significant clusters too.
# clustering will be based off Cz for this

channel  = 'Cz' #choose the channel you want to run the cluster permutation tests on
tmin = 0
tmax = 1

stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
if stat == 'beta':
    dat2use = deepcopy(data)
elif stat == 'tstat':
    dat2use = deepcopy(data_t)

contrastlist = ['errorincorrvsdef', 'errorincorrvsjust', 'errorjustvsdef', 'errdefcorrect', 'errjustcorrect', 'errincorrect']
for channel in ['FCz', 'Cz', 'Pz']:
    for contrast in contrastlist:
        
    
        #get the clusters again
        t_cope, clu_cope, clupv_cope, _ = runclustertest_epochs(data = dat2use,
                                                                contrast_name = contrast,
                                                                channels = [channel],
                                                                tmin = tmin,
                                                                tmax = tmax,
                                                                gauss_smoothing = None,
                                                                out_type = 'indices', n_permutations = 5000
                                                                )
        clutimes = deepcopy(dat2use['incorrect'][0]).crop(tmin = tmin, tmax = tmax).times
        masks_cope = np.asarray(clu_cope)[clupv_cope <= 0.05]
        print('\ncontrast %s: %s significant clusters\n'%(contrast, str(len(masks_cope))))
        
        gave = mne.grand_average(dat2use[contrast]); #get the grand average data
    
        #worth remembering that the betas for these trial types (i.e. not parametric regressors) are on the same scale as the data (e.g. microvolts)
        
        vmin, vmax = -3, 3 #microvolts for the min/max values of the topographies we're going to plot
    #    if contrast == 'errdefcorrect':
    #        vmin,vmax = -1,1
    
        fig = gave.plot_joint(picks = 'eeg', title = contrast,
                              topomap_args = dict(contours = 0, outlines = 'skirt', vmin = vmin, vmax = vmax),
                              times = np.arange(0.15, 0.5, .05)) #plot topos in this range
        ax = fig.axes[0] #get the axis of the erp plot
        tmins, tmaxs = [], [] #get the tmin and tmax of these clusters too as we're going to plot their topographies
        for mask in masks_cope:
            yline = np.round(gave.data.min() * 10**6) -1
            ax.hlines(y = yline,
                      xmin = clutimes[mask[1]].min(),
                      xmax = clutimes[mask[1]].max(),
                      lw = 5, color = '#bdbdbd', alpha = .5)
            tmins.append(clutimes[mask[1]].min())
            tmaxs.append(clutimes[mask[1]].max())
        fig.savefig(fname = op.join(figpath, '%s_jointplot_20subs_threshold%d_clusters_%s_.eps'%(contrast, abs(threshold), channel)), format = 'eps', dpi = 300)
        fig.savefig(fname = op.join(figpath, '%s_jointplot_20subs_threshold%d_clusters_%s_.pdf'%(contrast, abs(threshold), channel)), format = 'pdf', dpi = 300)
    
    #now plot the topographies  
        for mask in range(len(tmins)):
            itmin = tmins[mask] #get the start time
            itmax = tmaxs[mask] #and the end time
            
            plotdatmin = deepcopy(gave).crop(tmin = itmin, tmax = itmax).data.min()
            plotdatmax = deepcopy(gave).crop(tmin = itmin, tmax = itmax).data.max()
#            print(itmin, plotdatmin, plotdatmax)
            
            tcentre = np.add(itmin, np.divide(np.subtract(itmax, itmin),2)) #get the halfway point
            twidth  = np.subtract(itmax,itmin) #and get the width of the significant cluster
            if contrast in ['errorincorrvsdef']:
                topovmin,topovmax = -1.4,1.4
            elif contrast in ['errorincorrvsjust']:
                topovmin = -4
                topovmax =  4
                if itmin == 0.304:
                    topovmin, topovmax = -8.5, 8.5
            else:
                topovmin = -3.5
                topovmax =  3.5
            fig = gave.plot_topomap(times = tcentre, #plot this time point
                                    average = twidth, #and average over this time width around it (half of this value either side), so we plot the cluster time width
                                    vmin = topovmin,
                                    vmax = topovmax,
                                    #colorbar=True,
                                    contours = 4,
                                    extrapolate = 'head',
                                    ch_type = 'eeg',
                                    res = 300, #resolution of the image
                                    title = '%s to %sms'%(str(itmin), str(itmax)))
            fig.savefig(fname = op.join(figpath, '%s_topomap_20subs_threshold%d_%s_to_%s_channel_%s_clusters.pdf'%(contrast, abs(threshold), itmin, itmax, channel)), format = 'pdf', dpi = 300)
            fig.savefig(fname = op.join(figpath, '%s_topomap_20subs_threshold%d_%s_to_%s_channel_%s_clusters.eps'%(contrast, abs(threshold), itmin, itmax, channel)), format = 'eps', dpi = 300)
            plt.close()


