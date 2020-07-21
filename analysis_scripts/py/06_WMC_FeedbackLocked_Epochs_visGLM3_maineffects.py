#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 15:51:28 2019

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
            evokeds = dict(
                    corr   = data['correct'],
                    incorr = data['incorrect'],
                    diff   = data['incorrvscorr']
                    ),
            colors = dict(
                    corr   = '#66c2a5',
                    incorr = '#fc8d62',
                    diff   = '#8da0cb'),
            legend = 'upper right', picks = channel,
            ci = .68, show_sensors = False, title = 'electrode ' + channel,
            truncate_xaxis=False
            )
gave = mne.grand_average(data['incorrvscorr'])

deepcopy(gave).plot_joint(times = np.arange(0.1,0.7,.1),
                title = 'incorr vs corr')
        
#%%
tmin, tmax = 0, 1
for channel in ['FCz', 'Cz']:
    t_ern, clu_ern, clupv_ern, h0_ern = runclustertest_epochs(data = data,
                                                              contrast_name = 'incorrvscorr',
                                                              channels = [channel],
                                                              tmin = tmin,
                                                              tmax = tmax,
                                                              gauss_smoothing = None,
                                                              out_type = 'indices', n_permutations = 'Default'
                                                              )
    clutimes = deepcopy(data['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_ern = np.asarray(clu_ern)[clupv_ern < 0.05]
            
    
    fig = plt.figure()
    ax = plt.axes()
    mne.viz.plot_compare_evokeds(
        evokeds = dict(
                corr   = data['correct'],
                incorr = data['incorrect'],
                diff   = data['incorrvscorr']
                ),
        colors = dict(
                corr   = '#66c2a5',
                incorr = '#fc8d62',
                diff   = '#8da0cb'),
        legend = 'upper right', picks = channel,
        ci = .68, show_sensors = False, title = 'electrode ' + channel,
        truncate_xaxis=False, axes = ax,
        vlines = [0, 0.5]
        )
            
    ax.set_title('feedback evoked response at electrode '+ channel)
    ax.set_ylabel('average beta (AU)')
    for mask in range(len(masks_ern)):
        ax.hlines(y = -2.5,
                  xmin = np.min(clutimes[masks_ern[mask][1]]),
                  xmax = np.max(clutimes[masks_ern[mask][1]]),
                  lw=5, color = '#4292c6', alpha = .5) #plot significance timepoints for difference effect
              
              
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

#%%
stat = 'beta' #choose whether you want to use the single subject tstats or betas for the analysis
if stat == 'beta':
    dat2use = deepcopy(data)
elif stat == 'tstat':
    dat2use = deepcopy(data_t)

contrast = 'incorrvscorr' #choose the cope you want to look at #print(contrasts) shows what there is
channel  = 'Cz' #choose the channel you want to run the cluster permutation tests on
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
                                                            out_type = 'indices', n_permutations = 'Default'
                                                            )
    clutimes = deepcopy(dat2use['grandmean'][0]).crop(tmin = tmin, tmax = tmax).times
    masks_cope = np.asarray(clu_cope)[clupv_cope <= 0.05]
    
    
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
        plotdat_incorr[i,:] = np.squeeze(deepcopy(dat2use['incorrect'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
    plotdatmean_incorr = np.nanmean(plotdat_incorr, axis = 0)
    plotdatsem_incorr  = sp.stats.sem(plotdat_incorr, axis = 0)
    
    plotdat_corr = np.empty(shape = (subs.size, plottimes.size))
    for i in range(subs.size):
        plotdat_corr[i,:] = np.squeeze(deepcopy(dat2use['correct'][i]).pick_channels([channel]).crop(tmax=tmax).data)
    
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
    ax.set_title('grand average ERP: incorrect vs correct trials at channel '+channel)
    ax.set_ylabel('beta (AU)')
    ax.set_xlabel('Time relative to feedback onset (s)')
    ax.legend(loc = 'upper left')
    for mask in masks_cope:
        ax.hlines(y = -5e-6,
                  xmin = np.min(clutimes[mask[1]]),
                  xmax = np.max(clutimes[mask[1]]),
                  lw=5, color = '#636363', alpha = .5) #plot significance timepoints for difference effect
#    fig.savefig(fname = op.join(figpath, 'diffwave_20subs_channel_'+channel+'_'+ stat +'.eps'), format = 'eps', dpi = 300)
#    fig.savefig(fname = op.join(figpath, 'diffwave_20subs_channel_'+channel+'_'+ stat +'.pdf'), format = 'pdf', dpi = 300)
#    
#    plt.close()
#%%
#the cell above gives single channel plots of the ERP
#now we're going to make a joint plot of the grand average to show all channels, and mark out the significant time points from the cluster tests
#we will then get the tmin and tmax of each significant cluster and plot the topography of them too
    
contrast = 'incorrvscorr' #choose the cope you want to look at #print(contrasts) shows what there is
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
if laplacian:
    vmin, vmax = -2, 2 #the units are different now, so its kinda arbitrary really

fig = gave.plot_joint(topomap_args = dict(contours = 0, outlines = 'skirt', vmin = vmin, vmax = vmax),
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
#fig.savefig(fname = op.join(figpath, 'incorrvscorr_jointplot_20subs_clusters_'+ channel + '_.eps'), format = 'eps', dpi = 300)
#fig.savefig(fname = op.join(figpath, 'incorrvscorr_jointplot_20subs_clusters_'+ channel + '_.pdf'), format = 'pdf', dpi = 300)

#now plot the topographies  
for mask in range(len(tmins)):
    itmin = tmins[mask] #get the start time
    itmax = tmaxs[mask] #and the end time
    
    tcentre = np.add(itmin, np.divide(np.subtract(itmax, itmin),2)) #get the halfway point
    twidth  = np.subtract(itmax,itmin) #and get the width of the significant cluster
    fig = gave.plot_topomap(times = tcentre, #plot this time point
                            average = twidth, #and average over this time width around it (half of this value either side), so we plot the cluster time width
                            vmin = vmin, vmax = vmax, contours = 0,
                            extrapolate = 'head',
                            res = 300, #resolution of the image
                            title = '%s to %sms'%(str(itmin), str(itmax)))
#    fig.savefig(fname = op.join(figpath, 'incorrvscorr_topomap_20subs_%s_to_%s_channel_%s_clusters.pdf'%(itmin, itmax, channel)), format = 'pdf', dpi = 300)
#    fig.savefig(fname = op.join(figpath, 'incorrvscorr_topomap_20subs_%s_to_%s_channel_%s_clusters.eps'%(itmin, itmax, channel)), format = 'eps', dpi = 300)

    
#%%
gave = mne.grand_average(data['grandmean']);
fig = plt.figure(figsize = (9,9))
ax = fig.add_subplot(111)
gave.plot_sensors(show_names=True, axes = ax)
fig.savefig(fname = op.join(figpath, 'sensor_locations.eps'), format = 'eps', dpi = 300)
fig.savefig(fname = op.join(figpath, 'sensor_locations.pdf'), format = 'pdf', dpi = 300)
del(gave)