
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:30:29 2019

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


sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, runclustertest_epochs

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

figpath = op.join(wd, 'figures', 'eeg_figs', 'feedbacklocked', 'epochs_glm4')

subs  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
subs  = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm
nsubs = subs.size


contrasts = ['grandmean', 'neutral', 'cued', 'cuedvsneutral', 'pside',
             'incorrvscorr', 'incorr', 'corr', 'error', 'confidence',
             'error_corr', 'error_incorr', 'error_incorrvscorr',
             'conf_corr', 'conf_incorr', 'confidence_incorrvscorr',
             'confupdate', 'confupdate_corr', 'confupdate_incorr', 'confupdate_incorrvscorr']

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
        data[name].append(   mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm4', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_betas-ave.fif'))[0])
        data_t[name].append( mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm4', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_tstats-ave.fif'))[0])

#drop right mastoid from literally everything here lol its not useful anymore
for cope in data.keys():
    for i in range(subs.size):
        data[cope][i]   = data[cope][i].drop_channels(['RM'])#.set_eeg_reference(ref_channels='average')
        data_t[cope][i] = data_t[cope][i].drop_channels(['RM'])#.set_eeg_reference(ref_channels='average')
#%%
        
#gave_pside = mne.grand_average(data['pside']); #gave_pside.data = toverparam(data['pside'])
#gave_pside.plot_joint(title = 'probed side (left vs right)', picks = 'eeg', topomap_args = dict(outlines='head', contours = 0))
#gave_pside.plot(picks = ['P7', 'P8'], window_title = 'probed left vs right', spatial_colors=True)

#gave_pleft = mne.grand_average(data['probedleft']); #gave_pleft.data = toverparam(data['probedleft'])
#gave_pleft.plot_joint(title = 'probed left', picks = 'eeg', topomap_args = dict(outlines='head', contours = 0))
#gave_pleft.plot(picks = ['P7', 'P8'], spatial_colors = True, window_title = 'probed left')
#
#gave_pright = mne.grand_average(data['probedright']); #gave_pright.data = toverparam(data['probedright'])
#gave_pright.plot_joint(title = 'probed right', picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0))
#gave_pright.plot(picks = ['P7', 'P8'], spatial_colors = True, window_title = 'probed right')
#        
#gave_corr = mne.grand_average(data['corr'])
#gave_incorr = mne.grand_average(data['incorr'])
#gave_ivsc = mne.grand_average(data['incorrvscorr'])
mne.viz.plot_sensors(data['incorr'][0].info, show_names=True)

for channel in ['FCZ', 'CZ', 'CPZ', 'PZ']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    incorr = data_t['incorr'],
                    corr   = data_t['corr'],
                    ivsc   = data_t['incorrvscorr']),
            colors = dict(
                    incorr = '#fc8d59',
                    corr   = '#91cf60',
                    ivsc   = '#91bfdb'
                    ),
                    ci = .68, show_legend = 'upper right',
                    picks = channel, show_sensors = False,#ylim = dict(eeg=[-4,4]),
                    truncate_xaxis = False)
#%%
#can we do a spatio-temporal cluster test on the incorrect vs correct data to see, without dipping, where the peak is centred?
#get all the data together into one array
tmin, tmax= 0, 0.5
spatiotemporal_cludat = np.empty(shape = (subs.size, deepcopy(data['corr'])[0].crop(tmin=tmin,tmax=tmax).times.size,61))
for i in range(subs.size):
    spatiotemporal_cludat[i,:,:] = deepcopy(data['incorrvscorr'])[i].crop(tmin=tmin,tmax=tmax).pick_types(eeg=True).data.reshape(-1,61)
            
st_t, st_clu, st_clupv, _ = mne.stats.spatio_temporal_cluster_1samp_test(spatiotemporal_cludat, out_type='mask',
                                                                         connectivity = mne.channels.find_ch_connectivity(mne.grand_average(data['incorrvscorr']).info, ch_type='eeg')[0])
goodclus = np.where(st_clupv < 0.01)[0]
st_masks = np.asarray(st_clu)[st_clupv<0.01]

mne.viz.plot_evoked_topomap(mne.grand_average(data['incorrvscorr']), mask=st_masks)

cluster_error_channels = np.unique(st_clu[goodclus[0]][1])
mne.grand_average(data['incorrvscorr']).plot_joint()
mne.grand_average(data['incorrvscorr']).drop_channels(['VEOG', 'HEOG'])

for i in range(len(goodclus)):
    mne.grand_average(data['incorrvscorr']).crop(tmin=tmin, tmax=tmax).plot_topomap(mask = st_clu[goodclus[i]].T,
                            mask_params = dict(marker='x',markerfacecolor='k', markeredgecolor='k', linewidth = 0, markersize = 5),
                            contours = 0, average = .1,
                            outlines = 'head', vmin = -3, vmax = 3,
                            times = np.arange(start = 0.15,stop = 0.25,step = .01))

#%%
gave_grandmean    = mne.grand_average(data['grandmean']); gave_grandmean.data = toverparam(data['grandmean'])
gave_incorrvscorr = mne.grand_average(data['incorrvscorr']); gave_incorrvscorr.data = toverparam(data['incorrvscorr'])
alltimes = gave_incorrvscorr.times
gave_incorrvscorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0))


#nonpara cluster t test to see where diff is significant
tmin, tmax = 0, 1
t_fern, clu_fern, clupv_fern, H0_fern = runclustertest_epochs(data = data,
                                                          channels = ['PZ'],
                                                          contrast_name = 'incorrvscorr',
                                                          tmin = tmin, tmax = tmax,
                                                          gauss_smoothing = None, out_type = 'indices',
                                                          n_permutations = 'Default')
masks_fern = np.asarray(clu_fern)[clupv_fern <= 0.05]

#this will plot the erp at every electrode for you
mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    #incorr = data['incorr'],
                    #corr   = data['corr'],
                    ivsc   = data['incorrvscorr']),
            colors = dict(
                    #incorr = '#fc8d59',
                    #corr   = '#91cf60',
                    ivsc   = '#91bfdb'
                    ),
                    ci = .68, show_legend = 'upper right',axes='topo',
                    picks = 'eeg', show_sensors = False,#ylim = dict(eeg=[-4,4]),
                    truncate_xaxis = False)


fig = plt.figure()
ax = plt.axes()
ferntimes = deepcopy(gave_incorrvscorr).crop(tmin=tmin, tmax=tmax).times
mne.viz.plot_compare_evokeds(
            evokeds = dict(incorr = data['incorr'], corr   = data['corr'], ivsc   = data['incorrvscorr']),
            colors  = dict(incorr = '#fc8d59',      corr   = '#91cf60',    ivsc   = '#91bfdb'),
                    ci = .68, show_legend = 'upper right',
                    picks = 'PZ', show_sensors = False, truncate_xaxis = False, axes = ax)
ax.set_title('feedback evoked response at electrode Pz')
ax.hlines(y = 0, linestyles = 'dashed', color = '#000000', lw = .75, xmin = alltimes.min(), xmax = alltimes.max())
ax.vlines(x = 0, linestyles = 'dashed', color = '#000000', lw = .75, ymin = -3, ymax = 14)
ax.set_ylabel('t-value')
ax.set_xlabel('Time relative to feedback onset (s)')
for mask in range(len(masks_fern)):
    ax.hlines(y = 0,
              xmin = np.min(ferntimes[masks_fern[mask][1]]),
              xmax = np.max(ferntimes[masks_fern[mask][1]]),
              lw=5, color = '#4292c6', alpha=.5) #plot significance timepoints for difference effect
              
              
#this will plot these significant times as an overlay onto the plot_joint image
#requires finding the right axis within the subplot thats drawn in order to draw them on properly              

#%%
fig1 = gave_incorrvscorr.plot_joint(picks = 'eeg',
                                    topomap_args = dict(contours=0, outlines='head', vmin = -5, vmax=5, scalings = dict(eeg=1), units = 'tstat'),
                                    ts_args = dict(unit = False, ylim = dict(eeg=[-9,9]), units = 'tstat'))
ax1 = fig1.axes[0]
for mask in range(len(mask_ern_05)):
    #x =  times[mask_diff_05[mask][1]]
    #y = np.zeros(len(x)); y.fill(-5.3)
    #ax.scatter(x, y, color = '#998ec3', alpha = .5, marker='s')
    ax1.hlines(y = -8,
              xmin = np.min(erntimes[mask_ern_05[mask][1]]),
              xmax = np.max(erntimes[mask_ern_05[mask][1]]),
              lw=5, color = '#4292c6', alpha = .5) #plot significance timepoints for difference effect

#times for this effect:
for mask in mask_ern_05:
    print(erntimes[mask[1]].min(), erntimes[mask[1]].max())
#%%    
gave_errorcorr   = mne.grand_average(data_t['error_corr']);   #gave_errorcorr.data = toverparam(data_t['error_corr'])
gave_errorincorr = mne.grand_average(data_t['error_incorr']); #gave_errorincorr.data = toverparam(data_t['error_incorr'])
gave_error_incorrvscorr = mne.grand_average(data_t['error_incorrvscorr']); #gave_error_incorrvscorr.data = toverparam(data_t['error_incorrvscorr'])
gave_error = mne.grand_average(data_t['error']); #gave_error.data =toverparam(data_t['error'])

dat2use= deepcopy(gave_errorcorr)
dat2use.plot_joint(topomap_args=dict(outlines='head'), picks='eeg')

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(alltimes, deepcopy(gave_errorcorr).pick_channels(['FCZ']).data[0], label = 'error - correct', color = '#2ca25f', lw = 1.5)
#ax.plot(alltimes, deepcopy(gave_errorincorr).pick_channels(['FCZ']).data[0], label = 'error - incorrect', color = '#2ca25f', ls = 'dashed', lw = 1.5)
#ax.plot(alltimes, deepcopy(gave_error_incorrvscorr).pick_channels(['FCZ']).data[0], label = 'error - incorrect vs correct', color = '#3182bd', lw = 1.5)
#ax.plot(alltimes, deepcopy(gave_error).pick_channels(['FCZ']).data[0], label = 'error-alltrials', color = '#7b3294', lw = 1.5)
#ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), linestyles = 'dashed', lw = .75, color = '#000000')
#ax.vlines(x = 0, ymin = -4, ymax = 4, linestyles = 'dashed', color = '#000000', lw = .75)
#fig.legend(loc = 'upper right')


#%%
gave_confcorr   = mne.grand_average(data_t['conf_corr']);   gave_confcorr.data = toverparam(data_t['conf_corr'])
gave_confincorr = mne.grand_average(data_t['conf_incorr']); gave_confincorr.data = toverparam(data_t['conf_incorr'])
gave_confincorrvscorr = mne.grand_average(data_t['confidence_incorrvscorr']); gave_confincorrvscorr.data = toverparam(data_t['confidence_incorrvscorr'])

gave_confcorr.plot_joint(picks = 'eeg', title = 'confidence - correct',
                            topomap_args = dict(contours=0, outlines='head', vmin = -4, vmax=4, scalings = dict(eeg=1), units = 'tstat'),
                            ts_args = dict(unit = False, ylim = dict(eeg=[-6,6]), units = 'tstat'))

gave_confincorr.plot_joint(picks = 'eeg',title = 'confidence - incorrect',
                              topomap_args = dict(contours=0, outlines='head', vmin = -4, vmax=4, scalings = dict(eeg=1), units = 'tstat'),
                              ts_args = dict(unit = False, ylim = dict(eeg=[-6,6]), units = 'tstat'))

gave_confincorrvscorr.plot_joint(picks = 'eeg',title = 'confidence - incorrect vs correct',
                                 ts_args = dict(unit = False, ylim = dict(eeg=[-6,6]), units = 'tstat'))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(alltimes, deepcopy(gave_confcorr).pick_channels(['FCZ']).data[0], label = 'confidence - correct', color = '#2ca25f', lw = 1)
ax.plot(alltimes, deepcopy(gave_confincorr).pick_channels(['FCZ']).data[0], label = 'confidence - incorrect', color = '#2ca25f', ls = 'dashed', lw = 1)
ax.plot(alltimes, deepcopy(gave_confincorrvscorr).pick_channels(['FCZ']).data[0], label = 'confidence - incorrect vs correct', color = '#3182bd', lw = 1)

ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), linestyles = 'dashed', lw = .75, color = '#000000')
ax.vlines(x = 0, ymin = -4, ymax = 4, linestyles = 'dashed', color = '#000000', lw = .75)
fig.legend(loc = 'upper right')



#%%

#gave_pside = mne.grand_average(data_t['pside']); gave_pside.data = toverparam(data_t['pside'])
#gave_pside.plot_joint(picks = 'eeg',
#                      topomap_args = dict(contours=0, outlines='head', vmin = -3, vmax=3, scalings = dict(eeg=1), units = 'tstat'),
#                      ts_args = dict(unit = False, ylim = dict(eeg=[-6,6]), units = 'tstat'))
#%%
#nonpara cluster t test to see where diff is significant
tmin, tmax = 0.0, 1.0 #specify time window for the cluster test to work
X_diff = np.empty(shape = (len(subs), 1, deepcopy(gave_confincorrvscorr).crop(tmin=tmin, tmax=tmax).times.size))
for i in range(len(data_t['confidence_incorrvscorr'])):
    tmp = deepcopy(data_t['confidence_incorrvscorr'][i])
    tmp.pick_channels(['FCZ'])
    tmp.crop(tmin = tmin, tmax = tmax) #take only first 600ms for cluster test, time window for ERN and PE components
    X_diff[i,:,:] = tmp.data
np.random.seed(seed=1)
t_diff, clusters_diff, cluster_pv_diff, H0_diff = mne.stats.permutation_cluster_1samp_test(X_diff, out_type = 'indices')
mask_diff_05 = np.asarray(clusters_diff)[cluster_pv_diff<0.05]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(alltimes, deepcopy(gave_confcorr).pick_channels(['FCZ']).data[0], label = 'confidence - correct', color = '#2ca25f', lw = 1.5)
ax.plot(alltimes, deepcopy(gave_confincorr).pick_channels(['FCZ']).data[0], label = 'confidence - incorrect', color = '#2ca25f', ls = 'dashed', lw = 1.5)
ax.plot(alltimes, deepcopy(gave_confincorrvscorr).pick_channels(['FCZ']).data[0], label = 'confidence - incorrect vs correct', color = '#3182bd', ls = 'dashed', lw = 1.5)
ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), linestyles = 'dashed', lw = .75, color = '#000000')
ax.vlines(x = 0, ymin = -4, ymax = 4, linestyles = 'dashed', color = '#000000', lw = .75)
fig.legend(loc = 'upper right')
for mask in range(len(mask_diff_05)):
    ax.hlines(y = -4,
              xmin = np.min(times[mask_diff_05[mask][1]]),
              xmax = np.max(times[mask_diff_05[mask][1]]),
              lw=5, color = '#3182bd', alpha = .5) #plot significance timepoints for difference effect

fig1 = gave_confincorrvscorr.plot_joint(picks = 'eeg',
                                    topomap_args = dict(contours=0, outlines='head', vmin = -3, vmax=3, scalings = dict(eeg=1), units = 'tstat'),
                                    ts_args = dict(unit = False, ylim = dict(eeg=[-5,5]), units = 'tstat'))
ax1 = fig1.axes[0]
for mask in range(len(mask_diff_05)):
    #x =  times[mask_diff_05[mask][1]]
    #y = np.zeros(len(x)); y.fill(-5.3)
    #ax.scatter(x, y, color = '#998ec3', alpha = .5, marker='s')
    ax1.hlines(y = -4,
              xmin = np.min(times[mask_diff_05[mask][1]]),
              xmax = np.max(times[mask_diff_05[mask][1]]),
              lw=5, color = '#3182bd', alpha = .5) #plot significance timepoints for difference effect
#%%

fig = plt.figure(figsize = (10,7))
ax  = fig.add_subplot(311)
ax.set_title('correct trials')
ax.plot(alltimes, deepcopy(gave_errorcorr).pick_channels(['FCZ']).data[0], label = 'error', color = '#d7191c', lw = 1)
ax.plot(alltimes, deepcopy(gave_confcorr).pick_channels(['FCZ']).data[0],  label = 'confidence', color = '#2c7bb6', lw = 1)
ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
ax.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax.set_ylabel('t-value')
ax.set_ylim([-6,3.5])

ax2 = fig.add_subplot(312)
ax2.set_title('incorrect trials')
ax2.plot(alltimes, deepcopy(gave_errorincorr).pick_channels(['FCZ']).data[0], color = '#d7191c', lw = 1)
ax2.plot(alltimes, deepcopy(gave_confincorr).pick_channels(['FCZ']).data[0], color = '#2c7bb6', lw = 1)
ax2.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
ax2.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax2.set_ylabel('t-value')
ax2.set_ylim([-4,4.5])

ax3 = fig.add_subplot(313)
ax3.set_title('incorrect-correct trials')
ax3.plot(alltimes, deepcopy(gave_error_incorrvscorr).pick_channels(['FCZ']).data[0], color = '#d7191c', lw = 1)
ax3.plot(alltimes, deepcopy(gave_confincorrvscorr).pick_channels(['FCZ']).data[0], color = '#2c7bb6', lw = 1)
ax3.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
ax3.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax3.set_ylabel('t-value')
ax3.set_ylim([-4,4.5])
ax3.set_xlabel('Time relative to feedback onset (s)')
fig.legend(loc = 'upper left')

plt.tight_layout()

#%%
np.random.seed(seed=1)
tmin, tmax = 0, 1
smooth_sigma = 2
dat2use = deepcopy(data_t)

gave_errorcorr = mne.grand_average(dat2use['error_corr']);  gave_errorcorr.data = toverparam(dat2use['error_corr'])
gave_errorincorr = mne.grand_average(dat2use['error_incorr']);gave_errorincorr.data = toverparam(dat2use['error_incorr'])
gave_error = mne.grand_average(dat2use['error']);gave_error.data = toverparam(dat2use['error'])
gave_error_incorrvscorr = mne.grand_average(dat2use['error_incorrvscorr']);gave_error_incorrvscorr.data = toverparam(dat2use['error_incorrvscorr'])

gave_confcorr = mne.grand_average(dat2use['conf_corr']);gave_confcorr.data = toverparam(dat2use['conf_corr'])
gave_confincorr = mne.grand_average(dat2use['conf_incorr']);gave_confincorr.data = toverparam(dat2use['conf_incorr'])
gave_conf = mne.grand_average(dat2use['confidence']);gave_conf.data = toverparam(dat2use['confidence'])
gave_confincorrvscorr = mne.grand_average(dat2use['confidence_incorrvscorr']);gave_confincorrvscorr.data = toverparam(dat2use['confidence_incorrvscorr'])


t_errcorr, clusters_errcorr, clusters_pv_errcorr, _          = runclustertest_epochs(data = dat2use, contrast_name = 'error_corr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_errcorr = np.asarray(clusters_errcorr)[clusters_pv_errcorr < 0.05]

t_confcorr, clusters_confcorr, clusters_pv_confcorr, _       = runclustertest_epochs(data = dat2use, contrast_name = 'conf_corr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_confcorr = np.asarray(clusters_confcorr)[clusters_pv_confcorr < 0.05]


t_errincorr, clusters_errincorr, clusters_pv_errincorr, _    = runclustertest_epochs(data = dat2use, contrast_name = 'error_incorr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_errincorr = np.asarray(clusters_errincorr)[clusters_pv_errincorr < 0.05]

t_confincorr, clusters_confincorr, clusters_pv_confincorr, _ = runclustertest_epochs(data = dat2use, contrast_name = 'conf_incorr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_confincorr = np.asarray(clusters_confincorr)[clusters_pv_confincorr < 0.05]

t_errivsc, clusters_errivsc, clusters_pv_errivsc, _          = runclustertest_epochs(data = dat2use, contrast_name = 'error_incorrvscorr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_errivsc = np.asarray(clusters_errivsc)[clusters_pv_errivsc < 0.05]

t_confivsc, clusters_confivsc, clusters_pv_confivsc, _       = runclustertest_epochs(data = dat2use, contrast_name = 'confidence_incorrvscorr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_confivsc = np.asarray(clusters_confivsc)[clusters_pv_confivsc < 0.05]

t_err, clusters_err, clusters_pv_err, _                 = runclustertest_epochs(data = dat2use, contrast_name = 'error', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_err = np.asarray(clusters_err)[clusters_pv_err < 0.05]

t_conf, clusters_conf, clusters_pv_conf, _              = runclustertest_epochs(data = dat2use, contrast_name = 'confidence', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_conf = np.asarray(clusters_conf)[clusters_pv_conf < 0.05]

times_ern = deepcopy(data_t['error_corr'][0]).crop(tmin=tmin, tmax=tmax).times

fig = plt.figure(figsize = (12,8))
ax  = fig.add_subplot(413)
ax.set_title('correct trials')
ax.plot(alltimes, deepcopy(gave_errorcorr).pick_channels(['FCZ']).data[0], label = 'error', color = '#d7191c', lw = 1)
ax.plot(alltimes, deepcopy(gave_confcorr).pick_channels(['FCZ']).data[0],  label = 'confidence', color = '#2c7bb6', lw = 1)
ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
#ax.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax.set_ylabel('t-value')
#ax.set_ylim([-6,6])
for mask in masks_errcorr:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax.hlines(y = 0, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#d7191c')
for mask in masks_confcorr:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax.hlines(y = 0, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')

ax2 = fig.add_subplot(414)
ax2.set_title('incorrect trials')
ax2.plot(alltimes, deepcopy(gave_errorincorr).pick_channels(['FCZ']).data[0], color = '#d7191c', lw = 1)
ax2.plot(alltimes, deepcopy(gave_confincorr).pick_channels(['FCZ']).data[0], color = '#2c7bb6', lw = 1)
ax2.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
#ax2.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax2.set_ylabel('t-value')
#ax2.set_ylim([-6,6])
for mask in masks_errincorr:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax2.hlines(y = 0, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#d7191c')
for mask in masks_confincorr:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax2.hlines(y = 0, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')

ax3 = fig.add_subplot(412)
ax3.set_title('incorrect-correct trials')
ax3.plot(alltimes, deepcopy(gave_error_incorrvscorr).pick_channels(['FCZ']).data[0], color = '#d7191c', lw = 1)
ax3.plot(alltimes, deepcopy(gave_confincorrvscorr).pick_channels(['FCZ']).data[0], color = '#2c7bb6', lw = 1)
ax3.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
#ax3.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax3.set_ylabel('t-value')
#ax3.set_ylim([-6,6])
ax3.set_xlabel('Time relative to feedback onset (s)')
for mask in masks_errivsc:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax3.hlines(y = 0, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#d7191c')
for mask in masks_confivsc:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax3.hlines(y = 0, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')
               
#gave_error = mne.grand_average(data_t['error']); gave_error.data = toverparam(data_t['error'])
#gave_conf  = mne.grand_average(data_t['confidence']); gave_conf.data = toverparam(data_t['confidence'])

ax4 = fig.add_subplot(411)
ax4.set_title('all trials')
ax4.plot(alltimes, deepcopy(gave_error).pick_channels(['FCZ']).data[0], color = '#d7191c', lw = 1)
ax4.plot(alltimes, deepcopy(gave_conf).pick_channels(['FCZ']).data[0], color = '#2c7bb6', lw = 1)
ax4.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
#ax4.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax4.set_ylabel('t-value')
#ax4.set_ylim([-6,6])
ax4.set_xlabel('Time relative to feedback onset (s)')
for mask in masks_err:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax4.hlines(y = 0, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#d7191c')
for mask in masks_conf:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax4.hlines(y = 0, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')

fig.legend(loc = 'upper left')
plt.tight_layout()

for axis in [ax, ax2, ax3, ax4]:
    axis.xaxis.set_ticks(np.arange(-.5, 1.51, .1))
#fig.savefig(fname = op.join(figpath, 'errorconf_FCZ_sepbytrialtypes_%02dsubs.eps'%(subs.size)), format = 'eps', dpi = 300, bbox_inches='tight')
#fig.savefig(fname = op.join(figpath, 'errorconf_FCZ_sepbytrialtypes_%02dsubs.pdf'%(subs.size)), format = 'pdf', dpi = 300, bbox_inches='tight')


#%%
#this is the same as above really (i.e. plots are the same, but restricting the permutation tests to a different time region, locked to feedback offset)

np.random.seed(seed=1)
tmin, tmax = 0.5, 1.0
smooth_sigma = None
betas = False
if betas:
    dat2use = deepcopy(data)
    errcorr = mne.grand_average(data['error_corr']); errcorr.data = toverparam(data['error_corr'])
    confcorr = mne.grand_average(data['conf_corr']); confcorr.data = toverparam(data['conf_corr'])
    
    errincorr = mne.grand_average(data['error_incorr']); errincorr.data = toverparam(data['error_incorr'])
    confincorr = mne.grand_average(data['conf_incorr']); confincorr.data = toverparam(data['conf_incorr'])
    
    errivsc = mne.grand_average(data['error_incorrvscorr']); errivsc.data = toverparam(data['error_incorrvscorr'])
    confivsc = mne.grand_average(data['confidence_incorrvscorr']); confivsc.data = toverparam(data['confidence_incorrvscorr'])
    
    err = mne.grand_average(data['error']); err.data = toverparam(data['error'])
    conf = mne.grand_average(data['confidence']); conf.data = toverparam(data['confidence'])
else:
    dat2use = deepcopy(data_t)
    errcorr = mne.grand_average(data_t['error_corr']); errcorr.data = toverparam(data_t['error_corr'])
    confcorr = mne.grand_average(data_t['conf_corr']); confcorr.data = toverparam(data_t['conf_corr'])
    
    errincorr = mne.grand_average(data_t['error_incorr']); errincorr.data = toverparam(data_t['error_incorr'])
    confincorr = mne.grand_average(data_t['conf_incorr']); confincorr.data = toverparam(data_t['conf_incorr'])
    
    errivsc = mne.grand_average(data_t['error_incorrvscorr']); errivsc.data = toverparam(data_t['error_incorrvscorr'])
    confivsc = mne.grand_average(data_t['confidence_incorrvscorr']); confivsc.data = toverparam(data_t['confidence_incorrvscorr'])
    
    err = mne.grand_average(data_t['error']); err.data = toverparam(data_t['error'])
    conf = mne.grand_average(data_t['confidence']); conf.data = toverparam(data_t['confidence'])

t_errcorr, clusters_errcorr, clusters_pv_errcorr, _          = runclustertest_epochs(data = dat2use, contrast_name = 'error_corr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_errcorr = np.asarray(clusters_errcorr)[clusters_pv_errcorr < 0.05]

t_confcorr, clusters_confcorr, clusters_pv_confcorr, _       = runclustertest_epochs(data = dat2use, contrast_name = 'conf_corr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_confcorr = np.asarray(clusters_confcorr)[clusters_pv_confcorr < 0.05]


t_errincorr, clusters_errincorr, clusters_pv_errincorr, _    = runclustertest_epochs(data = dat2use, contrast_name = 'error_incorr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_errincorr = np.asarray(clusters_errincorr)[clusters_pv_errincorr < 0.05]

t_confincorr, clusters_confincorr, clusters_pv_confincorr, _ = runclustertest_epochs(data = dat2use, contrast_name = 'conf_incorr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_confincorr = np.asarray(clusters_confincorr)[clusters_pv_confincorr < 0.05]

t_errivsc, clusters_errivsc, clusters_pv_errivsc, _          = runclustertest_epochs(data = dat2use, contrast_name = 'error_incorrvscorr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_errivsc = np.asarray(clusters_errivsc)[clusters_pv_errivsc < 0.05]

t_confivsc, clusters_confivsc, clusters_pv_confivsc, _       = runclustertest_epochs(data = dat2use, contrast_name = 'confidence_incorrvscorr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_confivsc = np.asarray(clusters_confivsc)[clusters_pv_confivsc < 0.05]

t_err, clusters_err, clusters_pv_err, _                 = runclustertest_epochs(data = dat2use, contrast_name = 'error', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_err = np.asarray(clusters_err)[clusters_pv_err < 0.05]

t_conf, clusters_conf, clusters_pv_conf, _              = runclustertest_epochs(data = dat2use, contrast_name = 'confidence', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_conf = np.asarray(clusters_conf)[clusters_pv_conf < 0.05]

times_ern = deepcopy(data_t['error_corr'][0]).crop(tmin=tmin, tmax=tmax).times

fig = plt.figure(figsize = (10,7))
ax  = fig.add_subplot(411)
ax.set_title('correct trials')
ax.plot(alltimes, deepcopy(errcorr).pick_channels(['FCZ']).data[0], label = 'error', color = '#d7191c', lw = 1)
ax.plot(alltimes, deepcopy(confcorr).pick_channels(['FCZ']).data[0],  label = 'confidence', color = '#2c7bb6', lw = 1)
ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
ax.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax.set_ylabel('t-value')
ax.set_ylim([-6,6])
for mask in masks_errcorr:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#d7191c')
for mask in masks_confcorr:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')

ax2 = fig.add_subplot(412)
ax2.set_title('incorrect trials')
ax2.plot(alltimes, deepcopy(errincorr).pick_channels(['FCZ']).data[0], color = '#d7191c', lw = 1)
ax2.plot(alltimes, deepcopy(confincorr).pick_channels(['FCZ']).data[0], color = '#2c7bb6', lw = 1)
ax2.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
ax2.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax2.set_ylabel('t-value')
ax2.set_ylim([-6,6])
for mask in masks_errincorr:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax2.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#d7191c')
for mask in masks_confincorr:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax2.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')

ax3 = fig.add_subplot(413)
ax3.set_title('incorrect-correct trials')
ax3.plot(alltimes, deepcopy(errivsc).pick_channels(['FCZ']).data[0], color = '#d7191c', lw = 1)
ax3.plot(alltimes, deepcopy(confivsc).pick_channels(['FCZ']).data[0], color = '#2c7bb6', lw = 1)
ax3.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
ax3.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax3.set_ylabel('t-value')
ax3.set_ylim([-6,6])
ax3.set_xlabel('Time relative to feedback onset (s)')
for mask in masks_errivsc:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax3.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#d7191c')
for mask in masks_confivsc:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax3.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')
               
ax4 = fig.add_subplot(414)
ax4.set_title('all trials')
ax4.plot(alltimes, deepcopy(err).pick_channels(['FCZ']).data[0], color = '#d7191c', lw = 1)
ax4.plot(alltimes, deepcopy(conf).pick_channels(['FCZ']).data[0], color = '#2c7bb6', lw = 1)
ax4.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
ax4.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax4.set_ylabel('t-value')
ax4.set_ylim([-6,6])
ax4.set_xlabel('Time relative to feedback onset (s)')
for mask in masks_err:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax4.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#d7191c')
for mask in masks_conf:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax4.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')

fig.legend(loc = 'upper left')
plt.tight_layout()
#%%
gave_confupdate = mne.grand_average(data_t['confupdate']); gave_confupdate.data = toverparam(data_t['confupdate'])
gave_confupdate_corr = mne.grand_average(data_t['confupdate_corr']); gave_confupdate.data = toverparam(data_t['confupdate_corr'])
gave_confupdate_incorr = mne.grand_average(data_t['confupdate_incorr']); gave_confupdate_incorr.data = toverparam(data_t['confupdate_incorr'])
gave_confupdate_incorrvscorr = mne.grand_average(data_t['confupdate_incorrvscorr']); gave_confupdate_incorrvscorr.data = toverparam(data_t['confupdate_incorrvscorr'])


np.random.seed(seed=1)
tmin, tmax = None, None
smooth_sigma = 8

t_confupdatecorr, clusters_confupdatecorr, clusters_pv_confupdatecorr, _          = runclustertest_epochs(data = data_t, contrast_name = 'confupdate_corr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_confupdatecorr = np.asarray(clusters_confupdatecorr)[clusters_pv_confupdatecorr < 0.05]

t_confupdateincorr, clusters_confupdateincorr, clusters_pv_confupdateincorr, _    = runclustertest_epochs(data = data_t, contrast_name = 'confupdate_incorr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_confupdateincorr = np.asarray(clusters_confupdateincorr)[clusters_pv_confupdateincorr < 0.05]

t_confupdateivsc, clusters_confupdateivsc, clusters_pv_confupdateivsc, _          = runclustertest_epochs(data = data_t, contrast_name = 'confupdate_incorrvscorr', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_confupdateivsc = np.asarray(clusters_confupdateivsc)[clusters_pv_confupdateivsc < 0.05]

t_confupdate, clusters_confupdate, clusters_pv_confupdate, _                 = runclustertest_epochs(data = data_t, contrast_name = 'confupdate', channels = ['FCZ'], tmin = tmin, tmax = tmax, gauss_smoothing = smooth_sigma) 
masks_confupdate = np.asarray(clusters_confupdate)[clusters_pv_confupdate < 0.05]

times_ern = deepcopy(data_t['confupdate'][0]).crop(tmin=tmin, tmax=tmax).times

fig = plt.figure(figsize = (12,8))
ax  = fig.add_subplot(111)
ax.set_title('correct trials')
ax.plot(times_ern, np.squeeze(t_confupdatecorr), label = 'confupdate corr', color = '#d7191c', lw = 1)
ax.plot(times_ern, np.squeeze(t_confupdateincorr),  label = 'confupdate incorr', color = '#2c7bb6', lw = 1)
ax.plot(times_ern, np.squeeze(t_confupdate),  label = 'confupdate', color = '#756bb1', lw = 1)
ax.plot(times_ern, np.squeeze(t_confupdateivsc),  label = 'confupdate ivsc', color = '#000000', lw = 1)
ax.hlines(y = 0, xmin = alltimes.min(), xmax = alltimes.max(), lw = .75, linestyles = 'dashed', color = '#000000')
ax.vlines(x = 0, ymin = -5, ymax = 6, linestyles = 'dashed', color = '#000000', lw = .75)
ax.set_ylabel('t-value')
ax.set_ylim([-6,6])
for mask in masks_confupdatecorr:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax.hlines(y = -5, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#d7191c')
for mask in masks_confupdatecorr:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax.hlines(y = -5.3, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')
for mask in masks_confupdate:
    start, stop = times_ern[mask[1]].min(), times_ern[mask[1]].max()
    ax.hlines(y = -5.3, xmin=start, xmax=stop, lw = 3, alpha=.5, color = '#2c7bb6')

fig.legend(loc = 'upper left')
plt.tight_layout()
#%%
gave_confupdate_corr.plot_joint(picks='eeg', topomap_args=dict(outlines='head', contours=0),
                                ts_args = dict(unit = False, ylim = dict(eeg=[-2,2]), units = 'tstat'))

gave_confupdate_incorr.plot_joint(picks='eeg', topomap_args=dict(outlines='head', contours=0),
                                ts_args = dict(unit = False, ylim = dict(eeg=[-5,5]), units = 'tstat'))

gave_confupdate_incorrvscorr.plot_joint(picks='eeg', topomap_args=dict(outlines='head', contours=0),
                                ts_args = dict(unit = False, ylim = dict(eeg=[-5,5]), units = 'tstat'))
#gave_confupdate.plot_joint(picks='eeg', topomap_args = dict(outlines='head', contours=0), scale='eeg')



#%%
#if you want to see what the smoothing does to individual subject traces -- it basically just gets rid of some of the weird jittering noise cos of the sample rate
plot_sub = True
for isub in range(nsubs):
    sigma = 5
    if plot_sub:
        tmp = np.squeeze(deepcopy(data_t['incorrvscorr'][isub]).pick_channels(['FCZ']).data)
        #tmp2 = np.squeeze(deepcopy(data_t['incorrvscorr'][isub]).resample(100).pick_channels(['FCZ']).data)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(deepcopy(data_t['incorrvscorr'][isub]).times, tmp, lw =1, label = 'raw tstat', color = '#000000', ls='dashed')
        ax.plot(deepcopy(data_t['incorrvscorr'][isub]).times, sp.ndimage.gaussian_filter1d(tmp, sigma=sigma), lw = 1, label = 'smoothed', color = '#d7191c')
        #ax.plot(deepcopy(data_t['incorrvscorr'][isub]).resample(100).times, tmp2, lw = 1, label = 'resampled', color = '#d7191c')
        ax.hlines(y=0, xmin = deepcopy(data_t['incorrvscorr'][isub]).times.min(), xmax=deepcopy(data_t['incorrvscorr'][isub]).times.max(), lw=1, linestyles='dashed', color='#000000')
        ax.vlines(x=0, ymin=-2, ymax=4, lw=1, linestyles='dashed', color='#000000')
        fig.legend(loc='upper right')

