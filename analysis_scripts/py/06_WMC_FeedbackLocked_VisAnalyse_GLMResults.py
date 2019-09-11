#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:11:42 2019

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

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, smooth

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
subs = np.array([1,       4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm
subs = np.array([1,       4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

alldata_grandmean       = []
alldata_neutral         = []
alldata_cued            = []
alldata_error           = []
alldata_conf            = []
alldata_underconf       = []
alldata_cvsn            = []

alldata_grandmean_t       = []
alldata_neutral_t         = []
alldata_cued_t            = []
alldata_error_t           = []
alldata_conf_t            = []
alldata_underconf_t       = []
alldata_cvsn_t            = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    alldata_grandmean.append(mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_grandmean_betas_baselined-tfr.h5'))[0])
    alldata_neutral.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_neutral_betas_baselined-tfr.h5'))[0])
    alldata_cued.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_cued_betas_baselined-tfr.h5'))[0])
    alldata_error.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_error_betas_baselined-tfr.h5'))[0])
    alldata_conf.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_confidence_betas_baselined-tfr.h5'))[0])
    alldata_underconf.append(mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_confdiffunderconf_betas_baselined-tfr.h5'))[0])
    alldata_cvsn.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_cuedvsneutral_betas-tfr.h5'))[0])
    
    
    alldata_grandmean_t.append(mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_grandmean_tstats_baselined-tfr.h5'))[0])
    alldata_neutral_t.append(mne.time_frequency.read_tfrs(fname   = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_neutral_tstats_baselined-tfr.h5'))[0])
    alldata_cued_t.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_cued_tstats_baselined-tfr.h5'))[0])
    alldata_error_t.append(mne.time_frequency.read_tfrs(fname     = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_error_tstats_baselined-tfr.h5'))[0])
    alldata_conf_t.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_confidence_tstats_baselined-tfr.h5'))[0])
    alldata_underconf_t.append(mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_confdiffunderconf_tstats_baselined-tfr.h5'))[0])
    alldata_cvsn_t.append(mne.time_frequency.read_tfrs(fname      = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_cuedvsneutral_tstats-tfr.h5'))[0])

#%%
    
timefreqs_all = {(.4, 10):(.4, 4),
                 (.6, 10):(.4, 4),
                 (.8, 10):(.4, 4),
                 (.4, 22):(.4, 16),
                 (.6, 22):(.4, 16),
                 (.8, 22):(.4, 16)}

timefreqs_alpha = {(.6, 10):(.2, 4),
                   (.7, 10):(.2, 4),
                   (.8, 10):(.2, 4),
                   (.9, 10):(.2, 4)}   

timefreqs_beta = {(.6, 22):(.2, 16),
                  (.7, 22):(.2, 16),
                  (.8, 22):(.2, 16),
                  (.9, 22):(.2, 16)}   


#going to visualise these regressors now just to look at effects

gave_gmean = mne.grand_average(alldata_grandmean); gave_gmean.drop_channels(['RM'])
gave_gmean.plot_joint(title = 'average feedback evoked response, average betas, preglm baseline', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0))
    
gave_gmean = mne.grand_average(alldata_grandmean); gave_gmean.data = toverparam(alldata_grandmean); gave_gmean.drop_channels(['RM'])
gave_gmean.plot_joint(title = 'average feedback evoked response, t over betas, preglm baseline', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -4, vmax = 4))

gave_gmean_t = mne.grand_average(alldata_grandmean_t); gave_gmean_t.data = toverparam(alldata_grandmean_t); gave_gmean_t.drop_channels(['RM'])
gave_gmean_t.plot_joint(title = 'average feedback evoked response, t over tstats, preglm baselined', timefreqs = timefreqs_alpha,
                        topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

plot_subs = True
# plot single subjects ...
if plot_subs:
    for i in range(len(alldata_grandmean)):
        tmp = deepcopy(alldata_grandmean[i])
        tmp.drop_channels(['RM']).plot_joint(title = 'subject %d ave feedback response, average betas'%(subs[i]), timefreqs = timefreqs_alpha,
                         topomap_args = dict(outlines = 'head', contours = 0))



# neutral trials
gave_neut = mne.grand_average(alldata_neutral); gave_neut.drop_channels(['RM'])
gave_neut.plot_joint(title = 'neutral trials, average betas, preglm baseline', timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0))

gave_cued = mne.grand_average(alldata_cued); gave_cued.drop_channels(['RM'])
gave_cued.plot_joint(title = 'cued trials, average betas, preglm baseline', timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0))

gave_cvsn = mne.grand_average(alldata_cvsn); gave_cvsn.drop_channels(['RM'])
gave_cvsn.plot_joint(title = 'cued vs neutral trials, average betas', timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0))#, tmin=-0.3, tmax = 1.3)

gave_cvsn = mne.grand_average(alldata_cvsn); gave_cvsn.data = toverparam(alldata_cvsn); gave_cvsn.drop_channels(['RM'])
gave_cvsn.plot_joint(title = 'cued vs neutral trials, t over betas', timefreqs = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))
    
    
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

gave_error = mne.grand_average(alldata_error); gave_error.drop_channels(['RM'])
gave_error.plot_joint(title = 'main effect of error, average betas, preglm baseline', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0))

plot_subs = False
if plot_subs:
    # plot single subjects ...
    from copy import deepcopy
    for i in range(len(alldata_error)):
        tmp = deepcopy(alldata_error[i])
        tmp.drop_channels(['RM']).plot_joint(title = 'subject %d main effect error, betas'%(subs[i]), timefreqs = timefreqs_alpha,
                         topomap_args = dict(outlines = 'head', contours = 0))


gave_error = mne.grand_average(alldata_error); gave_error.data = toverparam(alldata_error); gave_error.drop_channels(['RM'])
gave_error.plot_joint(title = 'main effect of error, t over betas, preglm baseline', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin=-2, vmax = 2))

gave_error_t = mne.grand_average(alldata_error_t); gave_error_t.data = toverparam(alldata_error_t); gave_error_t.drop_channels(['RM'])
gave_error_t.plot_joint(title = 'main effect of error, t over tstats, preglm baseline', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))
#lower evoked centro-parietal alpha power associated with larger error

#confidence width now
gave_conf = mne.grand_average(alldata_conf); gave_conf.data = toverparam(alldata_conf); gave_conf.drop_channels(['RM'])
gave_conf.plot_joint(title = 'main effect of confidence, t over betas, preglm baseline', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))

gave_conf_t = mne.grand_average(alldata_conf_t); gave_conf_t.data = toverparam(alldata_conf_t); gave_conf_t.drop_channels(['RM'])
gave_conf_t.plot_joint(title = 'main effect of confidence, t over tstats, preglm baseline', timefreqs = timefreqs_alpha,
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))
#lower alpha power (slightly later on than in error effect) is associated with higher confidence

gave_underconf_t = mne.grand_average(alldata_underconf_t); gave_underconf_t.data = toverparam(alldata_underconf_t); gave_underconf_t.drop_channels(['RM'])
gave_underconf_t.plot_joint(title = 'main effect of error awareness (confdiff) on underconfident trials, t over tstats, preglm baseline',
                            timefreqs = timefreqs_alpha,
                            topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))
#the sign of this regressor is flipped, it indicates metacognitive error (i.e. error related signal)
#-ve tstat = -ve relationship
#lower alpha power is associated with higher metacognitive error (i.e. WORSE error awareness during the trial)

allvis_chans = np.array(['P7', 'P5', 'P3','P1','PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2'])
vis_chans = np.array(['PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2'])
#plot tfr images of just the visual channels

gave_error_t.plot(      picks = vis_chans, baseline = None, mode = 'mean', vmin = -2, vmax = 2, combine = 'mean',
                  title = 'main effect of error across posterior channels (8)')
gave_conf_t.plot(       picks = vis_chans, baseline = None, mode = 'mean', vmin = -2, vmax = 2, combine = 'mean',
                  title = 'main effect of confidence across posterior channels (8)')
gave_underconf_t.plot(  picks = vis_chans, baseline = None, mode = 'mean', vmin = -2, vmax = 2, combine = 'mean',
                      title = 'main effect of confidence interval deviation across posterior channels (8)')

#plot tfr images of the frontal channels
frontal_chans = np.array(['AFZ', 'AF3', 'AF4', 'FZ', 'FPZ'])

gave_error_t.plot(      picks = frontal_chans, baseline = None, mode = 'mean', vmin = -2, vmax = 2, combine = 'mean',
                  title = 'main effect of error across frontal channels (5)')
gave_conf_t.plot(       picks = frontal_chans, baseline = None, mode = 'mean', vmin = -2, vmax = 2, combine = 'mean',
                  title = 'main effect of confidence across frontal channels (5)')
gave_underconf_t.plot(  picks = frontal_chans, baseline = None, mode = 'mean', vmin = -2, vmax = 2, combine = 'mean',
                      title = 'main effect of confidence interval deviation across frontal channels (5)')


# - - - - -- - - - - 

#try some cluster stats?

connectivity, ch_names = mne.channels.find_ch_connectivity(gave_error_t.info, ch_type = 'eeg')

plt.imshow(connectivity.toarray(), cmap = 'gray', origin = 'lower', interpolation = 'nearest')
plt.xlabel('{} EEG sensors'.format(len(ch_names)))
plt.ylabel('{} EEG sensors'.format(len(ch_names)))
plt.title('Between-sensor adjacency')

threshold = 2.5
n_perms = 100

#create array of data for this
nsubs = len(subs)
tfdata = np.zeros(shape = (nsubs, 39, 200)); tfdata.fill(np.nan) #fill with nans #for all frequencies
#tfdata = np.zeros(shape = (nsubs, 25, 200)); tfdata.fill(np.nan) #for just above 15Hz frequencies
for i in range(len(alldata_underconf_t)):
    tmp = deepcopy(alldata_underconf_t[i]); tmp.drop_channels(['RM'])
    tmp.pick_channels(frontal_chans) #select only these posterior channels
    #tmp.crop(fmin=15)
    tmp = tmp.data
    tmp = np.squeeze(np.nanmean(tmp.data, 0)) #average across these channels
    tfdata[i,:,:] = tmp
    
tfdata.shape #(nsubs, times, frequencies) #we averaged across some channels, so lose that dimension


freqs = alldata_conf_t[0].freqs
pwrdata = deepcopy(tfdata)
fmin = 10
pwrdata = pwrdata[:,np.greater_equal(alldata_conf_t[0].freqs, fmin),:]

t_alpha, clusters_alpha, cluster_pv_alpha, H0_alpha = \
    mne.stats.permutation_cluster_1samp_test(pwrdata,
                                             out_type = 'mask')

mask_alpha_05 = np.asarray(clusters_alpha)[cluster_pv_alpha<0.2]

mask_allfreqs = np.zeros(shape = (mask_alpha_05.shape[0], len(freqs), 200), dtype = 'bool')
mask_allfreqs[:,np.greater_equal(freqs,fmin),:] = mask_alpha_05

pwr_avg = toverparam(tfdata)

tmp = deepcopy(alldata_error[0]); tmp.pick_channels(vis_chans);
tmp.data = pwr_avg

plt.contourf(tmp.data, cmap = 'RdBu_r', origin = 'lower', vmin = -2.5, vmax = 2.5, levels = 64)
plt.colorbar()
for i in range(mask_allfreqs.shape[0]):
    plt.contour(np.squeeze(mask_allfreqs[i,:,:]), colors = 'black', linewidths = .75, corner_mask = False, antialiased = False)

gave_underconf_t.plot(picks = frontal_chans, vmin = -2.5, vmax = 2.5,cmap = 'RdBu_r', combine = 'mean',
                 mask = np.squeeze(mask_allfreqs), mask_style='both', mask_alpha=.9, mask_cmap = 'RdBu_r')

gave_underconf_t.plot_joint(title = 'main effect of confidence interval deviation, t over tstats, preglm baseline',
                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -2, vmax = 2))


times = gave_error_t.times

gave_error_t = gave_error_t.crop(fmin = 8, fmax = 12).pick_channels(allvis_chans)
gave_conf_t  = gave_conf_t.crop(fmin = 8, fmax = 12).pick_channels(allvis_chans)
gave_underconf_t = gave_underconf_t.crop(fmin = 8, fmax = 12).pick_channels(allvis_chans)

#gave_error_t.data.shape = (17,5,200) #channels, frequencies, time
#need to collapse across channels and frequencies
gave_error_t = np.squeeze(np.nanmean(gave_error_t.data, 0))
gave_error_t = np.squeeze(np.nanmean(gave_error_t, 0))

gave_conf_t = np.squeeze(np.nanmean(gave_conf_t.data, 0))
gave_conf_t = np.squeeze(np.nanmean(gave_conf_t, 0))

gave_underconf_t = np.squeeze(np.nanmean(gave_underconf_t.data, 0))
gave_underconf_t = np.squeeze(np.nanmean(gave_underconf_t, 0))


filt = sp.signal.windows.boxcar(20) #samples to smooth

#smooth these signals a bit if you need
smoothing = True
if smoothing:
    gave_error_t = np.convolve(filt/filt.sum(), gave_error_t, mode = 'same')
    gave_conf_t = np.convolve(filt/filt.sum(), gave_conf_t, mode = 'same')
    gave_underconf_t = np.convolve(filt/filt.sum(), gave_underconf_t, mode = 'same')
    
    

plt.figure()
plt.title('8-12Hz across all visual channels')
plt.plot(times, gave_error_t, label = 'alpha power association with error')
plt.plot(times, gave_conf_t,  label = 'alpha power association with confidence')
plt.plot(times, gave_underconf_t, label = 'alpha power association with prediction error')
plt.legend()


plt.figure()
plt.plot(times, gave_error_t)
plt.plot(times, np.convolve(filt/filt.sum(), gave_error_t, mode='same'))
