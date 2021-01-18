# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:13:34 2020

@author: sammi
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

# sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
# from wmConfidence_funcs import get_subject_info_wmConfidence
# from wmConfidence_funcs import gesd, plot_AR, toverparam, smooth, runclustertest_epochs
sys.path.insert(0, 'C:\\Users\\sammi\\Desktop\\Experiments\\DPhil\\wmConfidence\\analysis_scripts')#because working from laptop to make this script

from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, smooth, runclustertest_epochs

# wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
os.chdir(wd)

subs = np.array([4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22,24, 25, 26])

glm_folder = 'epochs_glm4'
# glm_folder = 'epochs_glm5'
figpath = op.join(wd,'figures', 'eeg_figs', 'fblocked', glm_folder)
if not os.path.exists(figpath): 
    os.mkdir(figpath)


if glm_folder == 'epochs_glm5':
    contrasts = ['correct', 'incorrect', 'confdiff', 'grandmean', 'incorrvscorr']
if glm_folder == 'epochs_glm4':
    contrasts = ['correct', 'incorrect', 'confdiff_correct' , 'confdiff_incorrect', 'pside', 'confdiff_incorrvscorr', 'confdiff']
            

laplacian = False
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
    
    param = {}
    param['path'] = 'C:/Users/sammi/Desktop/Experiments/DPhil/wmConfidence/data'
    param['subid'] = 's%02d'%(i)
    sub = dict(loc = 'windows', id = i)
    
    for name in contrasts:
        data[name].append(   mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', glm_folder, 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_' + lapstr + name + '_betas-ave.fif'))[0])        
        data_t[name].append( mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', glm_folder, 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_' + lapstr + name + '_tstats-ave.fif'))[0])        

#%%
gave = mne.grand_average(data['correct']); times = gave.times;
fig = plt.figure(figsize = (9,9))
ax = fig.add_subplot(111)
gave.plot_sensors(show_names=True, axes = ax)
#fig.savefig(fname = op.join(figpath, 'sensor_locations.eps'), format = 'eps', dpi = 300)
#fig.savefig(fname = op.join(figpath, 'sensor_locations.pdf'), format = 'pdf', dpi = 300)
del(gave)
plt.close()
#%%
if glm_folder == 'epochs_glm4':
    mne.grand_average(deepcopy(data['confdiff_correct'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'error - confidence: correct trials',
                                                                       ts_args = dict(ylim = dict(eeg = [-1,1])),
                                                                       topomap_args = dict(vmin=-1, vmax=1))
    
    mne.grand_average(deepcopy(data['confdiff_incorrect'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'error - confidence: incorrect trials',
                                                                       ts_args = dict(ylim = dict(eeg = [-1.5,1.5])),
                                                                       topomap_args = dict(vmin=-1, vmax=1))
    
    mne.grand_average(deepcopy(data['confdiff_incorrvscorr'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'error - confidence: incorrect vs correct trials',
                                                                       ts_args = dict(ylim = dict(eeg = [-1.5,1.5])),
                                                                       topomap_args = dict(vmin=-1.5, vmax=1.5))
    
elif glm_folder == 'epochs_glm5':
    mne.grand_average(deepcopy(data['confdiff'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'error - confidence: all trials',
                                                                   ts_args = dict(ylim = dict(eeg = [-1.5,1.5])),
                                                                   topomap_args = dict(vmin=-1, vmax=1))

# mne.viz.plot_compare_evokeds(
#     evokeds = dict(confdiff_correct   = data['confdiff_correct'],
#                   confdiff_incorrect = data['confdiff_incorrect']),
#                   #confdiff_ivsc      = data['confdiff_incorrvscorr'],
#                   #confdiff_all       = data['confdiff']),
#     picks = 'FCz', show_sensors = False, ci = .68, truncate_xaxis = False, legend = True)


#%% for the error-confidence (metacognitive error) regressor:
        
tmin, tmax = 0,1 #set time window for the clustering

#gonna store things in a dictionary so we dont need to constantly rerun
t_cope, clu_cope, clupv_cope, h0_cope = dict(), dict(), dict(), dict()
masks_cope = dict()
for contrast in ['confdiff_correct', 'confdiff_incorrect', 'confdiff_incorrvscorr', 'confdiff']:
# for contrast in ['confdiff']:
    for dictobj in [t_cope, clu_cope, clupv_cope, h0_cope, masks_cope]:
        dictobj[contrast] = dict()
        for channel in ['FCz', 'Cz']:
            dictobj[contrast][channel] = []

clutimes = deepcopy(data['correct'][0]).crop(tmin = tmin, tmax = tmax).times
for contrast in ['confdiff_correct', 'confdiff_incorrect', 'confdiff_incorrvscorr', 'confdiff']:
# for contrast in ['confdiff']:
    for channel in ['FCz', 'Cz']:
        t_cope[contrast][channel], clu_cope[contrast][channel], clupv_cope[contrast][channel], h0_cope[contrast][channel] = runclustertest_epochs(
                    data = data,
                    contrast_name = contrast,
                    channels = [channel],
                    tmin = tmin, tmax = tmax,
                    gauss_smoothing = None,
                    out_type = 'indices', n_permutations = 5000)
        masks_cope[contrast][channel] = np.asarray(clu_cope[contrast][channel])[clupv_cope[contrast][channel] < 0.05]
        

for channel in ['FCz']:
    fig = plt.figure()
    ax = fig.subplots(1)
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    correct    = data['confdiff_correct'],
                    incorrect  = data['confdiff_incorrect'],
                    difference = data['confdiff_incorrvscorr'],
                    all_trials = data['confdiff']),
            colors = dict(
                    correct = '#4daf4a',
                    incorrect = '#e41a1c',
                    difference = '#525252',
                    all_trials = '#1f78b4'),
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'error regressor: electrode '+channel, ylim = dict(eeg=[-4, 4]),
            axes = ax, truncate_xaxis = False, vlines = [0, 0.5])
    ax.set_title('error - confidence regressor on the feedback ERP at electrode '+channel)
    ax.set_ylabel('beta (AU)')
    for contrast in ['confdiff','confdiff_correct','confdiff_incorrect']:#,'confdiff_incorrvscorr']:
        if contrast == 'confdiff_correct':
            col = '#4daf4a'
            liney = -2.5
        elif contrast == 'confdiff_incorrect':
            col = '#e41a1c'
            liney = -2.7
        elif contrast == 'confdiff_incorrvscorr':
            col = '#525252'
            liney = -2.9
        elif contrast == 'confdiff':
            col = '#1f78b4'
            liney = -3.1
        for imask in range(len(masks_cope[contrast][channel])):
            ax.hlines(y = liney,
                      xmin = np.min( clutimes[masks_cope[contrast][channel][imask][1]] ),
                      xmax = np.max( clutimes[masks_cope[contrast][channel][imask][1]] ),
                      lw = 4, color = col, alpha = .5)
    fig.savefig(fname = op.join(figpath, 'feedbackERP_confdiffRegressor_clustertimesFrom_%s_betas.eps'%(channel)), dpi = 300, format = 'eps')
    fig.savefig(fname = op.join(figpath, 'feedbackERP_confdiffRegressor_clustertimesFrom_%s_betas.pdf'%(channel)), dpi = 300, format = 'pdf')
            
#and now just plot the topographies of any significant clusters
contrast = ['confdiff', 'confdiff_correct']
for contrast in ['confdiff', 'confdiff_correct']:
    for channel in ['FCz']:
        tmins, tmaxs = [], [] #we're going to get the start and end points in time of these clusters
        for mask in masks_cope[contrast][channel]: #loop over clusters
            tmins.append( clutimes[mask[1]].min() )
            tmaxs.append( clutimes[mask[1]].max() )
        
        for imask in range(len(tmins)):
            itmin = tmins[imask] #get start time of the cluster
            itmax = tmaxs[imask] #get   end time of the cluster
            
            tcentre = np.add(itmin, np.divide(np.subtract(itmax,itmin),2)) #get the halfway point
            twidth  = np.subtract(itmax,itmin) #get the width of the cluster (creates timerange for the topomap)
            fig = mne.grand_average(data[contrast]).plot_topomap(times = tcentre,
                                                                   average = twidth,
                                                                   vmin = -.75, vmax = .75, contours = 0,
                                                                   extrapolate = 'head', res = 300, #dpi
                                                                   title = 'error-confidence from %s, %s to %sms\nclu from channel: %s'%(contrast, str(np.round(itmin,2)), str(np.round(itmax,2)), channel))
            fig.set_figwidth(5); fig.set_figheight(5)
            fig.savefig(fname = op.join(figpath, 'confdiff_%s_cluster_%sms_to_%sms_betas_fromChannel_%s.eps'%(contrast, str(np.round(itmin,2)), str(np.round(itmax,2)),channel)), dpi = 300, format = 'eps')
            fig.savefig(fname = op.join(figpath, 'confdiff_%s_cluster_%sms_to_%sms_betas_fromChannel_%s.pdf'%(contrast, str(np.round(itmin,2)), str(np.round(itmax,2)),channel)), dpi = 300, format = 'pdf')
