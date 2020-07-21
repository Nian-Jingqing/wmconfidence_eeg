#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:44:48 2019

@author: sammirc
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 22:34:27 2019

@author: sammirc
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:07:38 2019

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
from wmConfidence_funcs import gesd, plot_AR, toverparam

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

subs = np.array([        4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 24, 25])

contrasts = [#'grandmean',
             'pleft_cued', 'pleft_neutral', 'pright_cued', 'pright_neutral', 'neutral', 'cued', 'crvsn', 'clvsn','clvsr', 'cuedvsneut',
             'err_pleft_neutral','err_pleft_cued','err_pright_neutral','err_pright_cued','err_clvsn','err_crvsn','err_cued','err_neutral','err_clvsr','err_cuedvsneut']

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
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    for name in contrasts:
        data[name].append(mne.time_frequency.read_tfrs(            fname = op.join(param['path'], 'glms', 'cue','tfrglm5_nogmean', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_betas-tfr.h5'))[0])
        data_t[name].append(mne.time_frequency.read_tfrs(          fname = op.join(param['path'], 'glms', 'cue','tfrglm5_nogmean', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_tstats-tfr.h5'))[0])
        data_baselined[name].append(mne.time_frequency.read_tfrs(  fname = op.join(param['path'], 'glms', 'cue','tfrglm5_nogmean', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_betas_baselined-tfr.h5'))[0])
        data_baselined_t[name].append(mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'cue','tfrglm5_nogmean', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_tstats_baselined-tfr.h5'))[0])
#%%
timefreqs = {(.4, 10):(.4, 4),
             (.6, 10):(.4, 4),
             (.8, 10):(.4, 4),
             (.4, 22):(.4, 16),
             (.6, 22):(.4, 16),
             (.8, 22):(.4, 16)}

timefreqs_alpha ={(.4, 10):(.4, 4),
                  (.6, 10):(.4, 4),
                  (.8, 10):(.4, 4),
                  (1., 10):(.4, 4),
                  (1.2, 10):(.4, 4)}

timefreqs_cue = {(-1.2, 10):(.4, 4),
                 (-1.0, 10):(.4, 4),
                 (-0.8, 10):(.4, 4),
                 (-0.6, 10):(.4, 4)}

visleftchans  = ['PO3', 'PO7', 'O1']

visrightchans = ['PO4','PO8','O2']

motrightchans = ['C2', 'C4']  #channels contra to the left hand (space bar)
motleftchans = ['C1', 'C3']   #channels contra to the right hand (mouse)

topoargs = dict(outlines= 'head', contours = 0)
topoargs_t = dict(outlines = 'head', contours = 0, vmin=-2, vmax = 2)
#%%
gave_gmean = mne.grand_average(data['clvsr']); #gave_gmean.data = toverparam(data['grandmean']); #gave_gmean.drop_channels(['RM'])
#gave_gmean.plot_joint(title = 'grandmean, t over tstats', timefreqs = timefreqs_cue,
#                      topomap_args = dict(outlines = 'head', contours = 0, vmin = -5, vmax = 5))

times = gave_gmean.times
#timesrel2cue = np.add(times, 1.5) #this sets zero to be the cue onset time
allfreqs = gave_gmean.freqs
del(gave_gmean)
#%%
#for i in range(subs.size):
#    for contrast in contrasts:
#        data[contrast][i].drop_channels(['RM'])
#        data_t[contrast][i].drop_channels(['RM'])
#        data_baselined[contrast][i].drop_channels(['RM'])
#        data_baselined_t[contrast][i].drop_channels(['RM'])

#%%
plt.close('all')
plot_sub = True
for isub in range(subs.size):
    if plot_sub:
        #fig = data_t['clvsr'][isub].plot_joint(topomap_args = dict(outlines='skirt', contours = 0, vmin=-2,vmax=2),timefreqs = timefreqs_alpha)
#        data_t['clvsr'][isub].plot_joint(topomap_args = dict(outlines='skirt', contours = 0, vmin=-2,vmax=2),timefreqs = timefreqs_alpha, title = 'sub %02d'%(subs[isub]))
        data['clvsr'][isub].plot_joint(topomap_args = dict(outlines='skirt', contours = 0,vmin=-5e-11,vmax=5e-11),
            timefreqs = timefreqs_alpha, title = 'sub %02d'%(subs[isub]), vmin = -1e-11, vmax = 1e-11)
#%%

topovmin = dict()
topovmin['beta'] = -5e-11
topovmin['tstat'] = -2.4

contrast = 'clvsr'
stat = 'beta'
if 'vs' in contrast:
    if stat == 'beta':
        dat2use = deepcopy(data)
    else:
        dat2use = deepcopy(data_t)
else:
    if stat == 'beta':
        dat2use = deepcopy(data_baselined)
    else:
        dat2use = deepcopy(data_baselined_t)


gave = mne.grand_average(dat2use[contrast])

plot_t=False
if plot_t:
    stat = 'tstat'
    gave.data = toverparam(dat2use[contrast])

gave.plot_joint(timefreqs = timefreqs_alpha, vmin=topovmin[stat], vmax = topovmin[stat]*-1,
                topomap_args = dict(contours = 0,
                                    vmin = topovmin[stat],
                                    vmax = topovmin[stat]*-1))
#%%

#contrast = 'err_pright_neutral'
stat = 'tstat'
if 'vs' in contrast:
    if stat == 'beta':
        dat2use = deepcopy(data)
    else:
        dat2use = deepcopy(data_t)
else:
    if stat == 'beta':
        dat2use = deepcopy(data_baselined)
    else:
        dat2use = deepcopy(data_baselined_t)

for contrast in ['err_neutral']:
    gave = mne.grand_average(dat2use[contrast]); gave.data = toverparam(dat2use[contrast])
    gave.plot_joint(topomap_args = dict(outlines='skirt', contours=0,vmin=-3,vmax=3), #baseline = (-1.75,-1.25), title ='baselined',#vmin=-4,vmax=4),
                    timefreqs = timefreqs_alpha, vmin=-3, vmax=3)

#looks like there might be a relationship here between motor activity in cue period and error (lower alpha ~ lower error = +ve beta)


    motlats = np.empty(shape = (subs.size, allfreqs.size, times.size))
    c3s     = np.empty(shape = (subs.size, allfreqs.size, times.size))
    c4s     = np.empty(shape = (subs.size, allfreqs.size, times.size))
    for i in range(subs.size):
        #for each subject get c3/c4 and calculate lateralisation for motor electrodes
        tmp_c3 = deepcopy(dat2use[contrast][i]).pick_channels(['C3']).data; c3s[i] = tmp_c3
        tmp_c4 = deepcopy(dat2use[contrast][i]).pick_channels(['C4']).data; c4s[i] = tmp_c4
        
        motlats[i] = tmp_c4 - tmp_c3 #right - left electrodes, so relative to the hand on space bar

    #plot average relationship between C3 time frequency and error
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(sp.stats.ttest_1samp(c3s,popmean=0,axis=0)[0], cmap = 'RdBu_r', aspect = 'auto', interpolation = 'gaussian', origin = 'lower',
               extent = (times.min(), times.max(), allfreqs.min(), allfreqs.max()), vmin = -1.5, vmax = 1.5)
    ax1.set_xlabel('time rel. 2 cue onset (s)')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('Error regressor at C3')    
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(sp.stats.ttest_1samp(c4s,popmean=0,axis=0)[0], cmap = 'RdBu_r', aspect = 'auto', interpolation = 'gaussian', origin = 'lower',
               extent = (times.min(), times.max(), allfreqs.min(), allfreqs.max()), vmin = -1.5, vmax = 1.5)
    ax2.set_xlabel('time rel. 2 cue onset (s)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Error regressor at C4')

#%%
contrast = 'err_neutral'
gave = mne.grand_average(data_baselined_t[contrast]); gave.data = toverparam(data_baselined_t[contrast])
gave.plot_joint(topomap_args = dict(outlines='skirt', contours=0,vmin=-3,vmax=3), #baseline = (-1.75,-1.25), title ='baselined',#vmin=-4,vmax=4),
                timefreqs = timefreqs_alpha, vmin=-3, vmax=3)

plot_c3 = True
if plot_c3:
    c3 = np.squeeze(deepcopy(gave).pick_channels(['C3']).data)
    fig,ax = plt.subplots()
    ax.imshow(c3,
        cmap = 'RdBu_r', aspect = 'auto', interpolation = 'gaussian', origin = 'lower', vmin=-3, vmax = 3,
        extent = ( np.min(times), np.max(times), np.min(allfreqs), np.max(allfreqs)))
    ax.set_xlabel('time rel. 2 cue onset')
    ax.set_ylabel('Frequency (Hz)')
    ax.vlines([-1.25, -1, 0, 0.25, 1.75], lw = 1, linestyles='dashed', color = '#000000', ymin=1, ymax=40)


flip_plot = False
if flip_plot:
    chnames =         np.array([       'FP1', 'FPZ', 'FP2', 
                                'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 
                  'F7',  'F5',   'F3',  'F1',  'FZ',  'F2',  'F4',  'F6',  'F8',
                 'FT7', 'FC5',  'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
                  'T7',  'C5',   'C3',  'C1',  'CZ',  'C2',  'C4',  'C6',  'T8',
                 'TP7', 'CP5',  'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                  'P7',  'P5',   'P3',  'P1',  'PZ',  'P2',  'P4',  'P6',  'P8',
                 'PO7',         'PO3',        'POZ',        'PO4', 'PO8',
                                        'O1',  'OZ',  'O2'])
    chids =         np.array([             1,  2,  3,
                                       4,  5,  6,  7,  8,
                               9, 10, 11, 12, 13, 14, 15, 16, 17,
                              18, 19, 20, 21, 22, 23, 24, 25, 26,
                              27, 28, 29, 30, 31, 32, 33, 34, 35,
                              36, 37, 38, 39, 40, 41, 42, 43, 44,
                              45, 46, 47, 48, 49, 50, 51, 52, 53,
                                      54, 55, 56, 57, 58,
                                          59, 60, 61
                                              ])
    chids = np.subtract(chids,1)
    flipids =       np.array([             3,  2,  1,
                                       8,  7,  6,  5,  4,
                              17, 16, 15, 14, 13, 12, 11, 10,  9,
                              26, 25, 24, 23, 22, 21, 20, 19, 18,
                              35, 34, 33, 32, 31, 30, 29, 28, 27,
                              44, 43, 42, 41, 40, 39, 38, 37, 36,
                              53, 52, 51, 50, 49, 48, 47, 46, 45,
                                      58, 57, 56, 55, 54,
                                          61, 60, 59
                                              ])
    lhs_chanids  = np.array([1,4,5,9,10,11,12,18,19,20,21,27,28,29,30,36,37,38,39,45,46,47,48,54,55,59])
#    rhs_chaninds = 
    midline_chaninds = np.array([2,6,13,22,31,40,49,56,60])
    flipids = np.subtract(flipids,1)
    flippednames = chnames[flipids]
    
    renaming_mapping = dict()
    for i in range(len(chnames)):
        renaming_mapping[chnames[i]] = flippednames[i]
    
    flipids = flipids
    flipped_gave = deepcopy(gave)
    flipped_gave.data = flipped_gave.data[flipids,:,:] #flip the data leave the channels where they are
    
    flipped_gave.plot_joint(topomap_args = dict(outlines='skirt', contours=0), #baseline = (-1.75,-1.25), title ='baselined',#vmin=-4,vmax=4),
                    timefreqs = timefreqs_alpha, title='flipped topo', vmin=-3, vmax=3)
    
    if contrast in ['clvsr', 'err_clvsr']:
        vmin, vmax = -2e-10, 2e-10
        vmin, vmax = -3,3
    else:
        vmin, vmax = -1e-10, 1e-10
    
    tmp = deepcopy(gave)
    
    
    tmp.data = np.subtract(tmp.data, flipped_gave.data)
    
#    if contrast in ['cued']:
#        tmp.data = np.multiply(tmp.data, -1)
    
    
#    tmp.data[lhs_chanids,:,:] = 0
#    tmp.data[midline_chaninds,:,:] = 0
    fig = tmp.plot_joint(topomap_args = dict(outlines='head', contours=0, vmin = vmin, vmax = vmax), #baseline = (-1.75,-1.25), title ='baselined',#vmin=-4,vmax=4),
                         timefreqs = timefreqs_alpha, title='rhs - lhs', vmin = vmin, vmax = vmax)
    axes = fig.axes
    axes[0].clear()
    if contrast in ['clvsr','pleft_cued','clvsn']:
        chans = visrightchans
    elif contrast in ['pright_cued', 'crvsn']:
        chans = visleftchans
    tfrplot = axes[0].imshow(np.squeeze(np.nanmean(deepcopy(tmp).pick_channels(chans).data,0)),
                        cmap = 'RdBu_r', aspect = 'auto', interpolation = 'gaussian', origin = 'lower', vmin=vmin, vmax = vmax,
                        extent = ( np.min(times), np.max(times), np.min(allfreqs), np.max(allfreqs)))
    axes[0].set_xlabel('time rel. 2 cue onset')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].vlines([-1.25, -1, 0, 0.25, 1.75], lw = 1, linestyles='dashed', color = '#000000', ymin=1, ymax=40)

plot_motor = False
if plot_motor and contrast == 'cuedvsneut':
    vminmot, vmaxmot = -2, 2
    cvsimot = np.empty((subs.size, allfreqs.size, times.size))
    
    for i in range(subs.size):
            
        dat = deepcopy(data[contrast])[i]#get grand average beta
        #cvsi to mouse hand is left channel - right channel == C3 - C4
        cvsimotor = np.subtract(np.squeeze(np.nanmean(deepcopy(dat).pick_channels(['C3','CP3']).data,0)),
                                np.squeeze(np.nanmean(deepcopy(dat).pick_channels(['C4', 'CP4']).data,0)))
        cvsimot[i,:,:] = cvsimotor
    
    cvsimot = sp.stats.ttest_1samp(cvsimot, axis=0, popmean=0)[0]
    
    fig,ax = plt.subplots()
    ax.imshow(cvsimot, cmap = 'RdBu_r', aspect = 'auto', interpolation = 'gaussian', origin = 'lower', vmin = vminmot, vmax = vmaxmot,
              extent = ( np.min(times), np.max(times), np.min(allfreqs), np.max(allfreqs)))
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time rel. 2 cue onset (s)')
    ax.vlines([-1.25, -1, 0, 0.25, 1.75], lw = 1, linestyles='dashed', color = '#000000', ymin=1, ymax=40)
#%%
cvsi = np.empty((subs.size, allfreqs.size, times.size))
for i in range(subs.size):
    
    cvsi[i,:,:] = np.subtract(
        np.nanmean(deepcopy(data['clvsr'])[i].pick_channels(visrightchans).data,0),
        np.nanmean(deepcopy(data['clvsr'])[i].pick_channels(visleftchans).data,0)
        )

#cvsi = np.subtract(
#        np.nanmean(deepcopy(gave).pick_channels(visrightchans).data,0),
#        np.nanmean(deepcopy(gave).pick_channels(visleftchans).data,0)
#        )

fig=plt.figure()
ax=fig.add_subplot(111)
tf = ax.imshow(np.squeeze(np.nanmean(cvsi,0)), cmap='RdBu_r',aspect='auto', interpolation = 'None', origin = 'lower',#vmin=-3,vmax=3,
          extent = (np.min(times), np.max(times), np.min(allfreqs), np.max(allfreqs)   ))
ax.vlines([-1.25, 0, 1.75], linestyles='dashed', color ='#000000', lw=1.5, ymin=1,ymax=40)
fig.colorbar(tf)


