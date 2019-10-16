#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:52:38 2019

@author: sammi
"""
import numpy as np
import scipy as sp
import pandas as pd
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
import pickle

sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence/analysis_scripts')
#sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence_eegfmri/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence

sys.path.insert(0, '/Users/sammi/Desktop/Experiments/BCEyes')
#sys.path.insert(0, '/home/sammirc/Desktop/DPhil/BCEyes')
import BCEyes as bce



wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
os.chdir(wd)


subs = np.array([4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16])
#%%

downsample = True
derivative = True
resample_freq = 200
tmin, tmax = -0.2, 1.5
timerange = np.arange(tmin, tmax,.001)

numsamps = timerange.size
resamp_numsamps = int(numsamps/(1000/resample_freq)) #this is for if gaze position is downsampled

allsubs_cuelocked = np.empty(shape = (subs.size, 3, 3, timerange.size)) #subject, eyetrace, condition (cued left, cued right, neutral), time
if downsample:
    _, trange_downsamp = sp.signal.resample(x = timerange, t = timerange, num = resamp_numsamps) #downsampled for gaze position
    allsubs_cuelocked = np.empty(shape = (subs.size, 3, 3, trange_downsamp.size))

if derivative:
    timerange = np.arange(tmin, tmax, .001)
    timerange = timerange[1:]
    allsubs_cuelocked = np.empty(shape = (subs.size, 3, 3, timerange.size))

if derivative and downsample:
    timerange = np.arange(tmin, tmax, .001)[1:]
    _, trange_deriv_downsamp = sp.signal.resample(x = timerange, t = timerange, num = resamp_numsamps-1)
    allsubs_cuelocked = np.empty(shape = (subs.size, 3, 3, trange_deriv_downsamp.size))

count = 0
for i in subs:
    sub = dict(loc = 'laptop', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    #read from pickle here
    print('-- reading in subject %d --\n'%(i))
    with open(op.join(param['path'], 'eyes', param['subid'], 'wmc_'+param['subid']+'_preprocessed_combined.pickle'), 'rb') as handle:
        data = pickle.load(handle)

    alltrigs = sorted(np.unique(data[0]['Msg'][:,2]))
    
    cue_trigs      = dict(neutral_left = 'trig11',            neutral_right = 'trig12',
                          cued_left    = 'trig13',            cued_right    = 'trig14',
                          neutral      = ['trig11','trig12'], cued          = ['trig13','trig14'])
    
    data = bce.average_eyes_blocked(data, traces = ['x'])
    
        
    #snip the ends off the data file here (1.5s after the last trigger (which should be feedback))
    data = bce.snip_end_of_blocks(data = data, traces_to_snip = ['ave_x'])
    blocks_with_nans = [x for x in range(len(data)) if np.sum(np.isnan(data[x]['ave_x'])) > 0]
    if len(blocks_with_nans) > 0:
        print(blocks_with_nans)            
    
    #we expect something here for some people where data was missing in entire blocks, e.g.:
    #subject 11 session 1 - block 8 (7 w/ indexing) 
    #subject 11 session 2 - block 3 (2 w/ indexing)
    #subject 12 session 1 - block 7 (6 w/ indexing)
    #subject 12 session 2 - blocks 6,7 (5, 6 w/ indexing)
                
    print([x for x in range(len(data)) if np.sum(np.isnan(data[x]['ave_x']))>0]) #check if any block has nans in the gaze data at all


    cuelocked = bce.epoch(data = data,
                          trigger_values = cue_trigs['cued'] + cue_trigs['neutral'],
                          traces = ['lx', 'rx'],
                          twin = [tmin, tmax],
                          srate = 1000)
    cuedleft = np.where(cuelocked['info']['trigger'] == cue_trigs['cued_left'])
    cuedright = np.where(cuelocked['info']['trigger'] == cue_trigs['cued_right'])
    neutral = np.where(np.logical_or(
            cuelocked['info']['trigger'] == cue_trigs['neutral_left'],
            cuelocked['info']['trigger'] == cue_trigs['neutral_right']
            ))
    

    cuelocked = bce.average_eyes(cuelocked, traces = ['x'])
    cuelocked = bce.apply_baseline(epoched_data  = cuelocked,
                                   traces = ['ave_x'],
                                   baseline_window = [-0.1,0],
                                   mode = 'median', baseline_shift_gaze = [960, 540])
    lpfilt_epochs = True
    if lpfilt_epochs:
        cuelocked['ave_x'] = bce.lpfilter_epochs(cuelocked, trace = 'ave_x', lp_cutoff = 60, srate = 1000, order = 2)
    #tmp = bce.lpfilter_epochs(cuelocked, trace = 'ave_x', lp_cutoff = 60, srate = 1000, order = 2)

    if downsample:
        print('downsampling traces\n')
        for trace in ['lx', 'rx', 'ave_x']:
            cuelocked[trace], trange = sp.signal.resample(x = cuelocked[trace], num = int(cuelocked[trace].shape[1]/(1000/resample_freq)), t = timerange, axis = 1)
            
    if derivative:
        for trace in ['lx', 'rx', 'ave_x']:
            cuelocked[trace] = np.diff(cuelocked[trace], axis = 1)
    
    allsubs_cuelocked[count, 0, 0, :] = np.nanmean(cuelocked['lx'][cuedleft] , 0)
    allsubs_cuelocked[count, 0, 1, :] = np.nanmean(cuelocked['lx'][cuedright], 0)
    allsubs_cuelocked[count, 0, 2, :] = np.nanmean(cuelocked['lx'][neutral]  , 0)
    
    allsubs_cuelocked[count, 1, 0, :] = np.nanmean(cuelocked['rx'][cuedleft] , 0)
    allsubs_cuelocked[count, 1, 1, :] = np.nanmean(cuelocked['rx'][cuedright], 0)
    allsubs_cuelocked[count, 1, 2, :] = np.nanmean(cuelocked['rx'][neutral]  , 0)
    
    allsubs_cuelocked[count, 2, 0, :] = np.nanmean(cuelocked['ave_x'][cuedleft] , 0)
    allsubs_cuelocked[count, 2, 1, :] = np.nanmean(cuelocked['ave_x'][cuedright], 0)
    allsubs_cuelocked[count, 2, 2, :] = np.nanmean(cuelocked['ave_x'][neutral]  , 0)
    
    
    
    count +=1
    
    
#    fig = plt.figure()
#    fig.suptitle('subject %d'%i)
#    ax = fig.add_subplot(111)
#    ax.plot(np.squeeze(np.nanmean(cuelocked['lx'][cuedleft] , 0)), np.arange(-.5,1.5,.001), color = '#e41a1c', lw = .5, label = 'cued left')
#    ax.plot(np.squeeze(np.nanmean(cuelocked['lx'][cuedright], 0)), np.arange(-.5,1.5,.001), color = '#4daf4a', lw = .5, label = 'cued right')
#    ax.plot(np.squeeze(np.nanmean(cuelocked['lx'][neutral], 0))  , np.arange(-.5,1.5,.001), color = '#377eb8', lw = .5, label = 'neutral')
#    ax.set_xlim(xmin = 910, xmax = 1010)
#    ax.vlines(x = 960, ymin = -.5, ymax = 1.5, ls = 'dashed', color = '#f0f0f0')
#    fig.legend(loc = 'upper right')


gave_cuedleft_lx = np.nanmean(np.squeeze(allsubs_cuelocked[:,0,0,:]), axis = 0)
gave_cuedleft_rx = np.nanmean(np.squeeze(allsubs_cuelocked[:,1,0,:]), axis = 0)
gave_cuedleft_x  = np.nanmean(np.squeeze(allsubs_cuelocked[:,2,0,:]), axis = 0)

gave_cuedright_lx = np.nanmean(np.squeeze(allsubs_cuelocked[:,0,1,:]), axis = 0)
gave_cuedright_rx = np.nanmean(np.squeeze(allsubs_cuelocked[:,1,1,:]), axis = 0)
gave_cuedright_x  = np.nanmean(np.squeeze(allsubs_cuelocked[:,2,1,:]), axis = 0)

gave_neutral_lx = np.nanmean(np.squeeze(allsubs_cuelocked[:,0,2,:]), axis = 0)
gave_neutral_rx = np.nanmean(np.squeeze(allsubs_cuelocked[:,1,2,:]), axis = 0)
gave_neutral_x  = np.nanmean(np.squeeze(allsubs_cuelocked[:,2,2,:]), axis = 0)

fig = plt.figure()
if derivative and not downsample:
    fig.suptitle('all subs, average gaze displacement')
    timewin = timerange
elif derivative and downsample:
    fig.suptitle('all subs, gaze displacement, downsampled')
    timewin = trange_deriv_downsamp
elif not derivative and downsample:
    fig.suptitle('all subs, gaze position, downsampled')
    timewin = trange_downsamp
elif not derivative and not downsample:
    fig.suptitle('all subjects, gaze position')
    timewin = timerange
    
ax = fig.add_subplot(111)
ax.plot(gave_cuedleft_x ,  timewin, lw = 1, label = 'gave cued left ave x')
ax.plot(gave_cuedright_x,  timewin, lw = 1, label = 'gave cued right ave x')
ax.plot(gave_neutral_x  , timewin, lw = 1, label = 'gave neutral ave x')

#ax.vlines(x = 960, ymin = -.499, ymax = 1.5, ls = 'dashed', color = '#bdbdbd')
if derivative:
    ax.hlines(y = [0, 1.5], xmin = -.2, xmax = .2, ls = 'dashed', color = '#bdbdbd')
    ax.set_xlim([-2, 2])
else:
    ax.hlines(y = [0, 1.5], xmin = 940, xmax = 980, ls = 'dashed', color = '#bdbdbd') 
    ax.vlines(x = 960, ymin = tmin, ymax = tmax, ls = 'dashed', color = '#bdbdbd')               

if not derivative:
    ax.set_xlabel('gaze coordinate x (pixels)')
elif derivative:
    ax.set_xlabel('gaze displacement (pixels)')
ax.set_ylabel('Time rel. to cue onset (s)')
fig.legend(loc = 'upper right')
