#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:37:53 2019

@author: sammirc
"""

import numpy as np
import scipy as sp
import pandas as pd
import mne
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
plt.ion()

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri/analysis_scripts')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1,2, 3, 4, 5, 6, 7])

isub = 1
i = subs[isub-1]
print('\n\nworking on subject ' + str(isub) +'\n\n')
sub = dict(loc = 'workstation', id = i)
param = get_subject_info_wmConfidence(sub)

epoch = mne.read_epochs(fname = param['cuelocked'],preload = True)
epoch.set_eeg_reference(ref_channels = ['RM'])
epoch.metadata['trialnumber'] = np.arange(1, len(epoch)+1,1)

#remove some trials based on behavioural data

#this removes trials where RT outside 2.5 SDs of condition mean, didn't click to respond, or blinked at array/cue presentation
epoch = epoch['DTcheck == 0 and clickresp == 1 and arraycueblink==0']

#will do an automated process of looking for trials with heightened variance (noise) and output which trials to keep
_, keeps = plot_AR(epoch, method = 'gesd', zthreshold = 1.5, p_out=.1, alpha = .05, outlier_side = 1)
keeps = keeps.flatten()

#get trials to discard based on the gesd approach
discards = np.ones(len(epoch), dtype = 'bool')
discards[keeps] = False
epoch = epoch.drop(discards)


epoch_cuedleft  = epoch['cued/probeleft']
epoch_cuedright = epoch['cued/proberight']
epoch_neutral   = epoch['neutral'] 


tl_cuedleft  = epoch_cuedleft.average() ; tl_cuedleft.apply_baseline((-0.5, 0))
tl_cuedright = epoch_cuedright.average(); tl_cuedright.apply_baseline((-0.5,0)) 

visright_picks = mne.pick_channels(tl_cuedleft.ch_names, ['PO8', 'O2', 'PO4'])
visleft_picks  = mne.pick_channels(tl_cuedleft.ch_names, ['PO7', 'O1', 'PO3'])

#plot diff between cued left and right trials
tl_lvsr = mne.combine_evoked([tl_cuedleft, -tl_cuedright], weights = 'equal')
tl_lvsr.set_eeg_reference(ref_channels = 'average')
tl_lvsr.plot_topomap(times = .6, average = .4, vmin = -1, vmax = 1)
tl_lvsr.plot_joint()

