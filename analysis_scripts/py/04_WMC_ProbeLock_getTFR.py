#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  13 14:00:10 2019

@author: sammirc
"""


import numpy as np
import pandas as pd
import mne
import os
import os.path as op
import sys

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri/analysis_scripts')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)

    probelocked = mne.epochs.read_epochs(fname = param['probelocked'], preload=True) #read raw data
    probelocked.resample(100) #downsample to 250Hz so don't overwork the workstation (100hz should be fine for frequencies up to 50Hz in brain)


    #will do an automated process of looking for trials with heightened variance (noise) and output which trials to keep
    _, keeps = plot_AR(probelocked, method = 'gesd', zthreshold = 1.5, p_out=.1, alpha = .05, outlier_side = 1)
    keeps = keeps.flatten()

    discards = np.ones(len(probelocked), dtype = 'bool')
    discards[keeps] = False
    probelocked = probelocked.drop(discards) #first we'll drop trials with excessive noise in the EEG

    #now we'll drop trials with behaviour problems (reaction time +/- 2.5 SDs of mean, didn't click to report orientation)
    probelocked = probelocked['DTcheck == 0 and clickresp == 1 and arraycueblink == 0'] #also exclude trials where blinks happened in the array or cue period
    probelocked.set_eeg_reference(ref_channels=['RM']) #re-reference to average of the two mastoids

    # set up params for TF decomp
    freqs = np.arange(1, 40, 1)  # frequencies from 2-35Hz
    n_cycles = freqs *.3  # 300ms timewindow for estimation

    print('\nrunning TF decomposition\n')
    # Run TF decomposition overall epochs
    tfr = mne.time_frequency.tfr_morlet(probelocked, freqs=freqs, n_cycles=n_cycles,
                         use_fft=True, return_itc=False, average=False)
    tfr.metadata.to_csv(param['probelocked_tfr_meta'], index=False)
    tfr.save(fname = param['probelocked_tfr'], overwrite = True)

    del(probelocked)
    del(tfr)
