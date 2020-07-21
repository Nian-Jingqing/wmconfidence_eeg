#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:03:10 2019

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

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri/analysis_scripts')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


#subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)

    fblocked = mne.epochs.read_epochs(fname = param['fblocked'], preload=True) #read raw data
    fblocked.resample(100) #downsample to 100Hz so don't overwork the workstation lols
    fblocked.shift_time(tshift = -0.025, relative = True) #delay recorded by photodiode

    
    bdata = fblocked.metadata
    bdata['nxttrlcwadj'] = bdata.nexttrlcw - bdata.confwidth
    
    for laplacian in [True, False]:
        # set up params for TF decomp
        freqs = np.arange(1, 41, 1)  # frequencies from 2-35Hz
        n_cycles = freqs *.3  # 300ms timewindow for estimation
        if laplacian:
            for stiff in [4]: 
                #get surface laplacian
                #the n_legendre_terms basically makes no difference here really
                slap = mne.preprocessing.compute_current_source_density(fblocked,
                                                                        stiffness = stiff)

                print('\n running TF decomposition')
                tfr = mne.time_frequency.tfr_morlet(slap, freqs=freqs, n_cycles=n_cycles,
                                                    use_fft = True, return_itc = False, average = False)
                tfr.metadata.to_csv(param['fblocked_tfr_meta'], index = False)

                tfr.save(fname = param['fblocked_tfr'].replace('fblocked-tfr', 'fblocked_laplacian_stiffness%d-tfr'%(stiff)), overwrite = True)
                del(slap)
        else:
            print('\nrunning TF decomposition\n')
            # Run TF decomposition overall epochs
            tfr = mne.time_frequency.tfr_morlet(fblocked, freqs=freqs, n_cycles=n_cycles,
                                 use_fft=True, return_itc=False, average=False)
            tfr.metadata.to_csv(param['fblocked_tfr_meta'], index=False)
            tfr.save(fname = param['fblocked_tfr'], overwrite = True)
    
        del(tfr)
    del(fblocked)
