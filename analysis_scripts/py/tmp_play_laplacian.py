#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:20:28 2020

@author: sammirc
"""

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


subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
epoch_to_create = 'cuelocked'
#%%
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)

    
    if not op.exists(param[epoch_to_create].replace(epoch_to_create, epoch_to_create+'_cleaned')):
        epochs = mne.epochs.read_epochs(fname = param['cuelocked'], preload=True) #read raw data
        epochs.shift_time(-0.025, relative =True) #shift times based on photodiode testing
        
        epochs.resample(100) #downsample to 250hz so don't overwork the workstation lols
        epochs = epochs.pick_types(eeg=True, misc = True) #if you don't do this, and run the gesd, then it can fail if the EOG variance makes trialwise variance estimation terrible
        #we don't need the eogs at this point anyways because we ica'd out blinks etc, so its just extra data to take up space
        
        bdata = epochs.metadata
        prevtrlerr = bdata.shift(1).absrdif.to_numpy()
        bdata['prevtrlerr'] = bdata.shift(1).absrdif.to_numpy()
        bdata['prevtrlconf'] = bdata.shift(1).confwidth.to_numpy()
        
        epochs.metadata = bdata
    
        #will do an automated process of looking for trials with heightened variance (noise) and output which trials to keep
        _, keeps = plot_AR(epochs, method = 'gesd', zthreshold = 1.5, p_out=.1, alpha = .05, outlier_side = 1)
        keeps = keeps.flatten()
    
        discards = np.ones(len(epochs), dtype = 'bool')
        discards[keeps] = False
        epochs = epochs.drop(discards) #first we'll drop trials with excessive noise in the EEG
    
        #now we'll drop trials with behaviour problems (reaction time +/- 2.5 SDs of mean, didn't click to report orientation)
        epochs = epochs['DTcheck == 0 and clickresp == 1']
        epochs.set_eeg_reference(ref_channels=['RM']) #re-reference to average of the two mastoids
        
        epochs.plot(n_epochs=3, n_channels=61, scalings = dict(eeg=100e-6)) #scan to see if any really noisy trials were missed, and drop these if they were
        #if there is excessive noise in the baseline period this is particularly bad
        
        outname = param[epoch_to_create].replace(epoch_to_create, epoch_to_create+'_cleaned')
        
        #this should really get saved out:
        epochs.save(fname = outname, overwrite=True)
        epochs.resample(100) #resample to 100 hz
    else:
        print('cleaned data already exists for subject %s'%str(i))
        epochs = mne.epochs.read_epochs(fname = param[epoch_to_create].replace(epoch_to_create, epoch_to_create+'_cleaned'), preload = True)
        epochs.resample(100) #resample to 100Hz
    
    epochs.drop_channels(['RM'])
    chnames = np.asarray(epochs.ch_names)
    chnamemapping = {}
    for x in range(len(chnames)):
        chnamemapping[chnames[x]] = chnames[x].replace('Z', 'z').replace('FP', 'Fp')
    mne.rename_channels(epochs.info, chnamemapping)
    epochs.set_montage('easycap-M1')
    
    laplacian = True
    if laplacian:
#        epochs.drop_channels(['RM', 'LM'])
        epochs = mne.preprocessing.compute_current_source_density(epochs)
     # set up params for TF decomp
    freqs = np.arange(1, 41, 1)  # frequencies from 1-40Hz
    n_cycles = freqs *.3  # 300ms timewindow for estimation

    print('\nrunning TF decomposition\n')
    # Run TF decomposition overall epochs
    tfr = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles,
                         use_fft=True, return_itc=False, average=False)
#    tfr.metadata.to_csv(param['cuelocked_tfr_meta'], index=False)
    
#    if laplacian:
#        tfr.save(fname = param['cuelocked_tfr'].replace('cuelocked-tfr', 'cuelocked_laplacian-tfr'))
#    tfr.save(fname = param['cuelocked_tfr'], overwrite = True)
#
#    del(cuelocked)
#    del(tfr)

    
    
    