#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:03:07 2019

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

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1,2])


subind = 2 #get first subject

sub = dict(loc='workstation', id=subs[subind-1]) #get subject name for param extraction
param = get_subject_info_wmConfidence(sub)

#data for this expt collected using normal EEG caps, so just easycap-M1 layout used
raw = mne.io.read_raw_eeglab(
        input_fname = param['rawset'],
        montage     = 'easycap-M1',
        eog = ['VEOG', 'HEOG'],
        preload = True
        )
raw.rename_channels({'PO6':'PO8'}) #error in the template set up file i made for mark, calls PO8 PO6 instead, so gest no location. correct this
raw.set_montage('easycap-M1')

raw.plot(duration = 4,
         n_channels = len(raw.info['chs']),
         color = dict(eog='blue', eeg = 'black'),
         highpass = 1, lowpass = 40
         )
#clean raw data first
raw.filter(1,40)

ica = mne.preprocessing.ICA(n_components = .99, method = 'infomax').fit(raw)

eog_epochs = mne.preprocessing.create_eog_epochs(raw)
eog_inds, eog_scores = ica.find_bads_eog(eog_epochs, threshold=2)
ica.plot_scores(eog_scores)

ica.plot_components(inst=raw)
ica.exclude.extend(eog_inds)
ica.apply(inst=raw)

raw.save(fname = param['rawcleaned'], fmt='double') #save ica cleaned data so can read this in in future

