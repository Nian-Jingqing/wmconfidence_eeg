#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 10:28:38 2019

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
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15])

alldata_left  = []
alldata_right = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    tfr = mne.time_frequency.read_tfrs(fname = op.join(param['path'], param['cuelocked_tfr']))[0]
    alldata_left.append(tfr['cuetrig == 13'])
    alldata_right.append(tfr['cuetrig == 14'])
    del(tfr)
    
timefreqs_alpha ={(.4, 10):(.4, 4),
                  (.6, 10):(.4, 4),
                  (.8, 10):(.4, 4),
                  (1., 10):(.4, 4)}    

#%%

for i in range(len(alldata_left)):
    alldata_left[i].drop_channels(['RM'])
    alldata_right[i].drop_channels(['RM'])
    alldata_left[i]  = alldata_left[i].average()
    alldata_right[i] = alldata_right[i].average()

alldata_lvsr = []
alldata_lvsr_norm = []

for i in range(len(subs)):
    tmp = deepcopy(alldata_left[i].average())
    tmp.data = np.subtract(tmp.data, alldata_right[i].average().data)
    alldata_lvsr.append(tmp)


gave_lvsr = mne.grand_average(alldata_lvsr)
gave_lvsr.plot_joint(timefreqs  = timefreqs_alpha,
                     topomap_args = dict(outlines = 'head', contours = 0))


for i in range(len(subs)):
    tmpleft = deepcopy(alldata_left[i].average())
    tmpright = deepcopy(alldata_right[i].average())
    
    newdat = np.multiply(np.divide(np.subtract(tmpleft.data, tmpright.data), np.add(tmpleft.data, tmpright.data)), 100)
    
    tmpleft.data = newdat
    alldata_lvsr_norm.append(tmpleft)

gave_lvsr_norm = mne.grand_average(alldata_lvsr_norm)
gave_lvsr_norm.plot_joint(timefreqs = timefreqs_alpha, topomap_args = dict(outlines = 'head', contours = 0))


#%%
chnames =         np.array([       'FP1', 'FPZ', 'FP2', 
                            'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 
              'F7',  'F5',   'F3',  'F1',  'FZ',  'F2',  'F4',  'F6',  'F8',
             'FT7', 'FC5',  'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
              'T7',  'C5',   'C3',  'C1',  'CZ',  'C2',  'C4',  'C6',  'T8',
             'TP7', 'CP5',  'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
              'P7',  'P5',   'P3',  'P1',  'PZ',  'P2',  'P4',  'P6',  'P8',
             'PO7',         'PO3',        'POZ',        'PO4', 'PO8',
                                    'O1',  'OZ',  'O2',
                                           'RM'])
chids =         np.array([             1,  2,  3,
                                   4,  5,  6,  7,  8,
                           9, 10, 11, 12, 13, 14, 15, 16, 17,
                          18, 19, 20, 21, 22, 23, 24, 25, 26,
                          27, 28, 29, 30, 31, 32, 33, 34, 35,
                          36, 37, 38, 39, 40, 41, 42, 43, 44,
                          45, 46, 47, 48, 49, 50, 51, 52, 53,
                                  54, 55, 56, 57, 58,
                                      59, 60, 61,
                                          62
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
                                      61, 60, 59,
                                          62
                                          ])
flipids = np.subtract(flipids,1)
flippednames = chnames[flipids]

renaming_mapping = dict()
for i in range(len(chnames[:61])):
    renaming_mapping[chnames[i]] = flippednames[i]

flipids = flipids[:61]
flipped_gave_lvsr_norm = deepcopy(gave_lvsr_norm)
flipped_gave_lvsr_norm.data = flipped_gave_lvsr_norm.data[flipids,:,:] #flip the data leave the channels where they are

gave_lvsr_norm.plot_joint(timefreqs=timefreqs_alpha, topomap_args = dict(outlines='head', contours=0), title = 'normal')
flipped_gave_lvsr_norm.plot_joint(timefreqs=timefreqs_alpha, topomap_args = dict(outlines='head', contours=0), title ='flipped')


norm_lat = deepcopy(gave_lvsr)
norm_lat.data = np.subtract(norm_lat.data, flipped_gave_lvsr_norm.data)
norm_lat.plot_joint(title = 'normalised contra-ipsi for cued left vs right',
                    timefreqs = timefreqs_alpha,
                    topomap_args = dict(outlines='head', contours=0))

norm_lat.plot(picks = ['PO8', 'O2', 'PO4'])






flipped_gave_lvsr_norm.reorder_channels(flippednames)
flipped_gave_lvsr_norm.plot_joint(timefreqs = timefreqs_alpha, topomap_args = dict(outlines='head', contours=0))


flipped_gave_lvsr_norm.data = np.subtract(gave_lvsr_norm.data, flipped_gave_lvsr_norm.data)
flipped_gave_lvsr_norm.plot_joint(timefreqs = timefreqs_alpha, topomap_args  = dict(outlines = 'head', contours = 0))



#for example looking across all channels at motor selection

#really just fucking around at this point because not sure quite how to get it to work,
gave_gmean    = mne.grand_average(alldata_gmean)
flipped_gmean = mne.grand_average(alldata_gmean);

gave_gmean.data = toverparam(alldata_gmean)
flipped_gmean.data = toverparam(alldata_gmean)
renaming_mapping = dict()
for i in range(len(chnames)):
    renaming_mapping[chnames[i]] = flippednames[i]
#mne.rename_channels(flipped_gmean.info, renaming_mapping)
import copy
cvsi = copy.deepcopy(gave_gmean)

flipped_gmean.reorder_channels(flippednames)
cvsi.data = np.subtract(gave_gmean.data, flipped_gmean.data)
cvsi.drop_channels(['RM'])
cvsi.plot(picks=['C4'], baseline = (None, None)) #C4  is contra to the left hand, which presses space bar to start response
cvsi.plot_joint(baseline = (-0.5, -0.3), topomap_args = dict(outlines='head', contours=0, vmin=-.5, vmax=.5),
                timefreqs = {(0,23):(.4, 14)})



