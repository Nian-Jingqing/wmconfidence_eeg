#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:27:38 2019

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

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri/analysis_scripts')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/glm')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1, 2, 4, 5, 6, 7, 8, 9, 10])

#%%   if the betas are already saved ...
alldata_gmean    = []; alldata_gmean_t    = []
alldata_neutral  = []; alldata_neutral_t  = []
alldata_cued     = []; alldata_cued_t     = []
alldata_DT       = []; alldata_DT_t       = []
alldata_pside    = []; alldata_pside_t    = []
alldata_cvsn     = []; alldata_cvsn_t     = []
alldata_dtxpside = []; alldata_dtxpside_t = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    #read in betas
    gmean    = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_grandmean_betas-tfr.h5'))[0]
    neutral  = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_neutral_betas-tfr.h5'))[0]
    cued     = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_cued_betas-tfr.h5'))[0]
    DT       = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_DT_betas-tfr.h5'))[0]
    pside    = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_pside_betas-tfr.h5'))[0]
    cvsn     = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_cvsn_betas-tfr.h5'))[0]
    dtxpside = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_DTxpside_betas-tfr.h5'))[0]
    
    alldata_gmean.append(gmean)
    alldata_neutral.append(neutral)
    alldata_cued.append(cued)
    alldata_DT.append(DT)
    alldata_pside.append(pside)
    alldata_cvsn.append(cvsn)
    alldata_dtxpside.append(dtxpside)
    
    #read the tstats
    gmean_t    = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_grandmean_tstats-tfr.h5'))[0]
    neutral_t  = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_neutral_tstats-tfr.h5'))[0]
    cued_t     = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_cued_tstats-tfr.h5'))[0]
    DT_t       = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_DT_tstats-tfr.h5'))[0]
    pside_t    = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_pside_tstats-tfr.h5'))[0]
    cvsn_t     = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_cvsn_tstats-tfr.h5'))[0]
    dtxpside_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_DTxpside_tstats-tfr.h5'))[0]

    alldata_gmean_t.append(gmean_t)
    alldata_neutral_t.append(neutral_t)
    alldata_cued_t.append(cued_t)
    alldata_DT_t.append(DT_t)
    alldata_pside_t.append(pside_t)
    alldata_cvsn_t.append(cvsn_t)
    alldata_dtxpside_t.append(dtxpside_t)

#%% get flippings for channels if you want to flip sides for certain analyses
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


#%%

timefreqs = {(-.1, 23): (.2, 14)} #this is beta (centred 23 Hz, +/- 7hz -- 16-30Hz) just prior to response
            


gave_gmean   = mne.grand_average(alldata_gmean); gave_gmean.drop_channels(['RM'])
gave_gmean_t = mne.grand_average(alldata_gmean_t); gave_gmean_t.drop_channels(['RM'])


gave_gmean.plot_joint(
        title = 'grand mean betas - average response across all trials',
        topomap_args = dict(outlines = 'skirt', contours = 0),
        baseline = (None,None) #demean entire epoch
        )

gave_gmean_t.plot_joint(
        title = 'grand mean average of tstats - average response across all trials',
        topomap_args = dict(outlines = 'skirt', contours = 0),
        baseline = (None,None) #demean entire epoch
        )

#run t over betas for grand mean
gave_gmean = gave_gmean.apply_baseline((None,None))
tmp = np.empty(shape = (len(alldata_gmean), 62, 39, 750))
for i in range(len(subs)):
    tmp[i,:,:] = alldata_gmean[i].data
tovert = sp.stats.ttest_1samp(tmp, popmean = 0, axis = 0)
gave_gmean.data = tovert[0]

#replot with the t over betas
gave_gmean.plot_joint(
        title = ' average evoked response - t over betas' ,
        topomap_args = dict(outlines = 'head', contours = 0),
        baseline = (None,None) #demean entire epoch
        )


#for regressors that arent the grand mean (i.e. parametric regressors) we can visualise t over tstats to get a sense for effects and timecourses
#tstats are normalised across frequencies whereas betas aren't, and should show similar things really


gave_pside_t = mne.grand_average(alldata_pside_t); gave_pside_t.drop_channels(['RM'])

gave_pside_t.plot_joint(
        title = ' side of probed item - average tstat' ,
        topomap_args = dict(outlines = 'head', contours = 0),
        baseline = (None,None) #demean entire epoch
        )

gave_pside_t.data = toverparam(alldata_pside_t)
gave_pside_t.plot_joint(
        title = ' side of probed item - t over tstat' ,
        topomap_args = dict(outlines = 'head', contours = 0),
        timefreqs = {(0, 10):(.3, 4)}, #this looks at 150ms either side of response initiation, in alpha band, locked to response initiation (so should se similar topo to nat neuro paper supplement)
        baseline = (None,None), #demean entire epoch,
        picks = 28 #this is C3
        )



#look at decision time regressor

gave_dt_t = mne.grand_average(alldata_DT_t); gave_dt_t.drop_channels(['RM'])

gave_dt_t.plot_joint(
        title = 'main effect of decision time - average tstat',
        topomap_args = dict(outlines = 'head', contours = 0,vmin = -1.8, vmax = 1.8),
        vmin = -1.8, vmax = 1.8,
        timefreqs = timefreqs,
        baseline = (None,None) #demean entire epoch
        )

gave_dt_t.data = toverparam(alldata_DT_t)
gave_dt_t.plot_joint(
        title = 'main effect of decision time - t over tstats',
        topomap_args = dict(outlines = 'head', contours = 0,vmin = -1.8, vmax = 1.8),
        vmin = -1.8, vmax = 1.8,
        timefreqs = timefreqs,
        baseline = (None,None) #demean entire epoch
        )


#look at interaction between decision time and probed side
#now basically this should be lateralised (sign flipped on either side of visual electrodes) but not for motor (as always same hand)
#maybe this lets me look at DT as a function of visual hemifield attention?
# i'm actually not sure if this regressor is the right way of looking at it though

gave_dtxpside_t = mne.grand_average(alldata_dtxpside_t); gave_dtxpside_t.drop_channels(['RM'])

gave_dtxpside_t.data = toverparam(alldata_dtxpside_t)
gave_dtxpside_t.plot_joint(
        title = 'interaction between decision time and probed side - t over tstats',
        topomap_args = dict(outlines = 'head', contours = 0,vmin = -1.8, vmax = 1.8),
        vmin = -1.8, vmax = 1.8,
        timefreqs = timefreqs,
        baseline = (None,None) #demean entire epoch
        )

#plot betas just to take a look
gave_dtxpside = mne.grand_average(alldata_dtxpside); gave_dtxpside.drop_channels(['RM'])
gave_dtxpside.data = toverparam(alldata_dtxpside)
gave_dtxpside.plot_joint(
        title = 'interaction between decision time and probed side - average betas',
        topomap_args = dict(outlines = 'head', contours = 0),
        timefreqs = timefreqs,
        baseline = (None,None) #demean entire epoch
        )


#look at cued vs neutral trials

gave_cvsn = mne.grand_average(alldata_cvsn);
gave_cvsn_t = mne.grand_average(alldata_cvsn_t);

gave_cvsn.data = toverparam(alldata_cvsn); gave_cvsn.drop_channels(['RM'])
gave_cvsn.plot_joint(
        title = 'cued vs neutral trials -- t over betas',
        topomap_args = dict(outlines = 'head', contours = 0),
        timefreqs = timefreqs,
        baseline = (None,None)
        )










