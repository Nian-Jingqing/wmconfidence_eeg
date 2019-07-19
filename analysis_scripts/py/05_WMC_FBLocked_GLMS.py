#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:38:35 2019

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

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/glm')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1,2,4,5,6,7])
count=1

cued= []
neutral = []
cuedvsneutral = []
confdiffint = []


for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    #get tfr
    
    epochs = mne.read_epochs(fname=param['fblocked'], preload=True)
    
    epochs = epochs['DTcheck == 0 and clickresp == 1 and arraycueblink==0']

    #will do an automated process of looking for trials with heightened variance (noise) and output which trials to keep
    _, keeps = plot_AR(epochs, method = 'gesd', zthreshold = 1.5, p_out=.1, alpha = .05, outlier_side = 1)
    keeps = keeps.flatten()
    
    #get trials to discard based on the gesd approach
    discards = np.ones(len(epochs), dtype = 'bool')
    discards[keeps] = False
    epochs = epochs.drop(discards)
    epochs.set_channel_types({'RM':'misc'})
    
    
    
    epochs = epochs.apply_baseline(baseline = (-0.3,-0.1))
    epochs.set_eeg_reference(ref_channels=['RM'])
    
    #separate into cued left and right    
    glmdata = glm.data.TrialGLMData(data = epochs.get_data(), time_dim=2, sample_rate=1000)
    
    regressors = list()
    regressors.append( glm.regressors.CategoricalRegressor(category_list = epochs.metadata.cue.to_numpy(), codes = 0, name = 'neutral') )
    regressors.append( glm.regressors.CategoricalRegressor(category_list = epochs.metadata.cue.to_numpy(), codes = 1, name = 'cued') )

    cues        = epochs.metadata.cue.to_numpy()
    cues        = np.where(cues==0, -1, cues)
    confdifcues = np.multiply(epochs.metadata.confdiff.to_numpy(), cues)
    regressors.append( glm.regressors.ParametricRegressor(name='confdif x cue', values=confdifcues, preproc='z', num_observations=glmdata.num_observations))    


    contrasts = list()
    contrasts.append( glm.design.Contrast([1,  0, 0], 'neutral') )
    contrasts.append( glm.design.Contrast([0,  1, 0], 'cued') )
    contrasts.append( glm.design.Contrast([-1, 1, 0], 'cued vs neutral') )
    contrasts.append( glm.design.Contrast([0,  0, 1], 'confdiff x cue'))
    
    
    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
    #glmdes.plot_summary()
    
    model = glm.fit.OLSModel( glmdes, glmdata)
    
    #because we have epoched data, lets just get it back into mne asap
    
    #first we need to create an info structure
    evoked_info = mne.create_info(
            ch_names = epochs.ch_names,
            sfreq    = 1000, #data always acquired at 1Khz in amplifier
            ch_types = np.concatenate([np.repeat('eeg', 61), np.array(['misc']), np.repeat('eog', 2)])
            )
    
    
    epoch_betas_neutral = mne.EvokedArray(info = evoked_info,
                                          data = np.squeeze(model.copes[0,:,:]),
                                          tmin = epochs.tmin,
                                          nave = len(epochs['neutral_left']) + len(epochs['neutral_right']),
                                          kind = 'average')
    epoch_betas_neutral.set_montage('easycap-M1')
    epoch_betas_neutral.plot_joint(topomap_args=dict(outlines='head'), title='betas neutral trials, subject'+str(i))
    
    
    epoch_betas_cued = mne.EvokedArray(info = evoked_info,
                                       data = np.squeeze(model.copes[1,:,:]),
                                       tmin = epochs.tmin,
                                       nave = len(epochs['cued_left']) + len(epochs['cued_right']),
                                       kind = 'average'
                                       )
    epoch_betas_cued.set_montage('easycap-M1')
    epoch_betas_cued.plot_joint(topomap_args=dict(outlines='head'), title = 'betas cued trials, subject'+str(i))
    
    epoch_betas_cuedvsneut = mne.EvokedArray(info = evoked_info,
                                             data = np.squeeze(model.copes[2,:,:]),
                                             tmin = epochs.tmin,
                                             nave = len(epochs['cued_left']) + len(epochs['cued_right']),
                                             kind = 'average'
                                             )
    epoch_betas_cuedvsneut.set_montage('easycap-M1')
    epoch_betas_cuedvsneut.plot_joint(topomap_args=dict(outlines='head'), title='betas cued vs neutral, subject'+str(i))

    epoch_betas_confdiffint = mne.EvokedArray(info = evoked_info,
                                             data = np.squeeze(model.copes[3,:,:]),
                                             tmin = epochs.tmin,
                                             nave = len(epochs),
                                             kind = 'average'
                                             )
    epoch_betas_confdiffint.set_montage('easycap-M1')
    epoch_betas_confdiffint.plot_joint(topomap_args=dict(outlines='head'), title='betas confdiff interaction, subject'+str(i))

    neutral.append(epoch_betas_neutral)
    cued.append(epoch_betas_cued)
    cuedvsneutral.append(epoch_betas_cuedvsneut)
    confdiffint.append(epoch_betas_confdiffint)


#now have some structures that have all current subjects so we can do some grand averaging
neut_gave = mne.grand_average(neutral)
cued_gave = mne.grand_average(cued)
cvsn_gave = mne.grand_average(cuedvsneutral)
cdif_int_gave = mne.grand_average(confdiffint)

fig = cvsn_gave.plot_joint(topomap_args = dict(outlines = 'head'),
                     ts_args = dict(hline = [0]),
                     title = 'cued vs neutral grand average')
axes = fig.get_axes()
axes[0].axvline(x=0  , ymin = -3, ymax = 3, linestyle = '--', linewidth = .5, color = 'k')
axes[0].axvline(x=0.5, ymin = -3, ymax = 3, linestyle = '--', linewidth = .5, color = 'k')


fig = cdif_int_gave.plot_joint(topomap_args = dict(outlines = 'head'),
                     ts_args = dict(hline = [0]),
                     title = 'confdiff x cue interaction grand average')
axes = fig.get_axes()
axes[0].axvline(x=0  , ymin = -3, ymax = 3, linestyle = '--', linewidth = .5, color = 'k')
axes[0].axvline(x=0.5, ymin = -3, ymax = 3, linestyle = '--', linewidth = .5, color = 'k')

    

    
    