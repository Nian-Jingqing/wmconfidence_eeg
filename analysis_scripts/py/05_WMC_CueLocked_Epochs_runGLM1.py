#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:55:08 2019

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


subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
laplacian=True

#%% only needs running if cuelocked TFR glms not already present
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)

    #get the epoched data
    epochs = mne.epochs.read_epochs(fname = param['cuelocked'].replace('cuelocked', 'cuelocked_cleaned'), preload = True)
    epochs.crop(tmin = -0.5, tmax = 1)
    epochs = mne.add_reference_channels(epochs, ref_channels = 'LM') #adds a channel of zeros for the left mastoid (as ref to itself in theory)
#    deepcopy(epochs['cued_left']).apply_baseline((-.25,0)).average().plot_joint(title = 'cued left, no re-ref', times = np.arange(.1,.7,.1))
#    deepcopy(epochs['cued_left']).set_eeg_reference(['RM']).apply_baseline((-.25,0)).average().plot_joint(title='cued_left, RM re-ref', times = np.arange(.1,.7,.1))
#    deepcopy(epochs['cued_left']).set_eeg_reference(['LM', 'RM']).apply_baseline((-.25,0)).average().plot_joint(title = 'cued left, ave mast ref', times = np.arange(.1,.7,.1))
        
    epochs.set_eeg_reference(ref_channels = ['LM','RM']) #re-reference average of the left and right mastoid now
#    epochs.set_eeg_reference('average')
    epochs.apply_baseline((-.25, 0)) #baseline 250ms prior to feedback
    epochs.resample(500) #resample to 500Hz
    if 'VEOG' in epochs.ch_names:
        epochs = epochs.drop_channels(['VEOG', 'HEOG'])
    ntrials = len(epochs)
    epochs.drop_channels(['RM', 'LM'])
    

    chnames = np.asarray(epochs.ch_names)
    chnamemapping = {}
    for x in range(len(chnames)):
        chnamemapping[chnames[x]] = chnames[x].replace('Z', 'z').replace('FP', 'Fp')
    mne.rename_channels(epochs.info, chnamemapping)
    
    
    epochs.set_montage('easycap-M1')
    
    if laplacian:
#        epochs.drop_channels(['RM', 'LM'])
        epochs = mne.preprocessing.compute_current_source_density(epochs, stiffness = 5)#default stiffness is 4
    

    glmdata         = glm.data.TrialGLMData(data = epochs.get_data(), time_dim = 2, sample_rate = 500)
    
    cues   = epochs.metadata.cue.to_numpy()
    pside = epochs.metadata.pside.to_numpy()
    #pside = np.where(pside == 0, 1, -1)

    regressors = list()
    probeleft = np.where(pside == 0, 1, 0)
    proberight = np.where(pside == 1, 1, 0)

    pleft_neut = np.where(np.logical_and(pside == 0, cues == 0), 1, 0)
    pleft_cued = np.where(np.logical_and(pside == 0, cues == 1), 1, 0)

    pright_neut = np.where(np.logical_and(pside == 1, cues == 0), 1, 0)
    pright_cued = np.where(np.logical_and(pside == 1, cues == 1), 1, 0)

    regressors.append(glm.regressors.CategoricalRegressor(category_list = pleft_neut, codes = 1, name = 'probe left neutral'))
    regressors.append(glm.regressors.CategoricalRegressor(category_list = pleft_cued, codes = 1, name = 'probe left cued'))

    regressors.append(glm.regressors.CategoricalRegressor(category_list = pright_neut, codes = 1, name = 'probe right neutral'))
    regressors.append(glm.regressors.CategoricalRegressor(category_list = pright_cued, codes = 1, name = 'probe right cued'))

    contrasts = list()
    contrasts.append(glm.design.Contrast([  1, 0, 0, 0], 'pleft_neutral'))#0
    contrasts.append(glm.design.Contrast([  0, 1, 0, 0], 'pleft_cued'))#1
    contrasts.append(glm.design.Contrast([  0, 0, 1, 0], 'pright_neutral'))#2
    contrasts.append(glm.design.Contrast([  0, 0, 0, 1], 'pright_cued'))#3
    contrasts.append(glm.design.Contrast([ -1, 1, 0, 0], 'clvsn'))#4
    contrasts.append(glm.design.Contrast([  0, 0,-1, 1], 'crvsn'))#5
    contrasts.append(glm.design.Contrast([  0, 1, 0,-1], 'clvsr'))#6
    contrasts.append(glm.design.Contrast([  1, 0, 1, 0], 'neutral'))#7
    contrasts.append(glm.design.Contrast([  0, 1, 0, 1], 'cued'))#8
    contrasts.append(glm.design.Contrast([ -1, 1,-1, 1], 'cuedvsneut'))#9


    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
    #if iglm == 0:
    #    glmdes.plot_summary()

    cleftnave  = len(epochs['cue==1 and pside==0'])
    crightnave = len(epochs['cue==1 and pside==1'])
    nleftnave  = len(epochs['cue==0 and pside==0'])
    nrightnave = len(epochs['cue==0 and pside==1'])
    tmin = epochs.tmin
    info = epochs.info

    del(epochs)


    print('\n - - - - -  running glm - - - - - \n')
    model = glm.fit.OLSModel( glmdes, glmdata)

    del(glmdata) #clear from RAM as not used from now on really
#        contrasts = np.stack([np.arange(len(contrasts)), glmdes.contrast_names], axis=1)
    for iname in range(len(glmdes.contrast_names)):
        name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name

        if iname in [0]:
            nave = nleftnave
        elif iname in [1]:
            nave = cleftnave
        elif iname in [2]:
            nave = nrightnave
        elif iname in [3]:
            nave = crightnave
        elif iname in [4]:
            nave = cleftnave + nleftnave
        elif iname in [5]:
            nave = crightnave + nrightnave
        elif iname in [6,8]:
            nave = cleftnave + crightnave
        elif iname in [7]:
            nave = nleftnave + nrightnave
        else:
            nave = cleftnave + crightnave + nleftnave + nrightnave


        tl_betas = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
                                   data = np.squeeze(model.copes[iname,:,:]))
        deepcopy(tl_betas).plot_joint(topomap_args = dict(outlines = 'head', contours = 0), times=np.arange(0.1,0.6,0.1))
        tl_betas.save(fname = op.join(param['path'], 'glms', 'cue', 'epochsglm1', 'wmc_' + param['subid'] + '_cuelocked_tl_laplacian_'+ name + '_betas-ave.fif'))
        del(tl_betas)

        tl_tstats = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
                                                  data = np.squeeze(model.get_tstats()[iname,:,:]))
        tl_tstats.save(fname = op.join(param['path'], 'glms', 'cue', 'epochsglm1', 'wmc_' + param['subid'] + '_cuelocked_tl_laplacian_'+ name + '_tstats-ave.fif'))
        del(tl_tstats)

#            tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
#                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
#            tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
#            del(tfr_varcopes)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    del(glmdes)
    del(model)
    
    
    