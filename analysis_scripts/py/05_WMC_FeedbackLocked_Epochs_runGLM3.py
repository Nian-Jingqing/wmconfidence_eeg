#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:15:49 2019

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
from wmConfidence_funcs import gesd, plot_AR, nanzscore

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/glm')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])


#%% only needs running if cuelocked TFR glms not already present
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    for laplacian in [True, False]:
        #get the epoched data
        epochs = mne.read_epochs(fname = param['fblocked'], preload = True) #this is loaded in with the metadata
    #    epochs.set_eeg_reference(['RM']) #don't need this as the re-referencing is done in the combine session script, after concatenation and before trial rejection
        
        #based on photodiode testing, there is a 25ms delay from trigger onset to maximal photodiode onset, so lets adjust times here
        epochs.shift_time(tshift = -0.025, relative = True)
        
        epochs.apply_baseline((-.25, 0)) #baseline 250ms prior to feedback
        epochs.resample(500) #resample to 500Hz
        if laplacian:
            epochs = mne.preprocessing.compute_current_source_density(epochs, stiffness=4)
            lapstr = 'laplacian_'
        else:
            lapstr = ''
        ntrials = len(epochs)
        print('\nSubject %02d has %03d trials\n'%(i, ntrials))
        
        #will do an automated process of looking for trials with heightened variance (noise) and output which trials to keep
        glmdata         = glm.data.TrialGLMData(data = epochs.get_data(), time_dim = 2, sample_rate = 500)
        nobs = glmdata.num_observations
        trials = np.ones(nobs)
    
        cues = epochs.metadata.cue.to_numpy()
        error = epochs.metadata.absrdif.to_numpy()
        confwidth = epochs.metadata.confwidth.to_numpy() #confidence width in degrees, higher = less confident
        conf = np.radians(np.multiply(confwidth, -1)) #reverse the sign so higher (less negative) = more confident, then convert to radians so same scale as error
        confdiff = np.radians(epochs.metadata.confdiff.to_numpy()) #error awareness (metacognition) on that trial (or prediction error)
        neutral = np.where(cues == 0, 1, 0)
        
        targinconf = np.less_equal(confdiff,0)
        targoutsideconf = np.greater(confdiff,0)
        incorrvscorr = np.where(targinconf == 0, 1, -1)
        
        errorcorr   = nanzscore(np.where(targinconf == 1, error, np.nan))
        errorincorr = nanzscore(np.where(targoutsideconf == 1, error, np.nan))
        
        confcorr    = nanzscore(np.where(targinconf == 1, conf, np.nan))
        confincorr  = nanzscore(np.where(targoutsideconf == 1, conf, np.nan))
        
        pside = epochs.metadata.pside.to_numpy()
        pside = np.where(pside == 0, 1, -1)
        
        #add regressors to the model
        
        regressors = list()
        regressors.append(glm.regressors.CategoricalRegressor(category_list = targinconf,      codes = 1, name = 'correct'))
        regressors.append(glm.regressors.CategoricalRegressor(category_list = targoutsideconf, codes = 1, name = 'incorrect'))
        regressors.append( glm.regressors.ParametricRegressor(name = 'error-correct',        values = errorcorr,   preproc = None, num_observations = nobs))
        regressors.append( glm.regressors.ParametricRegressor(name = 'error-incorrect',      values = errorincorr, preproc = None, num_observations = nobs))
        regressors.append( glm.regressors.ParametricRegressor(name = 'conf-correct',         values = confcorr,    preproc = None, num_observations = nobs))
        regressors.append( glm.regressors.ParametricRegressor(name = 'conf-incorrect',       values = confincorr,  preproc = None, num_observations = nobs))
        regressors.append( glm.regressors.ParametricRegressor(name = 'pside',                values = pside,       preproc = None, num_observations = nobs))
        
        contrasts = list()
        contrasts.append(glm.design.Contrast([ 1, 0, 0, 0, 0, 0, 0], 'correct'))
        contrasts.append(glm.design.Contrast([ 0, 1, 0, 0, 0, 0, 0], 'incorrect'))
        contrasts.append(glm.design.Contrast([ 0, 0, 1, 0, 0, 0, 0], 'error correct'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 1, 0, 0, 0], 'error incorrect'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 1, 0, 0], 'conf correct'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 1, 0], 'conf incorrect'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 1], 'pside'))
        contrasts.append(glm.design.Contrast([-1, 1, 0, 0, 0, 0, 0], 'incorr vs corr'))
        contrasts.append(glm.design.Contrast([ 0, 0,-1, 1, 0, 0, 0], 'error incorr vs corr'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0,-1, 1, 0], 'conf incorr vs corr'))
        contrasts.append(glm.design.Contrast([ 1, 1, 0, 0, 0, 0, 0], 'grandmean'))
        contrasts.append(glm.design.Contrast([ 0, 0, 1, 1, 0, 0, 0], 'error'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 1, 1, 0], 'conf'))
        
        
        
        glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
    #    glmdes.plot_summary()
    
        total_nave = len(epochs)
        neut_nave  = len(epochs['cue==0'])
        cued_nave  = len(epochs['cue==1'])
        underconf_nave = targinconf.sum()
        overconf_nave  = targoutsideconf.sum()
        tmin = epochs.tmin
        info = epochs.info
    
        del(epochs)
    
        print('\nrunning glm\n')
        model = glm.fit.OLSModel( glmdes, glmdata)
    
        del(glmdata) #clear from RAM as not used from now on really
        #contrasts = np.stack([np.arange(len(contrasts)), glmdes.contrast_names], axis=1)
        for iname in range(len(glmdes.contrast_names)):
            name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name
            if iname in [0, 2, 4]:
                nave = underconf_nave
            elif iname in [1, 3, 5]:
                nave = overconf_nave
            else:
                nave = total_nave
    
    
            tl_betas = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
                                                      data = np.squeeze(model.copes[iname,:,:]))
    #        deepcopy(tl_betas).drop_channels(['RM']).plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0))
            tl_betas.save(fname = op.join( param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'+ lapstr + name + '_betas-ave.fif'))
            del(tl_betas)
    
            tl_tstats = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
                                                      data = np.squeeze(model.get_tstats()[iname,:,:]))
            tl_tstats.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'+ lapstr + name + '_tstats-ave.fif'))
            del(tl_tstats)
    
    #        tl_varcopes = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
    #                                                  data = np.squeeze(model.varcopes[iname,:,:]))
    #        tl_varcopes.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_varcopes-ave.fif'))
    #        del(tl_varcopes)
    
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)
