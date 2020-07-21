#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:37:29 2019

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
    iglm = 1
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)

    #get tfr
    tfr = mne.time_frequency.read_tfrs(fname=param['arraylocked_tfr'])[0]
    tfr.metadata    = pd.read_csv(param['arraylocked_tfr_meta'], index_col=None) #read in and attach metadata
    tfr.drop_channels(['LM'])
    if iglm == 0:
        addtopath = ''
        baseline_input = False
    elif iglm == 1:
        addtopath = '_baselined'
        baseline_input = True

    if baseline_input:
       print(' -- baselining the TFR data -- ')
       tfr = tfr.apply_baseline((-0.525, -0.325))

    glmdata         = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 100)

    nobs = glmdata.num_observations
    trials = np.ones(glmdata.num_observations) #regressor for just grand mean response
    
    
    error = tfr.metadata.absrdif.to_numpy()
    confwidth = tfr.metadata.confwidth.to_numpy() #confidence width in degrees, higher = less confident
    conf = np.radians(np.multiply(confwidth, -1)) #reverse the sign so higher (less negative) = more confident, then convert to radians so same scale as error
    confdiff = tfr.metadata.confdiff.to_numpy() #error awareness (metacognition) on that trial (or prediction error)

    
    err = nanzscore(error) #dont worry about nans this is just normal zscore
    cw  = nanzscore(confwidth)
    confidence = nanzscore(conf)

    
    regressors = list()
    regressors.append(glm.regressors.ParametricRegressor(name = 'mean induced', values = trials, preproc = None, num_observations = nobs))
    regressors.append(glm.regressors.ParametricRegressor(name = 'error',        values = err,    preproc = None, num_observations = nobs))
    regressors.append(glm.regressors.ParametricRegressor(name = 'confwidth',    values = cw,     preproc = None, num_observations = nobs))

    contrasts = list()
    contrasts.append(glm.design.Contrast([  1, 0, 0], 'induced response'))#0
    contrasts.append(glm.design.Contrast([  0, 1, 0], 'error'))#1
    contrasts.append(glm.design.Contrast([  0, 0, 1], 'confidence width'))#2


    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
    #if iglm == 0:
    #    glmdes.plot_summary()

    times = tfr.times
    freqs = tfr.freqs
    info = tfr.info

    del(tfr)


    print('\n - - - - -  running glm - - - - - \n')
    model = glm.fit.OLSModel( glmdes, glmdata)
    
    del(glmdata) #clear from RAM as not used from now on really
#        contrasts = np.stack([np.arange(len(contrasts)), glmdes.contrast_names], axis=1)
    for iname in range(len(glmdes.contrast_names)):
        name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name

        nave = nobs


        tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                  data = np.squeeze(model.copes[iname,:,:,:]))
#        deepcopy(tfr_betas).plot_joint(timefreqs = {(.4,10):(.4,.4), (.6,10):(.4,4), (.8,10):(.4,4)},
#                                              topomap_args = dict(outlines='skirt', contours = 0))#,baseline=(-2,-1.5))
        tfr_betas.save(fname = op.join(param['path'], 'glms', 'array', 'tfrglm', 'wmc_' + param['subid'] + '_arraylocked_tfr_'+ name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
        del(tfr_betas)

        tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                  data = np.squeeze(model.get_tstats()[iname,:,:,:]))
#        deepcopy(tfr_tstats).plot_joint(timefreqs = {(.4,10):(.4,.4), (.6,10):(.4,4), (.8,10):(.4,4)},
#                 topomap_args = dict(outlines='skirt', contours = 0, vmin=-3, vmax=3))#,baseline=(-2,-1.5))

        tfr_tstats.save(fname = op.join(param['path'], 'glms', 'array', 'tfrglm', 'wmc_' + param['subid'] + '_arraylocked_tfr_'+ name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
        del(tfr_tstats)

#            tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
#                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
#            tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
#            del(tfr_varcopes)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    del(glmdes)
    del(model)
