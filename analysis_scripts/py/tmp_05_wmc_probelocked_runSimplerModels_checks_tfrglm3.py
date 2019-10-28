#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:47:15 2019

@author: sammirc

the purpose of this script is to see if we can reduce the number of regressors in the model or not, and still get something meaningful
"""

import numpy as np
import scipy as sp
import pandas as pd
import mne
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from copy import deepcopy

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR


sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)
subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
subs = np.array([        4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16])
#subs = np.array([15, 16])
subs = np.array([17, 18])


#subs = np.array([10]) #encountered memory error in subject 7 so rerun from here
#%% only needs running if Probelocked TFR glms not already present
#subs = np.array([7])
glmstorun = 2
for i in subs:
    for iglm in range(glmstorun):
        print('\n\nrunning glm %d/%d'%(iglm+1, glmstorun))
        print('-- working on subject ' + str(i) +' --\n\n')
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub)
        #for imodel in ['simple', 'simple_withmaineffects']:
        for imodel in ['simple_effectsbycond', 'simple_effectsbycond_withmain']:
            print(' --- doing the %s model --- '%imodel)

            #get tfr
            tfr             = mne.time_frequency.read_tfrs(fname=param['probelocked_tfr'])[0]
            tfr.metadata    = pd.read_csv(param['probelocked_tfr_meta'], index_col=None) #read in and attach metadata
    
    
            if iglm == 0:
                addtopath = ''
                baseline_input = False
            elif iglm == 1:
                addtopath = '_baselined'
                baseline_input = True
    
            if baseline_input:
               print(' -- baselining the TFR data -- ')
               tfr = tfr.apply_baseline((-2.0, -1.7))
    
            glmdata         = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 100)
            nobs = glmdata.num_observations
    
            #get some behavioural things we're going to look at
            trials = np.ones(glmdata.num_observations) #regressor for just grand mean response
            #note, it seems like this is really useful in making some of the other regressors interpretable
            #as they then relate to unique variance associated with their respective condition
    
            cues   = tfr.metadata.cue.to_numpy()
            pside = tfr.metadata.pside.to_numpy()
            pside = np.where(pside == 0, 1, -1)
            
            cued = np.multiply(cues, pside)
            neutral = np.multiply(np.where(cues==0,1,0), pside)
    
            DT = tfr.metadata.DT.to_numpy()
            error = tfr.metadata.absrdif.to_numpy()
            error = np.where(error == 0, 0.1, error)
            confwidth = tfr.metadata.confwidth.to_numpy()
            cw = np.where(confwidth == 0, 0.1, confwidth)
            
            
            dt_cued = np.multiply(np.log(DT), cued) #cued left = +1, cued right = -1
            dt_neut = np.multiply(np.log(DT), neutral)
            
            err_cued = np.multiply(np.log(error), cued)
            err_neut = np.multiply(np.log(error), neutral)
            
            cw_cued = np.multiply(np.log(cw), cued)
            cw_neut = np.multiply(np.log(cw), neutral)

    
    
            regressors = list()
            
            if imodel == 'simple':
                regressors.append(glm.regressors.ParametricRegressor(name = 'grand mean',   values = trials,    preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'cued',         values = cued,      preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'neutral',      values = neutral,   preproc = None, num_observations = nobs))
        
                contrasts = list()
                contrasts.append(glm.design.Contrast([ 1, 0, 0], 'grand mean'))
                contrasts.append(glm.design.Contrast([ 0, 1, 0], 'cued lvsr'))
                contrasts.append(glm.design.Contrast([ 0, 0, 1], 'neutral lvsr'))
                contrasts.append(glm.design.Contrast([ 1, 1, 0], 'cued left'))
                contrasts.append(glm.design.Contrast([ 1, 0, 1], 'cued right'))
            
            elif imodel == 'simple_withmaineffects':
                regressors.append(glm.regressors.ParametricRegressor(name = 'grand mean',   values = trials,        preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'cued',         values = cued,          preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'neutral',      values = neutral,       preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'DT',           values = np.log(DT),    preproc = 'z',  num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'error',        values = np.log(error), preproc = 'z',  num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'confidence',   values = np.log(cw),    preproc = 'z',  num_observations = nobs))
        
                contrasts = list()
                contrasts.append(glm.design.Contrast([ 1, 0, 0, 0, 0, 0], 'grand mean'))
                contrasts.append(glm.design.Contrast([ 0, 1, 0, 0, 0, 0], 'cued lvsr'))
                contrasts.append(glm.design.Contrast([ 0, 0, 1, 0, 0, 0], 'neutral lvsr'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 1, 0, 0], 'DT'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 1, 0], 'error'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 1], 'confidence'))
                contrasts.append(glm.design.Contrast([ 1, 1, 0, 0, 0, 0], 'cued left'))
                contrasts.append(glm.design.Contrast([ 1, 0, 1, 0, 0, 0], 'cued right'))
            elif imodel == 'simple_effectsbycond':
                regressors.append(glm.regressors.ParametricRegressor(name = 'grand mean', values = trials,   preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'cued',       values = cued,     preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'neutral',    values = neutral,  preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'dt-cued',    values = dt_cued,  preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'dt-neut',    values = dt_neut,  preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'err-cued',   values = err_cued, preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'err-neut',   values = err_neut, preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'cw-cued',    values = cw_cued,  preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'cw-neut',    values = cw_neut,  preproc = 'z', num_observations = nobs))
                
                contrasts = list()
                contrasts.append(glm.design.Contrast([ 1, 0, 0, 0, 0, 0, 0, 0, 0], 'grand mean'))
                contrasts.append(glm.design.Contrast([ 0, 1, 0, 0, 0, 0, 0, 0, 0], 'cued lvsr'))
                contrasts.append(glm.design.Contrast([ 0, 0, 1, 0, 0, 0, 0, 0, 0], 'neutral lvsr'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 1, 0, 0, 0, 0, 0], 'dtcued'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 1, 0, 0, 0, 0], 'dtneut'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 1, 0, 0, 0], 'errorcued'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 1, 0, 0], 'errorneut'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 1, 0], 'confcued'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 1], 'confneut'))
            
            elif imodel == 'simple_effectsbycond_withmain':
                regressors.append(glm.regressors.ParametricRegressor(name = 'grand mean', values = trials,   preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'cued',       values = cued,     preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'neutral',    values = neutral,  preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'DT',         values = np.log(DT),      preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'error',      values = np.log(error),   preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'confidence', values = np.log(cw),      preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'dt-cued',    values = dt_cued,  preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'dt-neut',    values = dt_neut,  preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'err-cued',   values = err_cued, preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'err-neut',   values = err_neut, preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'cw-cued',    values = cw_cued,  preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'cw-neut',    values = cw_neut,  preproc = 'z', num_observations = nobs))
                
                contrasts = list()
                contrasts.append(glm.design.Contrast([ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'grand mean'))
                contrasts.append(glm.design.Contrast([ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'cued lvsr'))
                contrasts.append(glm.design.Contrast([ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'neutral lvsr'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'DT'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'error'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'confidence'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'dtcued'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'dtneut'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'errorcued'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'errorneut'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'confcued'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'confneut'))                    
                
                
                
                
    
            glmdes = glm.design.GLMDesign.initialise(regressors, contrasts); #designs.append(glmdes)
            #if iglm == 0:
            #    glmdes.plot_summary()
    
            total_nave = len(tfr)
#            neut_nave  = len(tfr['cue==0'])
#            cued_nave  = len(tfr['cue==1'])
#            neut_pleft_nave = len(tfr['cuetrig == 11'])
#            cued_pleft_nave = len(tfr['cuetrig == 13'])
#            neut_pright_nave = len(tfr['cuetrig == 12'])
#            cued_pright_nave = len(tfr['cuetrig == 14'])
            times = tfr.times
            freqs = tfr.freqs
            info = tfr.info
    
            del(tfr)
            print('\n - - - - -  running glm - - - - - \n')
            model = glm.fit.OLSModel( glmdes, glmdata )
    
            del(glmdata) #clear from RAM as not used from now on really
            names = glmdes.contrast_names
    
            for iname in range(len(glmdes.contrast_names)):
                name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name
    
                nave = total_nave #this doesn't even matter at the moment really
    
    
                tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                          data = np.squeeze(model.copes[iname,:,:,:]))
    #            tfr_betas.plot_joint(title = '%s, betas'%(name),
    #                                 timefreqs = {
    #                                         (-1.1, 10):(.4, 4), (-.9,10):(.4,4),(-.7,10):(.4,4), #some cue locked topos
    #                                         (.4, 10):(.4, 4),(.6, 10):(.4, 4),(.8, 10):(.4, 4)}, #baseline = (-2, -1.8),
    #                                 topomap_args = dict(outlines = 'head', contours = 0,
    #                                                     vmin = np.divide(np.min(tfr_betas.data),10), vmax = np.divide(np.min(tfr_betas.data), -10)))
                tfr_betas.save(fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm3', imodel , 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
                del(tfr_betas)
    
                tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                          data = np.squeeze(model.get_tstats()[iname,:,:,:]))
    #            deepcopy(tfr_tstats).drop_channels(['RM']).plot_joint(title = '%s, tstats'%(name),
    #                                 timefreqs = {
    #                                         (-1.1, 10):(.4, 4), (-.9,10):(.4,4),(-.7,10):(.4,4), #some cue locked topos
    #                                         (.4, 10):(.4, 4),(.6, 10):(.4, 4),(.8, 10):(.4, 4)}, #baseline = (-2, -1.8),
    #                                 topomap_args = dict(outlines = 'head', contours = 0,
    #                                                     vmin = -2, vmax = 2))
                tfr_tstats.save(fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm3', imodel, 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
                del(tfr_tstats)
    
                tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                          data = np.squeeze(model.varcopes[iname,:,:,:]))
                tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm3', imodel, 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
                del(tfr_varcopes)
    
            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            del(glmdes)
            del(model)
#%%
