#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 11:41:06 2019

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


sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, nanzscore

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm


wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
#%% only needs running if cuelocked TFR glms not already present
glmstorun = 2
for i in subs:
    for iglm in range(glmstorun):
        print('\n\nrunning glm %d/%d'%(iglm+1, glmstorun))
        print('\n\nworking on subject ' + str(i) +'\n\n')
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub)

        for laplacian in [True, False]:
            if laplacian:
                lapstr = 'laplacian_'
                stiff = 4
            else:
                lapstr = ''
            if laplacian:
                tfr = mne.time_frequency.read_tfrs(fname = param['fblocked_tfr'].replace('fblocked-tfr', 'fblocked_laplacian_stiffness%d-tfr'%(stiff))); tfr = tfr[0]; #read in data with surface laplacian filter applied to epoched data
            else:
                tfr = mne.time_frequency.read_tfrs(fname=param['fblocked_tfr']); tfr=tfr[0]
            tfr.metadata = pd.read_csv(param['fblocked_tfr_meta'], index_col=None) #read in and attach metadata
        
            if iglm == 0:
                addtopath = ''
                baseline_input = False
            elif iglm == 1:
                addtopath = '_baselined'
                baseline_input = True
    
            if baseline_input:
               print(' -- baselining the TFR data -- ')
               tfr = tfr.apply_baseline((-0.5, -0.3))
    
            glmdata         = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 100)
    
            ntrials = len(tfr)
    
    
            nobs        = glmdata.num_observations
            alltrials = np.ones(ntrials)
            cues = tfr.metadata.cue.to_numpy()
            error = tfr.metadata.absrdif.to_numpy()
            confwidth = tfr.metadata.confwidth.to_numpy() #confidence width in degrees, higher = less confident
            conf = np.radians(np.multiply(confwidth, -1)) #reverse the sign so higher (less negative) = more confident, then convert to radians so same scale as error
            confdiff = np.radians(tfr.metadata.confdiff.to_numpy()) #error awareness (metacognition) on that trial (or prediction error)
            neutral = np.where(cues == 0, 1, 0)
    
            targinconf = np.less_equal(confdiff,0)
            targoutsideconf = np.greater(confdiff,0)
            incorrvscorr = np.where(targinconf == 0, 1, -1)
    
            errorcorr   = nanzscore(np.where(targinconf == 1, error, np.nan))
            errorincorr = nanzscore(np.where(targoutsideconf == 1, error, np.nan))
    
            confcorr    = nanzscore(np.where(targinconf == 1, conf, np.nan))
            confincorr  = nanzscore(np.where(targoutsideconf == 1, conf, np.nan))
    
            pside = tfr.metadata.pside.to_numpy()
            pside = np.where(pside == 0, 1, -1)
    
            #add regressors to the model
    
            regressors = list()
            contrasts  = list()
            
            regressors.append( glm.regressors.ParametricRegressor(name = 'grandmean', values = alltrials, preproc = None, num_observations = nobs))
            regressors.append( glm.regressors.CategoricalRegressor(category_list = targinconf,      codes = 1, name = 'correct'))
            regressors.append( glm.regressors.CategoricalRegressor(category_list = targoutsideconf, codes = 1, name = 'incorrect'))
            regressors.append( glm.regressors.ParametricRegressor(name = 'error-correct',        values = errorcorr,   preproc = None, num_observations = nobs))
            regressors.append( glm.regressors.ParametricRegressor(name = 'error-incorrect',      values = errorincorr, preproc = None, num_observations = nobs))
            regressors.append( glm.regressors.ParametricRegressor(name = 'conf-correct',         values = confcorr,    preproc = None, num_observations = nobs))
            regressors.append( glm.regressors.ParametricRegressor(name = 'conf-incorrect',       values = confincorr,  preproc = None, num_observations = nobs))
            regressors.append( glm.regressors.ParametricRegressor(name = 'pside',                values = pside,       preproc = None, num_observations = nobs)) #this can probs be removed tbh
    
            contrasts.append(glm.design.Contrast([ 1, 0, 0, 0, 0, 0, 0, 0], 'grandmean'))
            contrasts.append(glm.design.Contrast([ 0, 1, 0, 0, 0, 0, 0, 0], 'correct'))
            contrasts.append(glm.design.Contrast([ 0, 0, 1, 0, 0, 0, 0, 0], 'incorrect'))
            contrasts.append(glm.design.Contrast([ 0, 0, 0, 1, 0, 0, 0, 0], 'error correct'))
            contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 1, 0, 0, 0], 'error incorrect'))
            contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 1, 0, 0], 'conf correct'))
            contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 1, 0], 'conf incorrect'))
            contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 1], 'pside'))
            contrasts.append(glm.design.Contrast([ 0,-1, 1, 0, 0, 0, 0, 0], 'incorr vs corr'))
            contrasts.append(glm.design.Contrast([ 0, 0, 0,-1, 1, 0, 0, 0], 'error incorr vs corr'))
            contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0,-1, 1, 0], 'conf incorr vs corr'))
            contrasts.append(glm.design.Contrast([ 0, 0, 0, 1, 1, 0, 0, 0], 'error'))
            contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 1, 1, 0], 'conf'))
    
            glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
            # if iglm ==0:
            #     glmdes.plot_summary()
    
            total_nave = len(tfr)
            neut_nave  = len(tfr['cue==0'])
            cued_nave  = len(tfr['cue==1'])
            underconf_nave = targinconf.sum()
            overconf_nave  = targoutsideconf.sum()
            times = tfr.times
            freqs = tfr.freqs
            info = tfr.info
    
            del(tfr)
    
            print('\nrunning glm\n')
            model = glm.fit.OLSModel( glmdes, glmdata)
    
            del(glmdata) #clear from RAM as not used from now on really
            # contrasts = np.stack([np.arange(len(contrasts)), glmdes.contrast_names], axis=1)
            for iname in range(len(glmdes.contrast_names)):
                name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name
                if iname in [1, 3, 5]:
                    nave = underconf_nave
                elif iname in [2, 4, 6]:
                    nave = overconf_nave
                else:
                    nave = total_nave
    
    
                tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                          data = np.squeeze(model.copes[iname,:,:,:]))
    #            deepcopy(tfr_betas).plot_joint()
                tfr_betas.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ lapstr + name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
                del(tfr_betas)
    
                tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                          data = np.squeeze(model.get_tstats()[iname,:,:,:]))
    #            deepcopy(tfr_tstats).plot_joint()
                tfr_tstats.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ lapstr + name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
                del(tfr_tstats)
    
    #            tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
    #                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
    #            tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ lapstr + name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
    #            del(tfr_varcopes)
    
            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            del(glmdes)
            del(model)
