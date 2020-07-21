#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:38:01 2019

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
from scipy import stats
import seaborn as sns

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, nanzscore

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])


glms2run = 2 #1 with no baseline, one where tfr input data is baselined
for i in subs:
    for iglm in range(glms2run):
        print('\n\nrunning glm %d/%d'%(iglm+1, glms2run))
        print('-- working on subject ' + str(i) +' --\n\n')
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub)
        
        for laplacian in [True, False]:
            if laplacian:
                lapstr = 'laplacian_'
            else:
                lapstr = ''
            for legterm in [14, 40]:
                #get tfr
                if laplacian:
                    tfr = mne.time_frequency.read_tfrs(fname = param['cuelocked_tfr'].replace('cuelocked-tfr', 'cuelocked_laplacian_legterms%d-tfr'%(legterm))); tfr = tfr[0]; #read in data with surface laplacian filter applied to epoched data
                else:
                    tfr = mne.time_frequency.read_tfrs(fname=param['cuelocked_tfr']); tfr=tfr[0]
                tfr.metadata = pd.read_csv(param['cuelocked_tfr_meta'], index_col=None) #read in and attach metadata
        
        #        tmpleft  = deepcopy(tfr)['cue==1 and pside==0'].average()
        #        tmpright = deepcopy(tfr)['cue==1 and pside==1'].average()
        
                if iglm == 0:
                    addtopath = ''
                    baseline_input = False
                elif iglm == 1:
                    addtopath = '_baselined'
                    baseline_input = True
        
                if baseline_input:
                   print(' -- baselining the TFR data -- ')
                   tfr = tfr.apply_baseline((-2,-1.5))
        
                glmdata         = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 100)
                nobs = glmdata.num_observations
                trials = np.ones(glmdata.num_observations) #regressor for just grand mean response
        
                if iglm == 0:
                    print('\nSubject %02d has %03d trials\n'%(i, trials.size))
        
                cues   = tfr.metadata.cue.to_numpy()
                pside = tfr.metadata.pside.to_numpy()
                pside = np.where(pside == 0, 1, -1)
        
                regressors = list()
                probeleft = np.where(pside == 1, 1, 0)
                proberight = np.where(pside == -1, 1, 0)
        
                pleft_neut = np.where(np.logical_and(pside == 1, cues == 0), 1, 0)
                pleft_cued = np.where(np.logical_and(pside == 1, cues == 1), 1, 0)
        
                pright_neut = np.where(np.logical_and(pside == -1, cues == 0), 1, 0)
                pright_cued = np.where(np.logical_and(pside == -1, cues == 1), 1, 0)
        
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
        
                cleftnave  = len(tfr['cue==1 and pside==0'])
                crightnave = len(tfr['cue==1 and pside==1'])
                nleftnave  = len(tfr['cue==0 and pside==0'])
                nrightnave = len(tfr['cue==0 and pside==1'])
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
        
        
                    tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                              data = np.squeeze(model.copes[iname,:,:,:]))
                    deepcopy(tfr_betas).plot_joint(timefreqs = {(.4,10):(.4,.4), (.6,10):(.4,4), (.8,10):(.4,4)},
                                                       topomap_args = dict(outlines='skirt', contours = 0))#,baseline=(-2,-1.5))
                    tfr_betas.save(fname = op.join(param['path'], 'glms', 'cue', 'tfrglm1', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ lapstr + name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
                    del(tfr_betas)
        
                    tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                              data = np.squeeze(model.get_tstats()[iname,:,:,:]))
        #            deepcopy(tfr_tstats).plot_joint(timefreqs = {(.4,10):(.4,.4), (.6,10):(.4,4), (.8,10):(.4,4)},
        #                                               topomap_args = dict(outlines='skirt', contours = 0, vmin=-3, vmax=3))#,baseline=(-2,-1.5))
        
                    tfr_tstats.save(fname = op.join(param['path'], 'glms', 'cue', 'tfrglm1', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ lapstr + name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
                    del(tfr_tstats)
        
        #            tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
        #                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
        #            tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
        #            del(tfr_varcopes)
        
                #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
                del(glmdes)
                del(model)
