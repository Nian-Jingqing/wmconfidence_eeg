#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:25:07 2019

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
        for imodel in ['simple', 'simple_nogmean', 'simple_withmaineffects', 'simple_nogmean_withmaineffects']:
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

            cues   = tfr.metadata.cue.to_numpy()
            pside = tfr.metadata.pside.to_numpy()
            pside = np.where(pside == 0, 1, -1)


            probeleft = np.where(pside == 1, 1, 0)
            proberight = np.where(pside == -1, 1, 0)

            pleft_neut = np.where(np.logical_and(pside == 1, cues == 0), 1, 0)
            pleft_cued = np.where(np.logical_and(pside == 1, cues == 1), 1, 0)

            pright_neut = np.where(np.logical_and(pside == -1, cues == 0), 1, 0)
            pright_cued = np.where(np.logical_and(pside == -1, cues == 1), 1, 0)
            DT = tfr.metadata.DT.to_numpy()
            error = tfr.metadata.absrdif.to_numpy()
            error = np.where(error == 0, 0.1, error)
            confwidth = tfr.metadata.confwidth.to_numpy()
            cw = np.where(confwidth == 0, 0.1, confwidth)


            regressors = list()

            if imodel == 'simple':

                regressors.append(glm.regressors.ParametricRegressor(name = 'grand mean', values = trials, preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pleft_neut, codes = 1, name = 'probe left neutral'))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pleft_cued, codes = 1, name = 'probe left cued'))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pright_neut, codes = 1, name = 'probe right neutral'))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pright_cued, codes = 1, name = 'probe right cued'))

                contrasts = list()
                contrasts.append(glm.design.Contrast([1, 0, 0, 0, 0], 'grand mean'))
                contrasts.append(glm.design.Contrast([0, 1, 0, 0, 0], 'pleft_neutral'))
                contrasts.append(glm.design.Contrast([0, 0, 1, 0, 0], 'pleft_cued'))
                contrasts.append(glm.design.Contrast([0, 0, 0, 1, 0], 'pright_neutral'))
                contrasts.append(glm.design.Contrast([0, 0, 0, 0, 1], 'pright_cued'))
                contrasts.append(glm.design.Contrast([0,-1, 1, 0, 0], 'pleft_cvsn'))
                contrasts.append(glm.design.Contrast([0, 0, 0,-1, 1], 'pright_cvsn'))
                contrasts.append(glm.design.Contrast([0, 0, 1, 0,-1], 'plvsr_cued'))


            elif imodel == 'simple_nogmean':
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pleft_neut, codes = 1, name = 'probe left neutral'))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pleft_cued, codes = 1, name = 'probe left cued'))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pright_neut, codes = 1, name = 'probe right neutral'))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pright_cued, codes = 1, name = 'probe right cued'))

                contrasts = list()
                contrasts.append(glm.design.Contrast([ 1, 0, 0, 0], 'pleft_neutral'))
                contrasts.append(glm.design.Contrast([ 0, 1, 0, 0], 'pleft_cued'))
                contrasts.append(glm.design.Contrast([ 0, 0, 1, 0], 'pright_neutral'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 1], 'pright_cued'))
                contrasts.append(glm.design.Contrast([-1, 1, 0, 0], 'pleft_cvsn'))
                contrasts.append(glm.design.Contrast([ 0, 0,-1, 1], 'pright_cvsn'))
                contrasts.append(glm.design.Contrast([ 0, 1, 0,-1], 'plvsr_cued'))

            elif imodel == 'simple_withmaineffects':
                regressors.append(glm.regressors.ParametricRegressor(name = 'grand mean', values = trials, preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pleft_neut, codes = 1, name = 'probe left neutral'))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pleft_cued, codes = 1, name = 'probe left cued'))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pright_neut, codes = 1, name = 'probe right neutral'))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pright_cued, codes = 1, name = 'probe right cued'))
                regressors.append(glm.regressors.ParametricRegressor(name = 'DT', values = np.log(DT), preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'error', values = np.log(error), preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'confidence', values = np.log(cw), preproc = 'z', num_observations = nobs))

                contrasts = list()
                contrasts.append(glm.design.Contrast([ 1, 0, 0, 0, 0, 0, 0, 0], 'grand mean'))
                contrasts.append(glm.design.Contrast([ 0, 1, 0, 0, 0, 0, 0, 0], 'pleft_neutral'))
                contrasts.append(glm.design.Contrast([ 0, 0, 1, 0, 0, 0, 0, 0], 'pleft_cued'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 1, 0, 0, 0, 0], 'pright_neutral'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 1, 0, 0, 0], 'pright_cued'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 1, 0, 0], 'DT'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 1, 0], 'error'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 1], 'confidence'))
                contrasts.append(glm.design.Contrast([ 0,-1, 1, 0, 0, 0, 0, 0], 'pleft_cvsn'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0,-1, 1, 0, 0, 0], 'pright_cvsn'))
                contrasts.append(glm.design.Contrast([ 0, 0, 1, 0,-1, 0, 0, 0], 'plvsr_cued'))

            elif imodel == 'simple_nogmean_withmaineffects':
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pleft_neut, codes = 1, name = 'probe left neutral'))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pleft_cued, codes = 1, name = 'probe left cued'))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pright_neut, codes = 1, name = 'probe right neutral'))
                regressors.append(glm.regressors.CategoricalRegressor(category_list = pright_cued, codes = 1, name = 'probe right cued'))
                regressors.append(glm.regressors.ParametricRegressor(name = 'DT', values = np.log(DT), preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'error', values = np.log(error), preproc = 'z', num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'confidence', values = np.log(cw), preproc = 'z', num_observations = nobs))

                contrasts = list()
                contrasts.append(glm.design.Contrast([ 1, 0, 0, 0, 0, 0, 0], 'pleft_neutral'))
                contrasts.append(glm.design.Contrast([ 0, 1, 0, 0, 0, 0, 0], 'pleft_cued'))
                contrasts.append(glm.design.Contrast([ 0, 0, 1, 0, 0, 0, 0], 'pright_neutral'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 1, 0, 0, 0], 'pright_cued'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 1, 0, 0], 'DT'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 1, 0], 'error'))
                contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 1], 'confidence'))
                contrasts.append(glm.design.Contrast([-1, 1, 0, 0, 0, 0, 0], 'pleft_cvsn'))
                contrasts.append(glm.design.Contrast([ 0, 0,-1, 1, 0, 0, 0], 'pright_cvsn'))
                contrasts.append(glm.design.Contrast([ 0, 1, 0,-1, 0, 0, 0], 'plvsr_cued'))


            glmdes = glm.design.GLMDesign.initialise(regressors, contrasts); #designs.append(glmdes)
            #if iglm == 0:
            #    glmdes.plot_summary()

            total_nave = len(tfr)
            neut_nave  = len(tfr['cue==0'])
            cued_nave  = len(tfr['cue==1'])
            neut_pleft_nave = len(tfr['cuetrig == 11'])
            cued_pleft_nave = len(tfr['cuetrig == 13'])
            neut_pright_nave = len(tfr['cuetrig == 12'])
            cued_pright_nave = len(tfr['cuetrig == 14'])
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
#                if iname == 1:
#                    nave = neut_pleft_nave
#                elif iname == 2:
#                    nave = cued_pleft_nave
#                elif iname == 3:
#                    nave = neut_pright_nave
#                elif iname == 4:
#                    nave = cued_pright_nave
#                else:
                nave = total_nave #this doesn't even matter at the moment really


                tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                          data = np.squeeze(model.copes[iname,:,:,:]))
    #            tfr_betas.plot_joint(title = '%s, betas'%(name),
    #                                 timefreqs = {
    #                                         (-1.1, 10):(.4, 4), (-.9,10):(.4,4),(-.7,10):(.4,4), #some cue locked topos
    #                                         (.4, 10):(.4, 4),(.6, 10):(.4, 4),(.8, 10):(.4, 4)}, #baseline = (-2, -1.8),
    #                                 topomap_args = dict(outlines = 'head', contours = 0,
    #                                                     vmin = np.divide(np.min(tfr_betas.data),10), vmax = np.divide(np.min(tfr_betas.data), -10)))
                tfr_betas.save(fname = op.join(param['path'], 'glms', 'probe', imodel , 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
                del(tfr_betas)

                tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                          data = np.squeeze(model.get_tstats()[iname,:,:,:]))
    #            deepcopy(tfr_tstats).drop_channels(['RM']).plot_joint(title = '%s, tstats'%(name),
    #                                 timefreqs = {
    #                                         (-1.1, 10):(.4, 4), (-.9,10):(.4,4),(-.7,10):(.4,4), #some cue locked topos
    #                                         (.4, 10):(.4, 4),(.6, 10):(.4, 4),(.8, 10):(.4, 4)}, #baseline = (-2, -1.8),
    #                                 topomap_args = dict(outlines = 'head', contours = 0,
    #                                                     vmin = -2, vmax = 2))
                tfr_tstats.save(fname = op.join(param['path'], 'glms', 'probe', imodel, 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
                del(tfr_tstats)

                tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                          data = np.squeeze(model.varcopes[iname,:,:,:]))
                tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'probe', imodel, 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
                del(tfr_varcopes)

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            del(glmdes)
            del(model)
#%%
