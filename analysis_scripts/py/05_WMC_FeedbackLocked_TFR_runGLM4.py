#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 14:11:13 2019

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


subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
subs = np.array([        4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18])

glmstorun = 2
for i in subs:
    for iglm in range(glmstorun):
        print('\n\nrunning glm %d/%d'%(iglm+1, glmstorun))
        print('-- working on subject ' + str(i) +' --\n\n')
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub)

        #get tfr
        tfr             = mne.time_frequency.read_tfrs(fname=param['fblocked_tfr'])[0]
        tfr.metadata    = pd.read_csv(param['fblocked_tfr_meta'], index_col=None) #read in and attach metadata

        #some regressors need to have baselined data as the input because we aren't looking at a specific contrast
        #this is true for grand mean responses, neutral trials only, cued trials only (i.e. average neutral response)
        #also true for main effects of error and reaction time (and confidence if included) that are not carried by a side interaction
        #so i think it might be best if i run the same glm twice, but with one the data is baselined and in the others it isn't.

        # for just subject 7 and 9 i want to force rerunning the second glm (with baselines), so ...
        #if i == 9:
        #    iglm = iglm +1

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
        nobs = glmdata.num_observations

        regressors = list()


        alltrials = np.ones(len(tfr), dtype = 'int')
        cues  = tfr.metadata.cue.to_numpy()
        pside = tfr.metadata.pside.to_numpy()
        pside = np.where(pside == 0, 1, -1)

        error = tfr.metadata.absrdif.to_numpy()
        confwidth = tfr.metadata.confwidth.to_numpy() #confidence width in degrees, higher = less confident
        conf = np.radians(np.multiply(confwidth, -1)) #reverse the sign so higher (less negative) = more confident, then convert to radians so same scale as error
        confdiff = tfr.metadata.confdiff.to_numpy() #error awareness (metacognition) on that trial (or prediction error)
        neutral = np.where(cues == 0, 1, 0)

        targinconf = np.less_equal(confdiff,0)
        targoutsideconf = np.greater(confdiff,0)

        incorrvscorr = np.where(targinconf == 0, 1, -1)

        errorcorr   = np.where(targinconf == 1, error, np.nan) #error on trials where target was inside confidence interval (correct, or good feedback)
        errorincorr = np.where(targoutsideconf == 1, error, np.nan) #error on trials where target was outside confidence interval (incorrect, or bad feedback)

        confcorr    = np.where(targinconf == 1, conf, np.nan)
        confincorr  = np.where(targoutsideconf == 1, conf, np.nan)

        #need to z-score these separately to each other

        errorcorr   = np.divide(np.subtract(errorcorr, np.nanmean(errorcorr)), np.nanstd(errorcorr))
        errorincorr = np.divide(np.subtract(errorincorr, np.nanmean(errorincorr)), np.nanstd(errorincorr))
        confcorr    = np.divide(np.subtract(confcorr, np.nanmean(confcorr)), np.nanstd(confcorr))
        confincorr  = np.divide(np.subtract(confincorr, np.nanmean(confincorr)), np.nanstd(confincorr))

        #set nans back to zero so those trials are not modelled
        errorcorr   = np.where(np.isnan(errorcorr), 0, errorcorr)
        errorincorr = np.where(np.isnan(errorincorr), 0, errorincorr)
        confcorr    = np.where(np.isnan(confcorr), 0, confcorr)
        confincorr  = np.where(np.isnan(confincorr), 0, confincorr)

        confdiff_corr = np.where(targinconf == 1, confdiff, np.nan)
        confdiff_corr = np.multiply(confdiff_corr, -1) #flip the sign of underconfident trials
        #previously, lower values (more negative) relate to more confidence error
        #flipping the sign of this makes lower numbers = lower confidence error (better calibration on a trial)
        confdiff_corr = np.divide(np.subtract(confdiff_corr, np.nanmean(confdiff_corr)), np.nanstd(confdiff_corr))
        confdiff_corr = np.where(np.isnan(confdiff_corr), 0, confdiff_corr) #set nans to 0 to take out of model

        confdiff_incorr = np.where(targinconf == 0, confdiff, np.nan)
        #this is set up where larger numbers indicate larger confidence error, so leave this as it is
        confdiff_incorr = np.divide(np.subtract(confdiff_incorr, np.nanmean(confdiff_incorr)), np.nanstd(confdiff_incorr))
        confdiff_incorr = np.where(np.isnan(confdiff_incorr), 0, confdiff_incorr) #set nans to 0 to take out of model

        regressors.append( glm.regressors.ParametricRegressor(name = 'grand mean', values = alltrials, preproc = None, num_observations = nobs))
        regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 0, name = 'neutral'))
        regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 1, name = 'cued'))

        #regressors to look at here
        regressors.append( glm.regressors.ParametricRegressor(name = 'incorr vs corr', values = incorrvscorr,    preproc = None, num_observations = nobs))
        regressors.append( glm.regressors.ParametricRegressor(name = 'error-corr',     values = errorcorr,       preproc = None, num_observations = nobs))
        regressors.append( glm.regressors.ParametricRegressor(name = 'error-incorr',   values = errorincorr,     preproc = None, num_observations = nobs))
        regressors.append( glm.regressors.ParametricRegressor(name = 'conferr-corr',   values = confdiff_corr,   preproc = None, num_observations = nobs))
        regressors.append( glm.regressors.ParametricRegressor(name = 'conferr-incorr', values = confdiff_incorr, preproc = None, num_observations = nobs))
        regressors.append( glm.regressors.ParametricRegressor(name = 'pside',          values = pside,           preproc = None, num_observations = nobs))

        contrasts = list()

        contrasts.append( glm.design.Contrast([1, 0, 0, 0, 0, 0, 0, 0, 0], 'grand mean'))
        contrasts.append( glm.design.Contrast([0, 1, 0, 0, 0, 0, 0, 0, 0], 'neutral'))
        contrasts.append( glm.design.Contrast([0, 0, 1, 0, 0, 0, 0, 0, 0], 'cued'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 1, 0, 0, 0, 0, 0], 'incorr vs corr'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 0, 0, 0, 0], 'error_corr'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 1, 0, 0, 0], 'error_incorr'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 1, 0, 0], 'conferr_corr'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 0, 1, 0], 'conferr_incorr'))
        contrasts.append( glm.design.Contrast([0,-1, 1, 0, 0, 0, 0, 0, 0], 'cued vs neutral'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 1, 0, 0, 0], 'error'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 1, 1, 0], 'conferror'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 0, 1, 0, 0], 'confidence_corr'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 1, 0, 1, 0], 'confidence_incorr'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 1, 1, 1, 0], 'confidence'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0,-1, 1, 0, 0, 0], 'error_incorrvscorr'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0,-1, 1, 0], 'conferror_incorrvscorr'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0,-1, 1,-1, 1, 0], 'confidence_incorrvscorr'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 0, 0, 1], 'pside'))




        glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
        #glmdes.plot_summary()

        total_nave = len(tfr)
        neut_nave  = len(tfr['cue==0'])
        cued_nave  = len(tfr['cue==1'])
        underconf_nave = targinconf.sum()
        overconf_nave  = targoutsideconf.sum()
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
            if iname == 1:
                nave = neut_nave
            elif iname == 2:
                nave = cued_nave
            elif iname in [4, 6, 11]:
                nave = underconf_nave
            elif iname in [5, 7, 12]:
                nave = overconf_nave
            else:
                nave = total_nave


            tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.copes[iname,:,:,:]))
#            tfr_betas.plot_joint(title = '%s, betas'%(name),
#                                 timefreqs = {
#                                         (-1.1, 10):(.4, 4), (-.9,10):(.4,4),(-.7,10):(.4,4), #some cue locked topos
#                                         (.4, 10):(.4, 4),(.6, 10):(.4, 4),(.8, 10):(.4, 4)}, #baseline = (-2, -1.8),
#                                 topomap_args = dict(outlines = 'head', contours = 0,
#                                                     vmin = np.divide(np.min(tfr_betas.data),10), vmax = np.divide(np.min(tfr_betas.data), -10)))
            tfr_betas.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_glm4', 'wmc_' + param['subid'] + '_fblocked_tfr_'+ name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_betas)

            tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.get_tstats()[iname,:,:,:]))
#            deepcopy(tfr_tstats).drop_channels(['RM']).plot_joint(title = '%s, tstats'%(name),
#                                 timefreqs = {
#                                         (-1.1, 10):(.4, 4), (-.9,10):(.4,4),(-.7,10):(.4,4), #some cue locked topos
#                                         (.4, 10):(.4, 4),(.6, 10):(.4, 4),(.8, 10):(.4, 4)}, #baseline = (-2, -1.8),
#                                 topomap_args = dict(outlines = 'head', contours = 0,
#                                                     vmin = -2, vmax = 2))
            tfr_tstats.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_glm4', 'wmc_' + param['subid'] + '_fblocked_tfr_'+ name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_tstats)

#            tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
#                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
#            tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm_2_conf', 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
#            del(tfr_varcopes)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)
