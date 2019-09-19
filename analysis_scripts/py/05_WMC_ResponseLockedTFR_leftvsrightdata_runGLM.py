#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:59:04 2019

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
from copy import deepcopy

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR


sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)
subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,16])

#subs = np.array([10]) #encountered memory error in subject 7 so rerun from here
#%% only needs running if resplocked TFR glms not already present
#subs = np.array([7])
glmstorun = 1
for i in subs:
    for iglm in range(glmstorun):
        print('\n\nrunning glm %d/%d'%(iglm+1, glmstorun))
        print('-- working on subject ' + str(i) +' --\n\n')
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub)

        #get tfr
        tfr             = mne.time_frequency.read_tfrs(fname=param['resplocked_tfr'])[0]
        tfr.metadata    = pd.read_csv(param['resplocked_tfr_meta'], index_col=None) #read in and attach metadata

        if iglm == 0:
            addtopath = ''
            baseline_input = False
        elif iglm == 1:
            addtopath = '_baselined'
            baseline_input = True

        if baseline_input:
           print(' -- baselining the TFR data -- ')
           tfr = tfr.apply_baseline((-0.5, -0.3))

        #perform the left vs right contrast in the single trial data here, and use that as the input data to the glm

        flipped = deepcopy(tfr)
        chids =         np.array([     1,  2,  3,
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
        flipped.data = flipped.data[:,flipids,:,:]

        chans_no_midline =np.array([           1,     3,
                                   4,  5,     7,  8,
                           9, 10, 11, 12,    14, 15, 16, 17,
                          18, 19, 20, 21,    23, 24, 25, 26,
                          27, 28, 29, 30,    32, 33, 34, 35,
                          36, 37, 38, 39,    41, 42, 43, 44,
                          45, 46, 47, 48,    50, 51, 52, 53,
                                  54, 55,    57, 58,
                                      59,    61,
                                          ])
        chans_no_midline = np.subtract(chans_no_midline, 1)


        tfr.data = np.subtract(tfr.data, flipped.data) #now this is right hand side minus left hand side of head (which is cvsi for attend left)
        #tfr.data = np.subtract(flipped.data, tfr.data) #this is right minus left (i.e. contra-ipsi for attend right)

        glmdata         = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 100)


        #get some behavioural things we're going to look at
        trials = np.ones(glmdata.num_observations) #regressor for just grand mean response
        cues   = tfr.metadata.cue.to_numpy()
        pside = tfr.metadata.pside.to_numpy()
        pside = np.where(pside == 0, 1, -1)

        regressors = list()
        regressors.append( glm.regressors.ParametricRegressor(name = 'grand mean', values = trials, preproc = None, num_observations = glmdata.num_observations))
        #regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 0, name = 'neutral'))
        regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 1, name = 'cued'))

        #set up a regressor for probed side only on neutral trials (e.g. any post-probe lateralisation in just neutral trials)
        neuttrls    = np.where(cues==0,1,0) #where no cue, set to 1 (neutral trial)
        cuedtrls    = cues                  #same as cues (where a cue was presented, set to 1)
        neutrallat  = np.multiply(pside, np.where(cues==0,1,0)) #set cued trials to 0, neutral to one (i.e. take out of model)
        cuedlat     = np.multiply(pside, cues) #cued trials == 1, neutral trials set to 0, so taken out of the average
        DT          = tfr.metadata.DT.to_numpy()
        error       = np.radians(tfr.metadata.absrdif.to_numpy())
        DTxpside    = np.multiply(DT, pside)
        errorxpside = np.multiply(tfr.metadata.absrdif.to_numpy(), pside)
        confwidth   = np.radians(tfr.metadata.confwidth.to_numpy()) #make to radians so on same scale as error
        confwidth   = np.multiply(confwidth, -1) #reverse the sign of this so larger numbers (less negative) are higher confidence
        confwidthxpside = np.multiply(confwidth, pside)

        regressors.append( glm.regressors.ParametricRegressor(name = 'pside-neutral',      values = neutrallat,      preproc = None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'pside-cued',         values = cuedlat,         preproc = None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'DT',                 values = DT,              preproc = 'z',  num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'error',              values = error,           preproc = 'z',  num_observations = glmdata.num_observations))
        #regressors.append( glm.regressors.ParametricRegressor(name = 'pside',              values = pside,           preproc = None, num_observations = glmdata.num_observations)) #cant include this as it will be rank deficient. make this at contrast level
        regressors.append( glm.regressors.ParametricRegressor(name = 'confidence',         values = confwidth,       preproc = 'z',  num_observations = glmdata.num_observations))

        #look at behaviour x pside in neutral and cued trials separately


        #get error on subset of trials
        error_neut = np.where(cues==0, error, np.nan) #and nan the rest
        error_cued = np.where(cues==1, error, np.nan)
        #multiply by pside to implicitly code the probed side interaction
        error_neut = np.multiply(error_neut, pside)
        error_cued = np.multiply(error_cued, pside)
        #zscore while ignoring the nans
        error_neut = np.divide(np.subtract(error_neut, np.nanmean(error_neut)), np.nanstd(error_neut))
        error_cued = np.divide(np.subtract(error_cued, np.nanmean(error_cued)), np.nanstd(error_cued))
        #now set the nan values to 0 to take them from the model
        error_neut = np.where(np.isnan(error_neut), 0, error_neut)
        error_cued = np.where(np.isnan(error_cued), 0, error_cued)
        #add these regressors into the glm. main effect across trials can be done at contrast level
        regressors.append( glm.regressors.ParametricRegressor(name = 'errorxpside-neutral', values = error_neut, preproc = None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'errorxpside-cued',    values = error_cued, preproc = None, num_observations = glmdata.num_observations))

        #get DT on subset of trials
        DT_neut = np.where(cues==0, DT, np.nan) #and nan the rest
        DT_cued = np.where(cues==1, DT, np.nan)
        #multiply by pside to implicitly code the probed side interaction
        DT_neut = np.multiply(DT_neut, pside)
        DT_cued = np.multiply(DT_cued, pside)
        #zscore while ignoring the nans
        DT_neut = np.divide(np.subtract(DT_neut, np.nanmean(DT_neut)), np.nanstd(DT_neut))
        DT_cued = np.divide(np.subtract(DT_cued, np.nanmean(DT_cued)), np.nanstd(DT_cued))
        #now set the nan values to 0 to take them from the model
        DT_neut = np.where(np.isnan(DT_neut), 0, DT_neut)
        DT_cued = np.where(np.isnan(DT_cued), 0, DT_cued)
        #add these regressors into the glm. main effect across trials can be done at contrast level
        regressors.append( glm.regressors.ParametricRegressor(name = 'DTxpside-neutral', values = DT_neut, preproc = None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'DTxpside-cued',    values = DT_cued, preproc = None, num_observations = glmdata.num_observations))

        #get confidence width on subset of trials
        conf_neut = np.where(cues==0, confwidth, np.nan) #and nan the rest
        conf_cued = np.where(cues==1, confwidth, np.nan)
        #multiply by pside to implicitly code the probed side interaction
        conf_neut = np.multiply(conf_neut, pside)
        conf_cued = np.multiply(conf_cued, pside)
        #zscore while ignoring the nans
        conf_neut = np.divide(np.subtract(conf_neut, np.nanmean(conf_neut)), np.nanstd(conf_neut))
        conf_cued = np.divide(np.subtract(conf_cued, np.nanmean(conf_cued)), np.nanstd(conf_cued))
        #now set the nan values to 0 to take them from the model
        conf_neut = np.where(np.isnan(conf_neut), 0, conf_neut)
        conf_cued = np.where(np.isnan(conf_cued), 0, conf_cued)
        #add these regressors into the glm. main effect across trials can be done at contrast level
        regressors.append( glm.regressors.ParametricRegressor(name = 'confxpside-neutral', values = conf_neut, preproc = None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'confxpside-cued',    values = conf_cued, preproc = None, num_observations = glmdata.num_observations))

        contrasts = list()
        contrasts.append(glm.design.Contrast([ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'grand mean')   )
        contrasts.append(glm.design.Contrast([ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'cued')         )
        contrasts.append(glm.design.Contrast([ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pside neutral'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pside cued')   )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'DT')           )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'error')        )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'confidence') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'errorxpside_neutral') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'errorxpside_cued') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'DTxpside_neutral') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'DTxpside_cued') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'confxpside_neutral') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'confxpside_cued') )
        contrasts.append(glm.design.Contrast([ 0, 0, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pside_nvsc') )
        contrasts.append(glm.design.Contrast([ 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pside_cvsn') )
        contrasts.append(glm.design.Contrast([ 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pside'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], 'errorxpside') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], 'DTxpside') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 'confxpside') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0], 'errorxpsidexcvsn') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0], 'DTxpsidexcvsn') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1], 'confxpsidexcvsn') )
        contrasts.append(glm.design.Contrast([ 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'neutral') )

        #contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'pside')        )

        glmdes = glm.design.GLMDesign.initialise(regressors, contrasts); #designs.append(glmdes)
        #if iglm == 0:
        #glmdes.plot_summary()

        total_nave = len(tfr)
        neut_nave  = len(tfr['cue==0'])
        cued_nave  = len(tfr['cue==1'])
        times = tfr.times
        freqs = tfr.freqs
        info = tfr.info

        del(tfr)
        print('\n - - - - -  running glm - - - - - \n')
        model = glm.fit.OLSModel( glmdes, glmdata )

        del(glmdata) #clear from RAM as not used from now on really

        for iname in range(len(glmdes.contrast_names)):
            name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name
            if iname in [1, 3, 12]:
                nave = neut_nave
            elif iname in [2, 4, 13]:
                nave = cued_nave
            else:
                nave = total_nave


            tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.copes[iname,:,:,:]))
#            tfr_betas.plot_joint(title = '%s, betas'%(name), timefreqs = {(.4, 10):(.4, 4),(.6, 10):(.4, 4),(.8, 10):(.4, 4),(1., 10):(.4, 4)},
#                                 topomap_args = dict(outlines = 'head', contours = 0,
#                                                     vmin = np.divide(np.min(tfr_betas.data),20), vmax = np.divide(np.min(tfr_betas.data), -20)))
            tfr_betas.save(fname = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_'+ name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_betas)

            tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.get_tstats()[iname,:,:,:]))
#            tfr_tstats.plot_joint(title = '%s, tstats'%(name),
#                                  timefreqs = {(.4, 10):(.4, 4),(.6, 10):(.4, 4),(.8, 10):(.4, 4),(1., 10):(.4, 4),
#                                               (.4, 22):(.4,14),(.6, 22):(.4,14),(.8, 22):(.4,14),(1., 22):(.4,14),},
#                                 topomap_args = dict(outlines = 'head', contours = 0, vmin=-1, vmax=1),
#                                 picks = chans_no_midline)
#            tfr_tstats.plot(picks=['C2','C4'], combine = 'mean', vmin=-2, vmax=2)
            tfr_tstats.save(fname = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_'+ name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_tstats)

#           # tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
#                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
#            #tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'response', 'tfr_glm_lvsr', 'wmConfidence_' + param['subid'] + '_resplocked_tfr_leftvsright_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
#            #del(tfr_varcopes)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)
