#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:57:57 2019

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

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR


sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)
subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
subs = np.array([11,12,13,14])
subs = np.array([11, 15])
subs = np.array([16])
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

        #get tfr
        tfr             = mne.time_frequency.read_tfrs(fname=param['probelocked_tfr'])[0]
        tfr.metadata    = pd.read_csv(param['probelocked_tfr_meta'], index_col=None) #read in and attach metadata

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


        #get some behavioural things we're going to look at
        trials = np.ones(glmdata.num_observations) #regressor for just grand mean response
        cues   = tfr.metadata.cue.to_numpy()
        pside = tfr.metadata.pside.to_numpy()
        pside = np.where(pside == 0, 1, -1)

        regressors = list()
        regressors.append( glm.regressors.ParametricRegressor(name = 'grand mean', values = trials, preproc = None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 0, name = 'neutral'))
        regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 1, name = 'cued'))

        #set up a regressor for probed side only on neutral trials (e.g. any post-probe lateralisation in just neutral trials)
        neuttrls    = np.where(cues==0,1,0) #where no cue, set to 1 (neutral trial)
        cuedtrls    = cues                  #same as cues (where a cue was presented, set to 1)
        neutrallat  = np.multiply(pside, np.where(cues==0,1,0)) #set cued trials to 0, neutral to one (i.e. take out of model)
        cuedlat     = np.multiply(pside, cues) #cued trials == 1, neutral trials set to 0, so taken out of the average
        DT          = tfr.metadata.DT.to_numpy()
        error       = tfr.metadata.absrdif.to_numpy()
        DTxpside    = np.multiply(DT, pside)
        errorxpside = np.multiply(tfr.metadata.absrdif.to_numpy(), pside)
        confwidth   = np.radians(tfr.metadata.confwidth.to_numpy()) #make to radians so on same scale as error
        confwidth   = np.multiply(confwidth, -1) #reverse the sign of this so larger numbers (less negative) are higher confidence
        confwidthxpside = np.multiply(confwidth, pside)
        confneut    = np.multiply(confwidth, neuttrls)
        confcued    = np.multiply(confwidth, cuedtrls)

        regressors.append( glm.regressors.ParametricRegressor(name = 'pside-neutral',      values = neutrallat,      preproc = None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'pside-cued',         values = cuedlat,         preproc = None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'DT',                 values = DT,              preproc = 'z',  num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'error',              values = error,           preproc = 'z',  num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'pside',              values = pside,           preproc = None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'DTxpside',           values = DTxpside,        preproc = 'z',  num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'errorxpside',        values = errorxpside,     preproc = 'z',  num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'confidence',         values = confwidth,       preproc = 'z',  num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'confidencexpside',   values = confwidthxpside, preproc = 'z',  num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'confidence-neutral', values= confneut,         preproc = 'z',  num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'confidence-cued',    values = confcued,        preproc = 'z',  num_observations = glmdata.num_observations))

        contrasts = list()
        contrasts.append(glm.design.Contrast([ 1,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0], 'grand mean')   )
        contrasts.append(glm.design.Contrast([ 0,  1,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0], 'neutral')      )
        contrasts.append(glm.design.Contrast([ 0,  0,  1,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0], 'cued')         )
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  1,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0], 'pside neutral'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  1,  0, 0, 0, 0, 0, 0, 0, 0, 0], 'pside cued')   )
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  1, 0, 0, 0, 0, 0, 0, 0, 0], 'DT')           )
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  0, 1, 0, 0, 0, 0, 0, 0, 0], 'error')        )
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  0, 0, 1, 0, 0, 0, 0, 0, 0], 'pside')        )
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  0, 0, 0, 1, 0, 0, 0, 0, 0], 'DT x pside')   )
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  0, 0, 0, 0, 1, 0, 0, 0, 0], 'error x pside'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 1, 0, 0, 0], 'confidence'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 1, 0, 0], 'confidence x pside'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 1, 0], 'confidence neutral'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 1], 'confidence cued'))

        glmdes = glm.design.GLMDesign.initialise(regressors, contrasts); #designs.append(glmdes)
        #if iglm == 0:
        #    glmdes.plot_summary()

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
            tfr_betas.save(fname = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_'+ name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_betas)

            tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.get_tstats()[iname,:,:,:]))
            tfr_tstats.save(fname = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_'+ name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_tstats)

            tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
            tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'probe', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_varcopes)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)
