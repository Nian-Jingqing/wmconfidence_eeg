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


subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
subs = np.array([11,12,13,14])
subs = np.array([11, 15])
#subs = np.array([9, 10]) # memory error when i got to subject 9 :(
#%% only needs running if cuelocked TFR glms not already present
glmstorun = 2
for i in subs:
    for iglm in range(glmstorun):
        print('\n\nrunning glm %d/%d'%(iglm+1, glmstorun))
        print('\n\nworking on subject ' + str(i) +'\n\n')
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub)

        #get tfr
        tfr = mne.time_frequency.read_tfrs(fname=param['fblocked_tfr'])[0]
        tfr.metadata    = pd.read_csv(param['fblocked_tfr_meta'], index_col=None) #read in and attach metadata
        #some regressors need to have baselined data as the input because we aren't looking at a specific contrast
        #this is true for grand mean responses, neutral trials only, cued trials only (i.e. average neutral response)
        #also true for main effects of error and reaction time (and confidence if included) that are not carried by a side interaction
        #so i think it might be best if i run the same glm twice, but with one the data is baselined and in the others it isn't.

        #drop trials where previous trials confdiff isnt present (i.e. first trial at start of session or block)
        #torem = np.squeeze(np.where(np.isnan(tfr.metadata.prevtrlconfdiff)))

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

        regressors = list()


        alltrials = np.ones(len(tfr), dtype = 'int')
        cues = tfr.metadata.cue.to_numpy()
        error = tfr.metadata.absrdif.to_numpy()
        confwidth = tfr.metadata.confwidth.to_numpy() #confidence width in degrees, higher = less confident
        conf = np.radians(np.multiply(confwidth, -1)) #reverse the sign so higher (less negative) = more confident, then convert to radians so same scale as error
        confdiff = np.radians(tfr.metadata.confdiff.to_numpy()) #error awareness (metacognition) on that trial (or prediction error)
        neutral = np.where(cues == 0, 1, 0)

        underconf = np.less_equal(confdiff,0)
        overconf  = np.greater(confdiff,0)

        #for underconfident trials, find where trials were overconfident and nan them. zscore this nan'd list, then change nans to 0

        currunder = np.where(underconf == True, confdiff, np.nan) #currently, smaller values indicate better error awareness (lower prediction error)
        currover  = np.where(overconf == True, confdiff, np.nan)  #here, larger values indicate lower error awareness (higher prediction error)

        currunder = np.multiply(currunder, -1) #flip the sign of this, so now larger values indicate worse error awareness (i.e. larger prediction error signals)
        #this aligns under and overconfident onto similar scales for comparison, and makes it easier to understand in terms of prediction errors

        currunder = np.divide(np.subtract(currunder, np.nanmean(currunder)), np.nanstd(currunder))
        currover  = np.divide(np.subtract(currover, np.nanmean(currover)),   np.nanstd(currover))

        #now set nans to 0
        currunder = np.where(np.isnan(currunder), 0, currunder)
        currover  = np.where(np.isnan(currover),  0, currover)




        regressors.append( glm.regressors.ParametricRegressor(name = 'trials', values = alltrials, preproc = None, num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 0, name = 'neutral') )
        regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 1, name = 'cued')    )
        regressors.append( glm.regressors.ParametricRegressor(name = 'error',           values = error,     preproc = 'z',  num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'confidence',      values = conf,      preproc = 'z',  num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'confdiff-underconftrials', values = currunder, preproc = None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'confdiff-overconftrials',  values = currover,  preproc = None, num_observations = glmdata.num_observations))


        contrasts = list()
        contrasts.append( glm.design.Contrast([1, 0, 0, 0, 0, 0, 0], 'grand mean') )
        contrasts.append( glm.design.Contrast([0, 1, 0, 0, 0, 0, 0], 'neutral') )
        contrasts.append( glm.design.Contrast([0, 0, 1, 0, 0, 0, 0], 'cued') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 1, 0, 0, 0], 'error') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 0, 0], 'confidence') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 1, 0], 'confdiff underconf') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 1], 'confdiff overconf') )
        contrasts.append( glm.design.Contrast([0,-1, 1, 0, 0, 0, 0], 'cued vs neutral') )


        glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
        if iglm ==0:
            glmdes.plot_summary()

        total_nave = len(tfr)
        neut_nave  = len(tfr['cue==0'])
        cued_nave  = len(tfr['cue==1'])
        underconf_nave = underconf.sum()
        overconf_nave  = overconf.sum()
        times = tfr.times
        freqs = tfr.freqs
        info = tfr.info

        del(tfr)

        print('\nrunning glm\n')
        model = glm.fit.OLSModel( glmdes, glmdata)

        del(glmdata) #clear from RAM as not used from now on really

        for iname in range(len(glmdes.contrast_names)):
            name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name
            if iname == 1:
                nave = neut_nave
            elif iname == 2:
                nave = cued_nave
            elif iname == 5:
                nave = underconf_nave
            elif iname == 6:
                nave = overconf_nave
            else:
                nave = total_nave


            tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.copes[iname,:,:,:]))
            tfr_betas.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_betas)

            tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.get_tstats()[iname,:,:,:]))
            tfr_tstats.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_tstats)

            tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
            tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_varcopes)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)
