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


#%% only needs running if cuelocked TFR glms not already present
glmstorun = 2
for i in subs:
    for iglm in range(glmstorun):
        print('\n\nrunning glm %d/%d'%(iglm+1, glmstorun))
        print('\n\nworking on subject ' + str(i) +'\n\n')
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub)

        #get tfr
        tfr = mne.time_frequency.read_tfrs(fname=param['arraylocked_tfr'])[0]
        tfr.metadata    = pd.read_csv(param['arraylocked_tfr_meta'], index_col=None) #read in and attach metadata
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
           #tfr = tfr.apply_baseline((-0.5, -0.3))
           tfr = tfr.apply_baseline((None,None)) #demean over the entire epoch as pre-array isn't great as a baseline when looking at prestimulus effects

        glmdata         = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 100)

        alltrials = np.ones(len(tfr), dtype = 'int')
        error = tfr.metadata.absrdif.to_numpy()
        confwidth = tfr.metadata.confwidth.to_numpy() #confidence width in degrees, higher = less confident
        conf = np.radians(np.multiply(confwidth, -1)) #reverse the sign so higher (less negative) = more confident, then convert to radians so same scale as error
        DT = tfr.metadata.DT.to_numpy()
        prevtrlconfdiff = np.radians(tfr.metadata.prevtrlconfdiff.to_numpy())
        underconf = np.less_equal(prevtrlconfdiff,0)
        overconf  = np.greater(prevtrlconfdiff, 0)

        #for underconfident trials, find where trials were overconfident and nan them. zscore this nan'd list, then change nans to 0
        prevtrlunderconf = np.where(underconf==True, prevtrlconfdiff, np.nan)
        prevtrlunderconf = np.divide(np.subtract(prevtrlunderconf, np.nanmean(prevtrlunderconf)), np.nanstd(prevtrlunderconf))
        #now set nans to 0
        prevtrlunderconf = np.where(np.isnan(prevtrlunderconf), 0, prevtrlunderconf)

        prevtrloverconf = np.where(overconf == True, prevtrlconfdiff, np.nan)
        prevtrloverconf = np.divide(np.subtract(prevtrloverconf, np.nanmean(prevtrloverconf)), np.nanstd(prevtrloverconf))
        #now set nans to 0 to take out of model
        prevtrloverconf = np.where(np.isnan(prevtrloverconf), 0, prevtrloverconf)




        regressors = list()

        regressors.append( glm.regressors.ParametricRegressor(name = 'trials',                      values = alltrials,        preproc = None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'error',                       values = error,            preproc = 'z',  num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'confidence',                  values = conf,             preproc = 'z',  num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'DT',                          values = DT,               preproc = 'z',  num_observations = glmdata.num_observations))
        #regressors.append( glm.regressors.ParametricRegressor(name = 'prevtrlconfdiff',             values = prevtrlconfdiff,  preproc = 'z',  num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'prevtrlconfdiff - underconf', values = prevtrlunderconf, preproc = None,  num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'prevtrlconfdiff - overconf',  values = prevtrloverconf,  preproc = None,  num_observations = glmdata.num_observations))

        contrasts = list()
        contrasts.append( glm.design.Contrast([1, 0, 0, 0, 0, 0], 'grand mean' ) )
        contrasts.append( glm.design.Contrast([0, 1, 0, 0, 0, 0], 'error') )
        contrasts.append( glm.design.Contrast([0, 0, 1, 0, 0, 0], 'confidence') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 1, 0, 0], 'DT') )
        #contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 0, 0], 'prev trial confdiff'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 0], 'prev trial confdiff underconf'))
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 1], 'prev trial confdiff overconf'))

        glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)

        if iglm == 0 and i >= 11: #just wanna see the new subjects glm summaries for now
            glmdes.plot_summary()

        total_nave = len(tfr)
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
            if iname == 4:
                nave = underconf_nave
            elif iname == 5:
                nave = overconf_nave
            else:
                nave = total_nave


            tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.copes[iname,:,:,:]))
            tfr_betas.save(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_'+ name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_betas)

            tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.get_tstats()[iname,:,:,:]))
            tfr_tstats.save(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_'+ name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_tstats)

            tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
            tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_' + param['subid'] + '_arraylocked_tfr_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_varcopes)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)
