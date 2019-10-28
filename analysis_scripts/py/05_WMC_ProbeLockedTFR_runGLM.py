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
subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
subs = np.array([        4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16])

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
           tfr = tfr.apply_baseline((-2.0, -1.7))

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
        regressors.append( glm.regressors.ParametricRegressor(name = 'pside-neutral',      values = neutrallat,      preproc = None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'pside-cued',         values = cuedlat,         preproc = None, num_observations = glmdata.num_observations))


        DT          = tfr.metadata.DT.to_numpy()
        error       = tfr.metadata.absrdif.to_numpy()
        confwidth   = np.radians(tfr.metadata.confwidth.to_numpy()) #make to radians so on same scale as error
        confwidth   = np.multiply(confwidth, -1) #reverse the sign of this so larger numbers (less negative) are higher confidence
        regressors.append( glm.regressors.ParametricRegressor(name = 'DT',                 values = DT,              preproc = 'z',  num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'error',              values = error,           preproc = 'z',  num_observations = glmdata.num_observations))
        #regressors.append( glm.regressors.ParametricRegressor(name = 'pside',              values = pside,           preproc = None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.ParametricRegressor(name = 'confidence',         values = confwidth,       preproc = 'z',  num_observations = glmdata.num_observations))

#        err_neut = np.where(cues==0, error, np.nan)
#        err_cued = np.where(cues==1, error, np.nan)
#        #zscore ignoring the nans
#        err_neut = np.divide(np.subtract(err_neut, np.nanmean(err_neut)), np.nanstd(err_neut))
#        err_cued = np.divide(np.subtract(err_cued, np.nanmean(err_cued)), np.nanstd(err_cued))   
#        regressors.append( glm.regressors.ParametricRegressor(name = 'error-neutral', values = err_neut, preproc = None, num_observations = glmdata.num_observations))
#        regressors.append( glm.regressors.ParametricRegressor(name = 'error-cued',    values = err_cued, preproc = None, num_observations = glmdata.num_observations))                     
#        
        err_cvsn = np.multiply(error, np.where(cues==0,-1,cues))
        regressors.append(glm.regressors.ParametricRegressor(name = 'error-cvsn', values = err_cvsn, preproc = 'z', num_observations = glmdata.num_observations))
        
        dt_cvsn = np.multiply(DT, np.where(cues==0,-1,cues))
        regressors.append(glm.regressors.ParametricRegressor(name = 'dt-cvsn', values = dt_cvsn, preproc = 'z', num_observations = glmdata.num_observations))
        
        cw_cvsn = np.multiply(confwidth, np.where(cues==0,-1,cues))
        regressors.append(glm.regressors.ParametricRegressor(name = 'cw-cvsn', values = cw_cvsn, preproc = 'z', num_observations = glmdata.num_observations))
        
        
#        dt_neut = np.where(cues==0, DT, np.nan)
#        dt_cued = np.where(cues==1, DT, np.nan)
#        #zscore ignoring the nans
#        dt_neut = np.divide(np.subtract(dt_neut, np.nanmean(dt_neut)), np.nanstd(dt_neut))
#        dt_cued = np.divide(np.subtract(dt_cued, np.nanmean(dt_cued)), np.nanstd(dt_cued))   
#        regressors.append( glm.regressors.ParametricRegressor(name = 'DT-neutral', values = dt_neut, preproc = None, num_observations = glmdata.num_observations))
#        regressors.append( glm.regressors.ParametricRegressor(name = 'DT-cued',    values = dt_cued, preproc = None, num_observations = glmdata.num_observations))            
#        
#        
#        cw_neut = np.where(cues==0, confwidth, np.nan)
#        cw_cued = np.where(cues==1, confwidth, np.nan)
#        #zscore ignoring the nans
#        cw_neut = np.divide(np.subtract(cw_neut, np.nanmean(cw_neut)), np.nanstd(cw_neut))
#        cw_cued = np.divide(np.subtract(cw_cued, np.nanmean(cw_cued)), np.nanstd(cw_cued))   
#        regressors.append( glm.regressors.ParametricRegressor(name = 'conf-neutral', values = cw_neut, preproc = None, num_observations = glmdata.num_observations))
#        regressors.append( glm.regressors.ParametricRegressor(name = 'conf-cued',    values = cw_cued, preproc = None, num_observations = glmdata.num_observations))   




        cued = np.where(cues==0,-1,cues)
        DTxpside    = np.multiply(DT, pside)
        errorxpside = np.multiply(tfr.metadata.absrdif.to_numpy(), pside)
        confwidthxpside = np.multiply(confwidth, pside)
        confneut    = np.multiply(confwidth, neuttrls)
        confcued    = np.multiply(confwidth, cuedtrls)


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
        contrasts.append(glm.design.Contrast([ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'grand mean')   )
        contrasts.append(glm.design.Contrast([ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'neutral')   )
        contrasts.append(glm.design.Contrast([ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'cued')   )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pside neutral')   )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pside cued')   )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'DT')   )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'error')   )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'confidence')   )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'error_cvsn') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'DT_cvsn') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'conf_cvsn') )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'errorxpside_neutral')   )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'errorxpside_cued')   )        
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'DTxpside_neutral')   )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'DTxpside_cued')   )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'confxpside_neutral')   )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'confxpside_cued')   )
        contrasts.append(glm.design.Contrast([ 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pside x cvsn'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0], 'error x pside x cvsn'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0], ' DT x pside x cvsn'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1], ' conf x pside x cvsn'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pside') )
        
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
            if iname in [1, 3, 8, 11, 13, 15]:
                nave = neut_nave
            elif iname in [2, 4, 9, 12, 14, 16]:
                nave = cued_nave
            else:
                nave = total_nave


            tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.copes[iname,:,:,:]))
            tfr_betas.save(fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_'+ name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_betas)

            tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.get_tstats()[iname,:,:,:]))
            tfr_tstats.save(fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_'+ name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_tstats)

            tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
            tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm', 'wmConfidence_' + param['subid'] + '_probelocked_tfr_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_varcopes)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)
