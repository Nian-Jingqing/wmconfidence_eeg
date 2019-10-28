#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:34:52 2019

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
from wmConfidence_funcs import gesd, plot_AR

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
subs = np.array([        4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16])


glms2run = 2 #1 with no baseline, one where tfr input data is baselined
for i in subs:
    for iglm in range(glms2run):
        print('\n\nrunning glm %d/%d'%(iglm+1, glms2run))
        print('-- working on subject ' + str(i) +' --\n\n')
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub)

        #get tfr
        tfr = mne.time_frequency.read_tfrs(fname=param['cuelocked_tfr']); tfr=tfr[0]
        tfr.metadata = pd.read_csv(param['cuelocked_tfr_meta'], index_col=None) #read in and attach metadata

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

        cues      = tfr.metadata.cue.to_numpy() #cue condition for trials
        absrdif   = tfr.metadata.absrdif.to_numpy() #response error on trial (lower is better)
        confwidth = tfr.metadata.confwidth.to_numpy() #confidence width in degrees, higher = less confident
        #conf = np.radians(np.multiply(confwidth, -1)) #reverse the sign so higher (less negative) = more confident, then convert to radians so same scale as error
        neuttrl = np.where(cues==0,1,0)
        probedside = tfr.metadata.pside.to_numpy()
        probedside = np.where(probedside == 0, 1, -1) #implicitly code this as left vs right (left = 1, right = -1)
        DT        = tfr.metadata.DT.to_numpy() #decision time (time until pressing space to start response phase) on each trial
        
        #rewrite some of these regressors as they need transforming
        
        dt_bc = sp.stats.boxcox(DT)
        fig = plt.figure()
        sns.distplot(DT, bins=60, hist = True, norm_hist = True, ax = fig.add_subplot(311))
        sns.distplot(np.log(DT), bins = 60, hist = True, norm_hist = True, ax = fig.add_subplot(323), axlabel = 'log DT')
        sns.distplot(dt_bc[0], bins = 60, hist = True, norm_hist = True, ax = fig.add_subplot(325), axlabel = 'boxcox DT')
        
        sns.distplot(sp.stats.zscore(np.log(DT)), bins = 60, hist = True, norm_hist = True, ax = fig.add_subplot(324), axlabel = 'zscore(log(DT))')
        sns.distplot(sp.stats.zscore(dt_bc[0]), bins = 60, hist = True, norm_hist = True, ax = fig.add_subplot(326), axlabel = 'zscore(boxcox(DT))')
        
        error = np.where(absrdif==0, 0.0001, absrdif)
        error_bc = sp.stats.boxcox(error)
        fig = plt.figure()
        sns.distplot(error, bins=60, hist = True, norm_hist = True, ax = fig.add_subplot(311))
        sns.distplot(np.log(error), bins = 60, hist = True, norm_hist = True, ax = fig.add_subplot(323), axlabel = 'log error')
        sns.distplot(error_bc[0], bins = 60, hist = True, norm_hist = True, ax = fig.add_subplot(325), axlabel = 'boxcox error')
        
        sns.distplot(sp.stats.zscore(np.log(error)), bins = 60, hist = True, norm_hist = True, ax = fig.add_subplot(324), axlabel = 'zscore(log(error))')
        sns.distplot(sp.stats.zscore(error_bc[0]), bins = 60, hist = True, norm_hist = True, ax = fig.add_subplot(326), axlabel = 'zscore(boxcox(error))')

        cw = np.where(confwidth == 0, 0.0001, confwidth)
        conf_bc = sp.stats.boxcox(cw)
        fig = plt.figure()
        sns.distplot(cw, bins=60, hist = True, norm_hist = True, ax = fig.add_subplot(311))
        sns.distplot(np.log(cw), bins = 60, hist = True, norm_hist = True, ax = fig.add_subplot(323), axlabel = 'log cw')
        sns.distplot(conf_bc[0], bins = 60, hist = True, norm_hist = True, ax = fig.add_subplot(325), axlabel = 'boxcox cw')
        
        sns.distplot(sp.stats.zscore(np.log(cw)), bins = 60, hist = True, norm_hist = True, ax = fig.add_subplot(324), axlabel = 'zscore(log(cw))')
        sns.distplot(sp.stats.zscore(conf_bc[0]), bins = 60, hist = True, norm_hist = True, ax = fig.add_subplot(326), axlabel = 'zscore(boxcox(cw))')

        

        regressors = list()
        regressors.append( glm.regressors.ParametricRegressor(name = 'trials', values = np.ones(glmdata.num_observations), preproc=None, num_observations = nobs))
        regressors.append( glm.regressors.CategoricalRegressor(name = 'neutral', category_list = cues, codes = 0)) #neutral trials
        regressors.append( glm.regressors.ParametricRegressor(name = 'cued', values = np.multiply(cues, probedside), preproc = None, num_observations = nobs))
        #regressors.append( glm.regressors.CategoricalRegressor(name = 'cued', category_list= cues, codes = 1))    #cued trials
        regressors.append( glm.regressors.ParametricRegressor(name = 'error',      values = error_bc[0],  preproc = 'z',  num_observations = nobs) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'confidence', values = conf_bc[0],     preproc = 'z',  num_observations = nobs) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'DT',         values = np.log(DT),       preproc = 'z',  num_observations = nobs) )
        
        #need behavioural variables correlating with evoked response by cue
        #only lateralised when a cue appeared (in theory)
        #so need so to be: behaviours x pside - cued trials
        
        err_pside = np.where(cues ==1, absrdif, np.nan)
        err_pside = np.multiply(err_pside, probedside) #flip the regressor to code left vs right (probed left vs right in cued trials is cued left vs right)
        err_pside = np.divide(np.subtract(err_pside, np.nanmean(err_pside)), np.nanstd(err_pside))
        err_pside = np.where(np.isnan(err_pside), 0, err_pside)
        
        regressors.append( glm.regressors.ParametricRegressor(name = 'error x pside - cued', values = err_pside, preproc = None, num_observations = nobs) )
        
        dt_pside = np.where(cues == 1, DT, np.nan)
        dt_pside = np.multiply(dt_pside, probedside)
        dt_pside = np.divide(np.subtract(dt_pside, np.nanmean(dt_pside)), np.nanstd(dt_pside))
        dt_pside = np.where(np.isnan(dt_pside), 0, dt_pside)
        
        regressors.append( glm.regressors.ParametricRegressor(name = 'DT x pside - cued', values = dt_pside, preproc = None, num_observations = nobs) )
        
        conf_pside = np.where(cues == 1, confwidth, np.nan)
        conf_pside = np.multiply(conf_pside, probedside)
        conf_pside = np.divide(np.subtract(conf_pside, np.nanmean(conf_pside)), np.nanstd(conf_pside))
        conf_pside = np.where(np.isnan(conf_pside), 0, conf_pside)
        
        regressors.append( glm.regressors.ParametricRegressor(name = 'conf x pside - cued', values = conf_pside, preproc = None, num_observations = nobs) )
        
        
        contrasts = list()
        
        contrasts.append(glm.design.Contrast([1, 0, 0, 0, 0, 0, 0, 0, 0], 'grand mean'))
        contrasts.append(glm.design.Contrast([0, 1, 0, 0, 0, 0, 0, 0, 0], 'neutral'))
        contrasts.append(glm.design.Contrast([0, 0, 1, 0, 0, 0, 0, 0, 0], 'cued'))
        contrasts.append(glm.design.Contrast([0, 0, 0, 1, 0, 0, 0, 0, 0], 'error'))
        contrasts.append(glm.design.Contrast([0, 0, 0, 0, 1, 0, 0, 0, 0], 'confidence'))
        contrasts.append(glm.design.Contrast([0, 0, 0, 0, 0, 1, 0, 0, 0], 'dt'))
        contrasts.append(glm.design.Contrast([0, 0, 0, 0, 0, 0, 1, 0, 0], 'error x cuedlvsr'))
        contrasts.append(glm.design.Contrast([0, 0, 0, 0, 0, 0, 0, 1, 0], 'dt x cuedlvsr'))
        contrasts.append(glm.design.Contrast([0, 0, 0, 0, 0, 0, 0, 0, 1], 'conf x cuedlvsr'))
        contrasts.append(glm.design.Contrast([0,-1, 1, 0, 0, 0, 0, 0, 0], 'cued vs neutral'))
        
        
        

        glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
        #if iglm == 0:
        #    glmdes.plot_summary()

        total_nave = len(tfr)
        cued_nave  = len(tfr['cue==1'])
        neut_nave  = len(tfr['cue==0'])
        times = tfr.times
        freqs = tfr.freqs
        info = tfr.info

        del(tfr)


        print('\n - - - - -  running glm - - - - - \n')
        model = glm.fit.OLSModel( glmdes, glmdata)

        del(glmdata) #clear from RAM as not used from now on really

        for iname in range(len(glmdes.contrast_names)):
            name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name
            
            if iname in [2, 6,7,8]:
                nave = cued_nave
            else:
                nave = total_nave


            tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.copes[iname,:,:,:]))
#            deepcopy(tfr_betas).drop_channels(['RM']).plot_joint(timefreqs = {(.4,10):(.4,.4), (.6,10):(.4,4), (.8,10):(.4,4)},
#                                              topomap_args = dict(outlines='head', contours = 0))
            tfr_betas.save(fname = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_betas)

            tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.get_tstats()[iname,:,:,:]))
#            deepcopy(tfr_tstats).drop_channels(['RM']).plot_joint(timefreqs = {(.4,10):(.4,.4), (.6,10):(.4,4), (.8,10):(.4,4)},
#                                               topomap_args = dict(outlines='head', contours = 0, vmin=-1.8, vmax=1.8))

            tfr_tstats.save(fname = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_tstats)

            tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
            tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm3', 'wmc_' + param['subid'] + '_cuelocked_tfr_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_varcopes)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)