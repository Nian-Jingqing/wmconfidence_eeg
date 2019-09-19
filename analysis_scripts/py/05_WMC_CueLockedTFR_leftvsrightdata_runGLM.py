#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:44:09 2019

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
from wmConfidence_funcs import gesd, plot_AR

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
#subs = np.array([11,12,13,14])
#subs = np.array([11, 15])
subs = np.array([16])

glms2run = 1 #1 with no baseline, one where tfr input data is baselined
for i in subs:
    for iglm in range(glms2run):
        print('\n\nrunning glm %d/%d'%(iglm+1, glms2run))
        print('-- working on subject ' + str(i) +' --\n\n')
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub)

        #get tfr
        tfr = mne.time_frequency.read_tfrs(fname=param['cuelocked_tfr']); tfr=tfr[0]
        tfr.metadata = pd.read_csv(param['cuelocked_tfr_meta'], index_col=None) #read in and attach metadata

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
        tfr.data = np.subtract(tfr.data, flipped.data) #now this is left minus right
                   
           
           

        glmdata         = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 100)


        cues      = tfr.metadata.cue.to_numpy() #cue condition for trials
        absrdif   = tfr.metadata.absrdif.to_numpy() #response error on trial (lower is better)
        confwidth = tfr.metadata.confwidth.to_numpy() #confidence width in degrees, higher = less confident
        conf = np.radians(np.multiply(confwidth, -1)) #reverse the sign so higher (less negative) = more confident, then convert to radians so same scale as error
        cuedside  = np.where(tfr.metadata.cuetrig==14, -1, cues) #cued left trials = 1, cued right = -1 (we implicitly code lateralisation in this regressor by flipping signs)
        DT        = tfr.metadata.DT.to_numpy() #decision time (time until pressing space to start response phase) on each trial

        regressors = list()
        regressors.append( glm.regressors.ParametricRegressor(name = 'trials', values = np.ones(glmdata.num_observations), preproc=None, num_observations = glmdata.num_observations))
        #regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 0, name = 'neutral') )
        #regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 1, name = 'cued') )
        regressors.append( glm.regressors.CategoricalRegressor(category_list = tfr.metadata.cuetrig.to_numpy(), codes = 14, name = 'cued right') )
        regressors.append( glm.regressors.CategoricalRegressor(category_list = tfr.metadata.cuetrig.to_numpy(), codes = 13, name = 'cued left') )
        #regressors.append( glm.regressors.ParametricRegressor(name = 'cuedside',          values = cuedside,                         preproc = None, num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'error',             values = absrdif,                          preproc = 'z',  num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'confidence',        values = conf,                        preproc = 'z',  num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'DT',                values = DT,                               preproc = 'z',  num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'error x side',      values = np.multiply(cuedside, absrdif),   preproc = 'z',  num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'confidence x side', values = np.multiply(cuedside, conf), preproc = 'z',  num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'DT x side',         values = np.multiply(cuedside, DT),        preproc = 'z',  num_observations = glmdata.num_observations) )

        contrasts = list()
        contrasts.append( glm.design.Contrast([1, 0, 0, 0, 0, 0, 0, 0, 0], 'grand mean') )
        contrasts.append( glm.design.Contrast([0, 1, 0, 0, 0, 0, 0, 0, 0], 'cued right') )
        contrasts.append( glm.design.Contrast([0, 0, 1, 0, 0, 0, 0, 0, 0], 'cued left') )
        contrasts.append( glm.design.Contrast([0,-1, 1, 0, 0, 0, 0, 0, 0], 'cued l vs r') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 1, 0, 0, 0, 0, 0], 'error') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 0, 0, 0, 0], 'confidence') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 1, 0, 0, 0], 'DT') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 1, 0, 0], 'error x side') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 0, 1, 0], 'confidence x side') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 0, 0, 1], 'DT x side') )


        glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
        #if iglm == 0:
        #    glmdes.plot_summary()

        total_nave = len(tfr)
        cued_nave  = len(tfr['cue==1'])
        neut_nave  = len(tfr['cue==0'])
        cleft_nave = len(tfr['cuetrig==13'])
        cright_nave = len(tfr['cuetrig==14'])
        times = tfr.times
        freqs = tfr.freqs
        info = tfr.info

        del(tfr)


        print('\n - - - - -  running glm - - - - - \n')
        model = glm.fit.OLSModel( glmdes, glmdata)

        del(glmdata) #clear from RAM as not used from now on really

        for iname in range(len(glmdes.contrast_names)):
            name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name
            
            if iname in [3, 7,8,9]:
                nave = cued_nave
            elif iname == 1:
                nave = cright_nave
            elif iname == 2:
                nave = cleft_nave
            else:
                nave = total_nave


            tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.copes[iname,:,:,:]))
            #tfr_betas.plot_joint(timefreqs = {(.4, 10):(.4, 4),(.6, 10):(.4, 4),(.8, 10):(.4, 4),(1., 10):(.4, 4),
            #                                  (.4, 22):(.4,14),(.6, 22):(.4,14),(.8, 22):(.4,14),(1., 22):(.4,14),},
            #                                  topomap_args = dict(outlines='head', contours=0))
            tfr_betas.save(fname = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_'+ name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_betas)

            tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.get_tstats()[iname,:,:,:]))
            tfr_tstats.save(fname = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_'+ name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_tstats)

            tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
            tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'cue_period', 'tfr_glm2', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_cvsi_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_varcopes)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)