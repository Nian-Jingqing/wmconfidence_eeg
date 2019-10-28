#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:41:25 2019

@author: sammirc
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 09:15:49 2019

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


subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18])

#%% only needs running if cuelocked TFR glms not already present
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)

    #get the epoched data
    epochs = mne.read_epochs(fname = param['fblocked'], preload = True) #this is loaded in with the metadata
    epochs.set_eeg_reference(['RM'])
    epochs.apply_baseline((-.25, 0)) #baseline 250ms prior to feedback
    epochs.resample(500) #resample to 500Hz
    ntrials = len(epochs)
    
    #will do an automated process of looking for trials with heightened variance (noise) and output which trials to keep
    _, keeps = plot_AR(epochs, method = 'gesd', zthreshold = 1.5, p_out=.1, alpha = .05, outlier_side = 1)
    plt.close()
    keeps = keeps.flatten()

    discards = np.ones(len(epochs), dtype = 'bool')
    discards[keeps] = False
    epochs = epochs.drop(discards) #first we'll drop trials with excessive noise in the EEG
    
    epochs = epochs['DTcheck == 0 and clickresp == 1']
    print('a total of %d trials have been dropped for this subjects'%(ntrials-len(epochs)))

    glmdata         = glm.data.TrialGLMData(data = epochs.get_data(), time_dim = 2, sample_rate = 500)
    nobs = glmdata.num_observations
    regressors = list()


    alltrials = np.ones(len(epochs), dtype = 'int')
    cues  = epochs.metadata.cue.to_numpy()
    pside = epochs.metadata.pside.to_numpy()
    pside = np.where(pside == 0, 1, -1)

    error = epochs.metadata.absrdif.to_numpy()
    confwidth = epochs.metadata.confwidth.to_numpy() #confidence width in degrees, higher = less confident
    conf = np.radians(np.multiply(confwidth, -1)) #reverse the sign so higher (less negative) = more confident, then convert to radians so same scale as error
    confdiff = epochs.metadata.confdiff.to_numpy() #error awareness (metacognition) on that trial (or prediction error)
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
    #regressors.append( glm.regressors.ParametricRegressor(name = 'pside',          values = pside,           preproc = None, num_observations = nobs))

    contrasts = list()
    
    contrasts.append( glm.design.Contrast([1, 0, 0, 0, 0, 0, 0, 0], 'grand mean'))
    contrasts.append( glm.design.Contrast([0, 1, 0, 0, 0, 0, 0, 0], 'neutral'))
    contrasts.append( glm.design.Contrast([0, 0, 1, 0, 0, 0, 0, 0], 'cued'))
    contrasts.append( glm.design.Contrast([0, 0, 0, 1, 0, 0, 0, 0], 'incorr vs corr'))
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 0, 0, 0], 'error_corr'))
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 1, 0, 0], 'error_incorr'))
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 1, 0], 'conferr_corr'))
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 0, 1], 'conferr_incorr'))
    contrasts.append( glm.design.Contrast([0,-1, 1, 0, 0, 0, 0, 0], 'cued vs neutral'))
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 1, 0, 0], 'error'))
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 1, 1], 'conferror'))
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 0, 1, 0], 'confidence_corr'))
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 1, 0,-1], 'confidence_incorr'))
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 1, 1,-1], 'confidence'))
    contrasts.append( glm.design.Contrast([0, 0, 0, 0,-1, 1, 0, 0], 'error_incorrvscorr'))
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0,-1, 1], 'conferror_incorrvscorr'))
    contrasts.append( glm.design.Contrast([0, 0, 0, 0,-1, 1,-1,-1], 'confidence_incorrvscorr'))
  # contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 0, 0, 1], 'pside'))

    
    

    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
    #glmdes.plot_summary()

    total_nave = len(epochs)
    neut_nave  = len(epochs['cue==0'])
    cued_nave  = len(epochs['cue==1'])
    underconf_nave = targinconf.sum()
    overconf_nave  = targoutsideconf.sum()
    tmin = epochs.tmin
    info = epochs.info

    del(epochs)

    print('\nrunning glm\n')
    model = glm.fit.OLSModel( glmdes, glmdata)

    del(glmdata) #clear from RAM as not used from now on really

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


        tl_betas = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
                                                  data = np.squeeze(model.copes[iname,:,:]))
        #tl_betas.drop_channels(['RM']).plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0))
        tl_betas.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm4', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_betas-ave.fif'))
        del(tl_betas)

        tl_tstats = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
                                                  data = np.squeeze(model.get_tstats()[iname,:,:]))
        tl_tstats.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm4', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_tstats-ave.fif'))
        del(tl_tstats)
#
#        tl_varcopes = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
#                                                  data = np.squeeze(model.varcopes[iname,:,:]))
#        tl_varcopes.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm4', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_varcopes-ave.fif'))
#        del(tl_varcopes)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    del(glmdes)
    del(model)
