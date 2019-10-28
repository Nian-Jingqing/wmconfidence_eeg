#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 09:08:17 2019

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


subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

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

    regressors = list()


    alltrials = np.ones(len(epochs), dtype = 'int')
    cues = epochs.metadata.cue.to_numpy()
    error = epochs.metadata.absrdif.to_numpy()
    confwidth = epochs.metadata.confwidth.to_numpy() #confidence width in degrees, higher = less confident
    conf = np.radians(np.multiply(confwidth, -1)) #reverse the sign so higher (less negative) = more confident, then convert to radians so same scale as error
    confdiff = np.radians(epochs.metadata.confdiff.to_numpy()) #error awareness (metacognition) on that trial (or prediction error)
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
    
    # is the target inside the confidence interval? I.e. confdiff < 0 , as green feedback
    targinconf = np.where(epochs.metadata.confdiff.to_numpy() <= 0, 1, 0)
    targoutsideconf = np.where(epochs.metadata.confdiff.to_numpy() > 0, 1, 0)
    gvsred = np.where(targinconf == 0, -1, targinconf) #this is just the contrast of the two but we can set it up as a contrast anyway, just here in case

    #to get grand average evoked response
    regressors.append( glm.regressors.ParametricRegressor(name = 'trials', values = alltrials, preproc = None, num_observations = glmdata.num_observations) )
    
    #to be able to look at feedback processing differences based on whether an item was attended or not
    regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 0, name = 'neutral') )
    regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 1, name = 'cued')    )
    
    #main effects to look at ERP variations due to these cognitive things
    regressors.append( glm.regressors.ParametricRegressor(name = 'error',           values = error,     preproc = 'z',  num_observations = glmdata.num_observations) )
    regressors.append( glm.regressors.ParametricRegressor(name = 'confidence',      values = conf,      preproc = 'z',  num_observations = glmdata.num_observations) )
    
    #to look just at whether or not there are differences based on the feedback colour (i.e. just good vs bad feedback)
    regressors.append( glm.regressors.CategoricalRegressor(category_list = targinconf     , codes = 1, name = 'targinconf') )
    regressors.append( glm.regressors.CategoricalRegressor(category_list = targoutsideconf, codes = 1, name = 'targoutsideconf') )
    
    #and this lets us look at the essentially prediction error signal on the different trial types
    regressors.append( glm.regressors.ParametricRegressor(name = 'confdiff-underconftrials', values = currunder, preproc = None, num_observations = glmdata.num_observations))
    regressors.append( glm.regressors.ParametricRegressor(name = 'confdiff-overconftrials',  values = currover,  preproc = None, num_observations = glmdata.num_observations))


    contrasts = list()
    contrasts.append( glm.design.Contrast([1, 0, 0, 0, 0, 0, 0, 0, 0], 'grand mean') )
    contrasts.append( glm.design.Contrast([0, 1, 0, 0, 0, 0, 0, 0, 0], 'neutral') )
    contrasts.append( glm.design.Contrast([0, 0, 1, 0, 0, 0, 0, 0, 0], 'cued') )
    contrasts.append( glm.design.Contrast([0, 0, 0, 1, 0, 0, 0, 0, 0], 'error') )
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 0, 0, 0, 0], 'confidence') )
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 1, 0, 0, 0], 'targinconf') )
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 1, 0, 0], 'targoutsideconf') )
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 0, 1, 0], 'confdiff underconf') )
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 0, 0, 1], 'confdiff overconf') )
    contrasts.append( glm.design.Contrast([0,-1, 1, 0, 0, 0, 0, 0, 0], 'cued vs neutral') )
    contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0,-1, 1, 0, 0], 'redvsgreen') ) #this is just a good vs bad feedback contrast between whether target was in or outside of reported confidence interval

    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
    glmdes.plot_summary()

    total_nave = len(epochs)
    neut_nave  = len(epochs['cue==0'])
    cued_nave  = len(epochs['cue==1'])
    underconf_nave = underconf.sum()
    overconf_nave  = overconf.sum()
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
        elif iname in [5, 7]:
            nave = underconf_nave
        elif iname in [6, 8]:
            nave = overconf_nave
        else:
            nave = total_nave


        tl_betas = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
                                                  data = np.squeeze(model.copes[iname,:,:]))
        tl_betas.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_betas-ave.fif'))
        del(tl_betas)

        tl_tstats = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
                                                  data = np.squeeze(model.get_tstats()[iname,:,:]))
        tl_tstats.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_tstats-ave.fif'))
        del(tl_tstats)

        tl_varcopes = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
                                                  data = np.squeeze(model.varcopes[iname,:,:]))
        tl_varcopes.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_varcopes-ave.fif'))
        del(tl_varcopes)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    del(glmdes)
    del(model)