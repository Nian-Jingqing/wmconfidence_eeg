#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 10:40:57 2019

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
from wmConfidence_funcs import gesd, plot_AR, nanzscore

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/glm')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])


#%% only needs running if cuelocked TFR glms not already present
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)

    #get the epoched data
    epochs = mne.read_epochs(fname = param['fblocked'], preload = True) #this is loaded in with the metadata
    #based on photodiode testing, there is a 25ms delay from trigger onset to maximal photodiode onset, so lets adjust times here
    epochs.shift_time(tshift = -0.025, relative = True)
    
    epochs.apply_baseline((-.25, 0)) #baseline 250ms prior to feedback
    epochs.resample(500) #resample to 500Hz
    if 'VEOG' in epochs.ch_names:
        epochs = epochs.drop_channels(['VEOG', 'HEOG'])
    ntrials = len(epochs)
    

    glmdata         = glm.data.TrialGLMData(data = epochs.get_data(), time_dim = 2, sample_rate = 500)
    nobs = glmdata.num_observations
    trials = np.ones(nobs)

    cues = epochs.metadata.cue.to_numpy()
    error = epochs.metadata.absrdif.to_numpy()
    confwidth = epochs.metadata.confwidth.to_numpy() #confidence width in degrees, higher = less confident
    conf = np.radians(np.multiply(confwidth, -1)) #reverse the sign so higher (less negative) = more confident, then convert to radians so same scale as error
    confdiff = epochs.metadata.confdiff.to_numpy() #error awareness (metacognition) on that trial (or prediction error)
    neutral = np.where(cues == 0, 1, 0)
    
    targinconf = np.less_equal(confdiff,0)
    targoutsideconf = np.greater(confdiff,0)
    threshold = -10
    if threshold == -6:
        glmnum = 5
    else:
        glmnum = 6
    defscorr = np.less_equal(confdiff, threshold) #easily correct (over 5 degrees underconfident)
    justcorr = np.isin(confdiff, np.arange(threshold+1, 1)) #within 5 degrees of the boundary (less than 5 degrees underconfident)
    
    
    err_defscorr = nanzscore(np.where(defscorr == 1, error, np.nan))
    err_justcorr = nanzscore(np.where(justcorr == 1, error, np.nan))
    err_incorr   = nanzscore(np.where(targoutsideconf == 1, error, np.nan))
    
    conf_defscorr = nanzscore(np.where(defscorr == 1, conf, np.nan))
    conf_justcorr = nanzscore(np.where(justcorr == 1, conf, np.nan))
    conf_incorr   = nanzscore(np.where(targoutsideconf == 1, conf, np.nan))
    
    
    pside = epochs.metadata.pside.to_numpy()
    pside = np.where(pside == 0, 1, -1)
    
    #add regressors to the model
    
    regressors = list()
    regressors.append(glm.regressors.CategoricalRegressor(category_list = defscorr, codes = 1, name = 'defcorrect'))
    regressors.append(glm.regressors.CategoricalRegressor(category_list = justcorr, codes = 1, name = 'justcorrect'))
    regressors.append(glm.regressors.CategoricalRegressor(category_list = targoutsideconf, codes = 1, name = 'incorrect'))
    
    regressors.append(glm.regressors.ParametricRegressor(name = 'error defs correct', values = err_defscorr, preproc = None, num_observations = nobs))
    regressors.append(glm.regressors.ParametricRegressor(name = 'error just correct', values = err_justcorr, preproc = None, num_observations = nobs))
    regressors.append(glm.regressors.ParametricRegressor(name = 'error incorrect',    values = err_incorr,   preproc = None, num_observations = nobs))
    
    regressors.append(glm.regressors.ParametricRegressor(name = 'conf defs correct', values = conf_defscorr, preproc = None, num_observations = nobs))
    regressors.append(glm.regressors.ParametricRegressor(name = 'conf just correct', values = conf_justcorr, preproc = None, num_observations = nobs))
    regressors.append(glm.regressors.ParametricRegressor(name = 'conf incorrect',    values = conf_incorr,   preproc = None, num_observations = nobs))
    regressors.append(glm.regressors.ParametricRegressor(name = 'pside',             values = pside,         preproc = None, num_observations = nobs))
    
    contrasts = list()
    contrasts.append( glm.design.Contrast([ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'defcorrect' ))
    contrasts.append( glm.design.Contrast([ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'justcorrect' ))
    contrasts.append( glm.design.Contrast([ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'incorrect' ))
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'err defcorrect' ))
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'err justcorrect' ))
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'err incorrect' ))
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'conf defcorrect' ))
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'conf justcorrect' ))
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'conf incorrect' ))
    contrasts.append( glm.design.Contrast([-1, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'incorr vs def' ))
    contrasts.append( glm.design.Contrast([ 0,-1, 1, 0, 0, 0, 0, 0, 0, 0], 'incorr vs just' ))
    contrasts.append( glm.design.Contrast([-1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'just vs def' ))
    contrasts.append( glm.design.Contrast([ 0, 0, 0,-1, 0, 1, 0, 0, 0, 0], 'error incorr vs def' ))
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 0,-1, 1, 0, 0, 0, 0], 'error incorr vs just' ))
    contrasts.append( glm.design.Contrast([ 0, 0, 0,-1, 1, 0, 0, 0, 0, 0], 'error just vs def' ))
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 0, 0, 0,-1, 0, 1, 0], 'conf incorr vs def' ))
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0,-1, 1, 0], 'conf incorr vs just' ))
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 0, 0, 0,-1, 1, 0, 0], 'conf just vs def' ))
    
    #these contrasts are just done to be able to do an F-test across the trial types (and error + confidence regressors)
    
    contrasts.append( glm.design.Contrast([-1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'justvdef' )) 
    contrasts.append( glm.design.Contrast([-1, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'invdef' ))
    
    contrasts.append( glm.design.Contrast([ 0, 0, 0,-1, 1, 0, 0, 0, 0, 0], 'errjvd'))
    contrasts.append( glm.design.Contrast([ 0, 0, 0,-1, 0, 1, 0, 0, 0, 0], 'errivd'))
    
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 0, 0, 0,-1, 1, 0, 0], 'confjvd'))
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 0, 0, 0,-1, 0, 1, 0], 'confivd'))
    
    ftests = list()
    ftests.append(glm.design.FTest([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0], 'ME_trialtype'))
    ftests.append(glm.design.FTest([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0], 'ME_error'))
    ftests.append(glm.design.FTest([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1], 'ME_confidence'))
    
    
    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts, ftests)
#    glmdes.plot_summary()

    total_nave = len(epochs)
    def_nave = defscorr.sum()
    just_nave = justcorr.sum()
    incorr_nave = targoutsideconf.sum()
    incorrvsdef_nave = incorr_nave + def_nave
    incorrvsjust_nave = incorr_nave+just_nave
    justvsdef_nave = just_nave+def_nave
    tmin = epochs.tmin
    info = epochs.info

    del(epochs)

    print('\nrunning glm\n')
    model = glm.fit.OLSModel( glmdes, glmdata)

    del(glmdata) #clear from RAM as not used from now on really
    #contrasts = np.stack([np.arange(len(contrasts)), glmdes.contrast_names], axis=1)
    #ftestz = np.stack([np.arange(len(ftests)), glmdes.ftest_names], axis = 1)
    for iname in range(len(glmdes.contrast_names)-6):
        name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name
        if iname in [0, 3, 6]:
            nave = def_nave
        elif iname in [1, 4, 7]:
            nave = just_nave
        elif iname in [2, 5, 8]:
            nave = incorr_nave
        elif iname == 9:
            nave = incorrvsdef_nave
        elif iname == 10:
            nave = incorrvsjust_nave
        elif iname == 11:
            nave = justvsdef_nave
        else:
            nave = total_nave


        tl_betas = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
                                                  data = np.squeeze(model.copes[iname,:,:]))
#        deepcopy(tl_betas).drop_channels(['RM']).plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0))
        tl_betas.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm'+str(glmnum), 'wmc_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_betas-ave.fif'))
        del(tl_betas)

        tl_tstats = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
                                                  data = np.squeeze(model.get_tstats()[iname,:,:]))
        tl_tstats.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm'+str(glmnum), 'wmc_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_tstats-ave.fif'))
        del(tl_tstats)

#        tl_varcopes = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
#                                                  data = np.squeeze(model.varcopes[iname,:,:]))
#        tl_varcopes.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm5', 'wmc_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_varcopes-ave.fif'))
#        del(tl_varcopes)

    for iname in range(len(glmdes.ftest_names)):
        name = glmdes.ftest_names[iname].replace(' ', '')
        
        tl_fstat = mne.EvokedArray(info = info, nave = total_nave, tmin = tmin,
                                   data = np.squeeze(model.fstats[iname,:,:]))
#        deepcopy(tl_fstat).drop_channels(['RM']).plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0))
        tl_fstat.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm'+str(glmnum), 'wmc_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_fstat-ave.fif'))

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    del(glmdes)
    del(model)
