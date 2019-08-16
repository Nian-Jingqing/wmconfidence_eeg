#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:48:53 2019

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


subs = np.array([1, 2, 4, 5, 6, 7, 8, 9, 10])
#subs = np.array([1,2,4,5,6])
#subs = np.array([7,8,9,10])
#subs = np.array([8, 9, 10]) #memory error on subject 8 so need to re run
#%% only needs running if cuelocked TFR glms not already present
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    #get epoched data
    epoch = mne.read_epochs(fname = param['fblocked'], preload = True)
    epoch.set_channel_types({'RM':'misc'}) #set to misc channel
    epoch.set_eeg_reference(ref_channels = ['RM']) #re-reference to right mastoid (now average of left and right mastoids)
    
    #trial rejection hasn't been done yet, so do it here!
    #will do an automated process of looking for trials with heightened variance (noise) and output which trials to keep
    _, keeps = plot_AR(epoch, method = 'gesd', zthreshold = 1.5, p_out=.1, alpha = .05, outlier_side = 1)
    keeps = keeps.flatten()

    discards = np.ones(len(epoch), dtype = 'bool')
    discards[keeps] = False
    epoch = epoch.drop(discards) #first we'll drop trials with excessive noise in the EEG
    
    #now we'll drop trials with behaviour problems (reaction time +/- 2.5 SDs of mean, didn't click to report orientation)
    epoch = epoch['DTcheck == 0 and clickresp == 1']
    epoch.resample(500) #resample
    
    glmdata = glm.data.TrialGLMData(data = epoch.get_data(),
                                    time_dim=2,
                                    sample_rate=500)
    
    
    
    cues = epoch.metadata.cue.to_numpy()
    
    regressors = list()
    regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 0, name = 'neutral') )
    regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 1, name = 'cued')    )
    
    cues = np.where(cues==0, -1, cues) #set neutral trials to -1, cued trials to 1 for interaction terms
    absrdif = epoch.metadata.absrdif.to_numpy()
    absrdifint = np.multiply(absrdif, cues)
    
    confwidth = np.radians(epoch.metadata.confwidth.to_numpy())
    confwidthint = np.multiply(confwidth, cues)

    regressors.append( glm.regressors.ParametricRegressor(name = 'absrdif'      , values = absrdif     , preproc = 'z', num_observations = glmdata.num_observations) )
    regressors.append( glm.regressors.ParametricRegressor(name = 'confwidth'    , values = confwidth   , preproc = 'z', num_observations = glmdata.num_observations) )
    regressors.append( glm.regressors.ParametricRegressor(name = 'absrdifxcue'  , values = absrdifint  , preproc = 'z', num_observations = glmdata.num_observations) )
    regressors.append( glm.regressors.ParametricRegressor(name = 'confwidthxcue', values = confwidthint, preproc = 'z', num_observations = glmdata.num_observations) )


    contrasts = list()
    contrasts.append( glm.design.Contrast([ 1, 0, 0, 0, 0, 0], 'neutral')        )
    contrasts.append( glm.design.Contrast([ 0, 1, 0, 0, 0, 0], 'cued')           )
    contrasts.append( glm.design.Contrast([ 0, 0, 1, 0, 0, 0], 'absrdif')        )
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 1, 0, 0], 'confwidth')      )
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 0, 1, 0], 'absrdifxcue')    )
    contrasts.append( glm.design.Contrast([ 0, 0, 0, 0, 0, 1], 'confwidthxcue')  )
    contrasts.append( glm.design.Contrast([-1, 1, 0, 0, 0, 0], 'cued vs neutral'))
    
    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
    glmdes.plot_summary()
    
    model = glm.fit.OLSModel( glmdes, glmdata )
    
    #neutral trials
    
    tl_betas_neutral = mne.EvokedArray(data = np.squeeze(model.copes[0,:,:]),
                                       info = epoch.info, tmin = epoch.tmin, nave = epoch['cue==0'].average().nave)
    #tl_betas_neutral.plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0))
    tl_betas_neutral.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_neutral_betas-ave.fif'))
    del(tl_betas_neutral)
    
    tl_tstats_neutral = mne.EvokedArray(data = np.squeeze(model.get_tstats()[0, :, :]),
                                        info = epoch.info, tmin = epoch.tmin, nave = epoch['cue==0'].average().nave)
    tl_tstats_neutral.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_neutral_tstats-ave.fif'))
    del(tl_tstats_neutral)

    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    #cued trials
    tl_betas_cued = mne.EvokedArray(data = np.squeeze(model.copes[1,:,:]),
                                       info = epoch.info, tmin = epoch.tmin, nave = epoch['cue==1'].average().nave)
    #tl_betas_cued.plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0))
    tl_betas_cued.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_cued_betas-ave.fif'))
    del(tl_betas_cued)
    
    tl_tstats_cued = mne.EvokedArray(data = np.squeeze(model.get_tstats()[1, :, :]),
                                        info = epoch.info, tmin = epoch.tmin, nave = epoch['cue==1'].average().nave)
    tl_tstats_cued.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_cued_tstats-ave.fif'))
    del(tl_tstats_cued)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    #main effect of response accuracy
    tl_betas_absrdif = mne.EvokedArray(data = np.squeeze(model.copes[2,:,:]),
                                       info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_betas_absrdif.plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0))
    tl_betas_absrdif.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_absrdif_betas-ave.fif'))
    del(tl_betas_absrdif)
    
    tl_tstats_absrdif = mne.EvokedArray(data = np.squeeze(model.get_tstats()[2, :, :]),
                                        info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    tl_tstats_absrdif.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_absrdif_tstats-ave.fif'))
    del(tl_tstats_absrdif)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    #main effect of confidence width
    tl_betas_confwidth = mne.EvokedArray(data = np.squeeze(model.copes[3,:,:]),
                                       info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_betas_confwidth.apply_baseline((-0.2,0)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0))
    tl_betas_confwidth.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_confwidth_betas-ave.fif'))
    del(tl_betas_confwidth)
    
    tl_tstats_confwidth = mne.EvokedArray(data = np.squeeze(model.get_tstats()[3, :, :]),
                                        info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    tl_tstats_confwidth.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_confwidth_tstats-ave.fif'))
    del(tl_tstats_confwidth)

    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    #interaction between cue and accuracy
    tl_betas_accxcue = mne.EvokedArray(data = np.squeeze(model.copes[4,:,:]),
                                       info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_betas_accxcue.apply_baseline((-0.2,0)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0))
    tl_betas_accxcue.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_accxcue_betas-ave.fif'))
    del(tl_betas_accxcue)
    
    tl_tstats_accxcue = mne.EvokedArray(data = np.squeeze(model.get_tstats()[4, :, :]),
                                        info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    tl_tstats_accxcue.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_accxcue_tstats-ave.fif'))
    del(tl_tstats_accxcue)

    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    #interaction between cue and confidence
    tl_betas_confxcue = mne.EvokedArray(data = np.squeeze(model.copes[5,:,:]),
                                       info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_betas_confxcue.apply_baseline((-0.2,0)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0))
    tl_betas_confxcue.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_confxcue_betas-ave.fif'))
    del(tl_betas_confxcue)
    
    tl_tstats_confxcue = mne.EvokedArray(data = np.squeeze(model.get_tstats()[5, :, :]),
                                        info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    tl_tstats_confxcue.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_confxcue_tstats-ave.fif'))
    del(tl_tstats_confxcue)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    #cued vs neutral
    tl_betas_cvsn = mne.EvokedArray(data = np.squeeze(model.copes[6,:,:]),
                                       info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_betas_cvsn.apply_baseline((-0.1,0)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0))
    tl_betas_cvsn.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_cvsn_betas-ave.fif'))
    del(tl_betas_cvsn)
    
    tl_tstats_cvsn = mne.EvokedArray(data = np.squeeze(model.get_tstats()[6, :, :]),
                                        info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    tl_tstats_cvsn.save(fname=op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_cvsn_tstats-ave.fif'))
    del(tl_tstats_cvsn)
    
    
    
    del(epoch)
    del(glmdata)
    del(glmdes)
    del(model)
    
#%%   if the betas are already saved ...
subs = np.array([1,2,4,5,6,7,8,9,10])
alldata_neutral         = []; alldata_neutral_t         = []
alldata_cued            = []; alldata_cued_t            = []
alldata_absrdif         = []; alldata_absrdif_t         = []
alldata_confwidth       = []; alldata_confwidth_t       = []
alldata_absrdif_int     = []; alldata_absrdif_int_t     = []
alldata_confwidth_int   = []; alldata_confwidth_int_t   = []
alldata_cvsn            = []; alldata_cvsn_t            = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)

    neutral = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_neutral_betas-ave.fif')); neutral = neutral[0]
    alldata_neutral.append(neutral)
    neutral_t = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_neutral_tstats-ave.fif')); neutral_t = neutral_t[0]
    alldata_neutral_t.append(neutral_t)
    
    cued = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_cued_betas-ave.fif')); cued = cued[0]
    alldata_cued.append(cued)
    cued_t = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_cued_tstats-ave.fif')); cued_t = cued_t[0]
    alldata_cued_t.append(cued_t)
    
    
    
    absrdif = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_absrdif_betas-ave.fif')); absrdif = absrdif[0]
    alldata_absrdif.append(absrdif)
    absrdif_t = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_absrdif_tstats-ave.fif')); absrdif_t = absrdif_t[0]
    alldata_absrdif_t.append(absrdif_t)
    
    confwidth = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_confwidth_betas-ave.fif')); confwidth = confwidth[0]
    alldata_confwidth.append(confwidth)
    confwidth_t = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_confwidth_tstats-ave.fif')); confwidth_t = confwidth_t[0]
    alldata_confwidth_t.append(confwidth_t)
    
    absrdif_int = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_accxcue_betas-ave.fif')); absrdif_int = absrdif_int[0]
    alldata_absrdif_int.append(absrdif_int)
    absrdif_int_t = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_accxcue_tstats-ave.fif')); absrdif_int_t = absrdif_int_t[0]
    alldata_absrdif_int_t.append(absrdif_int_t)
    
    confwidth_int = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_confxcue_betas-ave.fif')); confwidth_int = confwidth_int[0]
    alldata_confwidth_int.append(confwidth_int)
    confwidth_int_t = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_confxcue_tstats-ave.fif')); confwidth_int_t = confwidth_int_t[0]
    alldata_confwidth_int_t.append(confwidth_int_t)
    
    
    cvsn = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_cvsn_betas-ave.fif')); cvsn = cvsn[0]
    alldata_cvsn.append(cvsn)
    cvsn_t = mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tl_cvsn_tstats-ave.fif')); cvsn_t = cvsn_t[0]
    alldata_cvsn_t.append(cvsn_t)
    

gave_neutral        = mne.grand_average(alldata_neutral)       ; gave_neutral.drop_channels(['RM'])
gave_cued           = mne.grand_average(alldata_cued)          ; gave_cued.drop_channels(['RM'])
gave_absrdif        = mne.grand_average(alldata_absrdif)       ; gave_absrdif.drop_channels(['RM'])
gave_confwidth      = mne.grand_average(alldata_confwidth)     ; gave_confwidth.drop_channels(['RM'])
gave_absrdifint     = mne.grand_average(alldata_absrdif_int)   ; gave_absrdifint.drop_channels(['RM'])
gave_confwidthint   = mne.grand_average(alldata_confwidth_int) ; gave_confwidthint.drop_channels(['RM'])
gave_cvsn           = mne.grand_average(alldata_cvsn)          ; gave_cvsn.drop_channels(['RM'])


baseline = (-0.1, 0)
vispicks = ['O1', 'O2', 'OZ', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ', 'PZ']
cppicks = ['CZ', 'CPZ', 'PZ', 'CP1', 'CP2']
frontpicks = ['AFZ', 'FPZ', 'AF3', 'AF4', 'FZ']
times = [0.5, 0.75, 1.0, 1.25]

gave_neutral.apply_baseline(baseline=baseline).plot_joint(
        topomap_args = dict(outlines='head', contours=0), #ts_args=dict(picks = cppicks, gfp=False, hline=[0]),
        title = 'grand average feedback response neutral trials - betas',
        times = times)
    
gave_cued.apply_baseline(baseline=baseline).plot_joint(
        topomap_args = dict(outlines='head', contours=0), #ts_args=dict(picks = cppicks, gfp=False, hline=[0]),
        title = 'grand average feedback response cued trials - betas',
        times = times)

gave_cvsn.apply_baseline(baseline=baseline).plot_joint(
        topomap_args = dict(outlines='head', contours=0), ts_args=dict(picks = cppicks, gfp=False, hline=[0]),
        title = 'grand average feedback response cued vs neutral trials - betas',
        times = times)

gave_absrdif.apply_baseline(baseline=baseline).plot_joint(
        topomap_args = dict(outlines='head', contours=0), #ts_args=dict(picks = vispicks, gfp=False, hline=[0]),
        title = 'feedback period, main effect of error - betas',
        times = [.1, .2, .3])#times)

gave_confwidth.apply_baseline(baseline=baseline).plot_joint(
        topomap_args = dict(outlines='head', contours=0), #ts_args=dict(picks = frontpicks, gfp=False, hline=[0]),
        title = 'feedback period, main effect of confidence - betas',
        times = [.1, .2, .3])#times)


tmp = mne.combine_evoked([gave_confwidth, -gave_absrdif], weights = 'equal').apply_baseline(baseline = baseline)
tmp.plot_joint(
        topomap_args = dict(outlines='head', contours=0), #ts_args=dict(picks = cppicks, gfp=False, hline=[0]),
        title = 'feedback period, confidence width-error  -- betas',
        times = [.1, .2, .3])#times)

gave_absrdifint.apply_baseline(baseline=baseline).plot_joint(
        topomap_args = dict(outlines='head', contours=0), #ts_args=dict(picks = cppicks, gfp=False, hline=[0]),
        title = 'feedback period, interaction between error and attention (cue) - betas',
        times = [.1, .2, .3])#times)

gave_confwidthint.apply_baseline(baseline=baseline).plot_joint(
        topomap_args = dict(outlines='head', contours=0), #ts_args=dict(picks = cppicks, gfp=False, hline=[0]),
        title = 'feedback period, interaction between confidence and attention (cue) - betas',
        times = [.1, .2, .3])#times)
