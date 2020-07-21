#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:46:52 2019

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
from scipy import ndimage

#stuff for the machine learning

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import sklearn as skl

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
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
#%% only needs running if cuelocked TFR glms not already present

alldata_left  = []; alldata_left_t  = []
alldata_right = []; alldata_right_t = []
alldata_lvsr  = []; alldata_lvsr_t  = []
laplacian = True

for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    #get epoched data
    epoch = mne.read_epochs(fname = param['resplocked'], preload = True)
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
    epoch = epoch['DTcheck == 0 and clickresp == 1 and arraycueblink == 0']
    epoch.resample(500) #resample
    epoch = epoch.pick('eeg') #subsample to only eeg electrodes
    
    
    if laplacian:
        epoch2 = mne.preprocessing.compute_current_source_density(epoch);
        
    
    glmdata = glm.data.TrialGLMData(data = epoch2.get_data(),
                                    time_dim=2,
                                    sample_rate=500)
    
    #get some behavioural things we're going to look at
    trials = np.ones(glmdata.num_observations) #regressor for just grand mean response
    cues   = epoch.metadata.cue.to_numpy()
    pside = epoch.metadata.pside.to_numpy()
    pside = np.where(pside == 0, 1, -1)
    
    regressors = list()
    regressors.append( glm.regressors.ParametricRegressor(name = 'grand mean', values = trials, preproc = None, num_observations = glmdata.num_observations))
    regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 0, name = 'neutral'))
    regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 1, name = 'cued'))
    regressors.append( glm.regressors.ParametricRegressor(name = 'DT', values = epoch.metadata.DT.to_numpy(), preproc='z', num_observations = glmdata.num_observations))
    regressors.append( glm.regressors.ParametricRegressor(name = 'DTxpside',values = np.multiply(epoch.metadata.DT.to_numpy(),pside), preproc = 'z', num_observations = glmdata.num_observations))
    
    
    contrasts = list()
    contrasts.append(glm.design.Contrast([ 1,  0,  0,  0,  0], 'grand mean')      )
    contrasts.append(glm.design.Contrast([ 0,  1,  0,  0,  0], 'neutral')         )
    contrasts.append(glm.design.Contrast([ 0,  0,  1,  0,  0], 'cued')            )
    contrasts.append(glm.design.Contrast([ 0,  0,  0,  1,  0], 'DT')              )
    contrasts.append(glm.design.Contrast([ 0, -1,  1,  0,  0], 'cued vs neutral') ) 
    contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  1], 'DT x pside'))
    
    
    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
    glmdes.plot_summary()
    
    model = glm.fit.OLSModel( glmdes, glmdata )
    
    #grand mean
    tl_betas_grandmean = mne.EvokedArray(data = np.squeeze(model.copes[0,:,:]),
                                       info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    tl_betas_grandmean.apply_baseline((None, None)).plot_joint(picks='eeg', times= np.arange(0,0.5,0.1),topomap_args=dict(outlines='head', contours=0)) #this baseline demeans the entire epoch
    tl_betas_grandmean.plot_joint(times=np.arange(0,0.5,0.1), topomap_args=dict(outlines='head', contours=0)) #can add picks=['C3', 'C4', 'C5', 'C6'] to look at lateralised motor electrodes
    tl_betas_grandmean.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_grandmean_betas-ave.fif'))
    del(tl_betas_grandmean)
    
    tl_tstats_grandmean = mne.EvokedArray(data = np.squeeze(model.get_tstats()[0, :, :]),
                                        info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_tstats_grandmean.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0))
    tl_tstats_grandmean.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_grandmean_tstats-ave.fif'))
    del(tl_tstats_grandmean)
    
    tl_varcope_grandmean = mne.EvokedArray(data = np.squeeze(model.varcopes[0,:,:]),
                                           info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_varcope_grandmean.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0), picks = ['C3', 'C4'])
    tl_varcope_grandmean.save(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_grandmean_varcope-ave.fif'))
    del(tl_varcope_grandmean)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    #neutral
    tl_betas_neutral = mne.EvokedArray(data = np.squeeze(model.copes[1,:,:]),
                                       info = epoch.info, tmin = epoch.tmin, nave = epoch['cue==0'].average().nave)
    #tl_betas_neutral.apply_baseline((None, None)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0)) #this baseline demeans the entire epoch
    #tl_betas_neutral.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0)) #can add picks=['C3', 'C4', 'C5', 'C6'] to look at lateralised motor electrodes
    tl_betas_neutral.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_neutral_betas-ave.fif'))
    del(tl_betas_neutral)
    
    tl_tstats_neutral = mne.EvokedArray(data = np.squeeze(model.get_tstats()[1, :, :]),
                                        info = epoch.info, tmin = epoch.tmin, nave = epoch['cue==0'].average().nave)
    #tl_tstats_neutral.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0))
    tl_tstats_neutral.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_neutral_tstats-ave.fif'))
    del(tl_tstats_neutral)
    
    tl_varcope_neutral = mne.EvokedArray(data = np.squeeze(model.varcopes[1,:,:]),
                                           info = epoch.info, tmin = epoch.tmin, nave = epoch['cue==0'].average().nave)
    #tl_varcope_neutral.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0), picks = ['C3', 'C4'])
    tl_varcope_neutral.save(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_neutral_varcope-ave.fif'))
    del(tl_varcope_neutral)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    #cued
    tl_betas_cued = mne.EvokedArray(data = np.squeeze(model.copes[2,:,:]),
                                       info = epoch.info, tmin = epoch.tmin, nave = epoch['cue==1'].average().nave)
    tl_betas_cued.apply_baseline((None, None)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0)) #this baseline demeans the entire epoch
    #tl_betas_cued.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0)) #can add picks=['C3', 'C4', 'C5', 'C6'] to look at lateralised motor electrodes
    tl_betas_cued.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_cued_betas-ave.fif'))
    del(tl_betas_cued)
    
    tl_tstats_cued = mne.EvokedArray(data = np.squeeze(model.get_tstats()[2, :, :]),
                                        info = epoch.info, tmin = epoch.tmin, nave = epoch['cue==1'].average().nave)
    #tl_tstats_cued.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0))
    tl_tstats_cued.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_cued_tstats-ave.fif'))
    del(tl_tstats_cued)
    
    tl_varcope_cued = mne.EvokedArray(data = np.squeeze(model.varcopes[2,:,:]),
                                           info = epoch.info, tmin = epoch.tmin, nave = epoch['cue==1'].average().nave)
    #tl_varcope_cued.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0), picks = ['C3', 'C4'])
    tl_varcope_cued.save(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_cued_varcope-ave.fif'))
    del(tl_varcope_cued)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    #DT
    tl_betas_DT = mne.EvokedArray(data = np.squeeze(model.copes[3,:,:]),
                                       info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_betas_DT.apply_baseline((None, None)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0)) #this baseline demeans the entire epoch
    #tl_betas_DT.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0)) #can add picks=['C3', 'C4', 'C5', 'C6'] to look at lateralised motor electrodes
    tl_betas_DT.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_DT_betas-ave.fif'))
    del(tl_betas_DT)
    
    tl_tstats_DT = mne.EvokedArray(data = np.squeeze(model.get_tstats()[3, :, :]),
                                        info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_tstats_DT.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0))
    tl_tstats_DT.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_DT_tstats-ave.fif'))
    del(tl_tstats_DT)
    
    tl_varcope_DT = mne.EvokedArray(data = np.squeeze(model.varcopes[3,:,:]),
                                           info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_varcope_cued.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0), picks = ['C3', 'C4'])
    tl_varcope_DT.save(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_DT_varcope-ave.fif'))
    del(tl_varcope_DT)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    #cued vs neutral
    tl_betas_cvsn = mne.EvokedArray(data = np.squeeze(model.copes[4,:,:]),
                                       info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_betas_cvsn.apply_baseline((None, None)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0)) #this baseline demeans the entire epoch
    #tl_betas_cvsn.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0)) #can add picks=['C3', 'C4', 'C5', 'C6'] to look at lateralised motor electrodes
    tl_betas_cvsn.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_cvsn_betas-ave.fif'))
    del(tl_betas_cvsn)
    
    tl_tstats_cvsn = mne.EvokedArray(data = np.squeeze(model.get_tstats()[4, :, :]),
                                        info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_tstats_cvsn.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0))
    tl_tstats_cvsn.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_cvsn_tstats-ave.fif'))
    del(tl_tstats_cvsn)
    
    tl_varcope_cvsn = mne.EvokedArray(data = np.squeeze(model.varcopes[4,:,:]),
                                           info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_varcope_cvsn.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0), picks = ['C3', 'C4'])
    tl_varcope_cvsn.save(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_cvsn_varcope-ave.fif'))
    del(tl_varcope_cvsn)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    #DT x probed side
    tl_betas_DTxpside = mne.EvokedArray(data = np.squeeze(model.copes[5,:,:]),
                                       info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_betas_DTxpside.apply_baseline((None, None)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0)) #this baseline demeans the entire epoch
    #tl_betas_DTxpside.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0)) #can add picks=['C3', 'C4', 'C5', 'C6'] to look at lateralised motor electrodes
    tl_betas_DTxpside.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_DTxpside_betas-ave.fif'))
    del(tl_betas_DTxpside)
    
    tl_tstats_DTxpside = mne.EvokedArray(data = np.squeeze(model.get_tstats()[5, :, :]),
                                        info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_tstats_DTxpside.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0))
    tl_tstats_DTxpside.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_DTxpside_tstats-ave.fif'))
    del(tl_tstats_DTxpside)
    
    tl_varcope_DTxpside = mne.EvokedArray(data = np.squeeze(model.varcopes[5,:,:]),
                                           info = epoch.info, tmin = epoch.tmin, nave = epoch.average().nave)
    #tl_varcope_DTxpside.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0), picks = ['C3', 'C4'])
    tl_varcope_DTxpside.save(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tl_DTxpside_varcope-ave.fif'))
    del(tl_varcope_DTxpside)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    del(epoch)
    del(glmdata)
    del(glmdes)
    del(model)
    
#%%   if the betas are already saved ...
subs = np.array([1,2,4,5,6,7,8,9,10])
    
gave_left  = mne.grand_average(alldata_left); gave_left.drop_channels(['RM'])
gave_right = mne.grand_average(alldata_right); gave_right.drop_channels(['RM'])
gave_lvsr  = mne.grand_average(alldata_lvsr); gave_lvsr.drop_channels(['RM']) 


baseline = (-0.1, 0)
vispicks = ['O1', 'O2', 'OZ', 'PO3', 'PO4', 'PO7', 'PO8', 'POZ']
cppicks = ['CZ', 'CPZ', 'PZ', 'CP1', 'CP2']
frontpicks = ['AFZ', 'FPZ', 'AF3', 'AF4', 'FZ']
times = [0.5, 0.75, 1.0, 1.25]


gave_left.apply_baseline(baseline = baseline).plot_joint(
        topomap_args = dict(outlines = 'head', contours = 0), #ts_args = dict(picks = vispicks),
        title = 'grand average feedback response, item probed was on left',
        times = times)

gave_right.apply_baseline(baseline = baseline).plot_joint(
        topomap_args = dict(outlines = 'head', contours = 0), #ts_args = dict(picks = vispicks),
        title = 'grand average feedback response, item probed was on right',
        times = times)

gave_lvsr.apply_baseline(baseline = baseline).plot_joint(
        topomap_args = dict(outlines = 'head', contours = 0), ts_args = dict(picks = vispicks, hline = [0]),
        title = 'grand average feedback response, item probed lvsr',
        times = times)

#
#gave_neutral.apply_baseline(baseline=baseline).plot_joint(
#        topomap_args = dict(outlines='head', contours=0), #ts_args=dict(picks = cppicks, gfp=False, hline=[0]),
#        title = 'grand average feedback response neutral trials - betas',
#        times = times)
#    
#gave_cued.apply_baseline(baseline=baseline).plot_joint(
#        topomap_args = dict(outlines='head', contours=0), #ts_args=dict(picks = cppicks, gfp=False, hline=[0]),
#        title = 'grand average feedback response cued trials - betas',
#        times = times)
#
#gave_cvsn.apply_baseline(baseline=baseline).plot_joint(
#        topomap_args = dict(outlines='head', contours=0), #ts_args=dict(picks = cppicks, gfp=False, hline=[0]),
#        title = 'grand average feedback response cued vs neutral trials - betas',
#        times = times)
#
#gave_absrdif.apply_baseline(baseline=baseline).plot_joint(
#        topomap_args = dict(outlines='head', contours=0), ts_args=dict(picks = vispicks, gfp=False, hline=[0]),
#        title = 'feedback period, main effect of accuracy - betas',
#        times = times)
#
#gave_confwidth.apply_baseline(baseline=baseline).plot_joint(
#        topomap_args = dict(outlines='head', contours=0), ts_args=dict(picks = frontpicks, gfp=False, hline=[0]),
#        title = 'feedback period, main effect of confidence - betas',
#        times = times)
#
#
#gave_absrdifint.apply_baseline(baseline=baseline).plot_joint(
#        topomap_args = dict(outlines='head', contours=0), #ts_args=dict(picks = cppicks, gfp=False, hline=[0]),
#        title = 'feedback period, interaction between accuracy and attention (cue) - betas',
#        times = times)
#
#gave_confwidthint.apply_baseline(baseline=baseline).plot_joint(
#        topomap_args = dict(outlines='head', contours=0), #ts_args=dict(picks = cppicks, gfp=False, hline=[0]),
#        title = 'feedback period, interaction between confidence and attention (cue) - betas',
#        times = times)

