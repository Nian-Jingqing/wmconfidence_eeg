#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:31:30 2019

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
#subs = np.array([5,6,7,8,9,10])
#subs = np.array([1,2,4,5,6])
#subs = np.array([7,8,9,10]) #memory error for subject 6 so running again separately
subs = np.array([9, 10]) #failed again, memory error on subject 9

#%% only needs running if cuelocked TFR glms not already present
#designs = []
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    #get tfr
    tfr = mne.time_frequency.read_tfrs(fname=param['resplocked_tfr']); tfr=tfr[0]
 
    tfr.metadata = pd.DataFrame.from_csv(path=param['resplocked_tfr_meta'], index_col=None) #read in and attach metadata
    
    glmdata = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 250)
    

    #get some behavioural things we're going to look at
    trials = np.ones(glmdata.num_observations) #regressor for just grand mean response
    cues   = tfr.metadata.cue.to_numpy()
    pside = tfr.metadata.pside.to_numpy()
    pside = np.where(pside == 0, 1, -1)
    
    regressors = list()
    regressors.append( glm.regressors.ParametricRegressor(name = 'grand mean', values = trials, preproc = None, num_observations = glmdata.num_observations))
    regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 0, name = 'neutral'))
    regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 1, name = 'cued'))
    regressors.append( glm.regressors.ParametricRegressor(name = 'DT', values = tfr.metadata.DT.to_numpy(), preproc='z', num_observations = glmdata.num_observations))
    regressors.append( glm.regressors.ParametricRegressor(name = 'pside', values = pside, preproc = None, num_observations = glmdata.num_observations))
    regressors.append( glm.regressors.ParametricRegressor(name = 'DTxpside',values = np.multiply(tfr.metadata.DT.to_numpy(),pside), preproc = 'z', num_observations = glmdata.num_observations))
    
    
    contrasts = list()
    contrasts.append(glm.design.Contrast([ 1,  0,  0,  0,  0,  0], 'grand mean')      )
    contrasts.append(glm.design.Contrast([ 0,  1,  0,  0,  0,  0], 'neutral')         )
    contrasts.append(glm.design.Contrast([ 0,  0,  1,  0,  0,  0], 'cued')            )
    contrasts.append(glm.design.Contrast([ 0,  0,  0,  1,  0,  0], 'DT')              )
    contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  1,  0], 'pside'))
    contrasts.append(glm.design.Contrast([ 0, -1,  1,  0,  0,  0], 'cued vs neutral') ) 
    contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  1], 'DT x pside'))
    
    
    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts); #designs.append(glmdes)
    #glmdes.plot_summary()
    
    total_nave = len(tfr)
    neut_nave  = len(tfr['cue==0'])
    cued_nave  = len(tfr['cue==1'])
    lside_nave = len(tfr['pside==0'])
    rside_nave = len(tfr['pside==1'])
    
    times = tfr.times
    freqs = tfr.freqs
    info = tfr.info
    
    del(tfr)
    print('\n - - - - -  running glm - - - - - \n')
    model = glm.fit.OLSModel( glmdes, glmdata )
    del(glmdata) #glmdata and design aren't used now, so can delete from ram to save space and try to prevent memory errors happening
    del(glmdes)
    
    #grand mean
    tfr_betas_grandmean = mne.time_frequency.AverageTFR(data = np.squeeze(model.copes[0,:,:]),
                                       info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_betas_grandmean.apply_baseline((None, None)).plot_joint(picks='eeg', topomap_args=dict(outlines='head', contours=0)) #this baseline demeans the entire tfr
    #tfr_betas_grandmean.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0)) #can add picks=['C3', 'C4', 'C5', 'C6'] to look at lateralised motor electrodes
    tfr_betas_grandmean.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_grandmean_betas-tfr.h5'))
    del(tfr_betas_grandmean)
    
    tfr_tstats_grandmean = mne.time_frequency.AverageTFR(data = np.squeeze(model.get_tstats()[0, :, :]),
                                        info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_tstats_grandmean.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0))
    tfr_tstats_grandmean.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_grandmean_tstats-tfr.h5'))
    del(tfr_tstats_grandmean)
    
    tfr_varcope_grandmean = mne.time_frequency.AverageTFR(data = np.squeeze(model.varcopes[0,:,:]),
                                           info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_varcope_grandmean.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0), picks = ['C3', 'C4'])
    tfr_varcope_grandmean.save(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_grandmean_varcope-tfr.h5'))
    del(tfr_varcope_grandmean)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    #neutral
    tfr_betas_neutral = mne.time_frequency.AverageTFR(data = np.squeeze(model.copes[1,:,:]),
                                       info = info, times = times, freqs = freqs, nave = neut_nave)
    #tfr_betas_neutral.apply_baseline((None, None)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0)) #this baseline demeans the entire tfr
    #tfr_betas_neutral.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0)) #can add picks=['C3', 'C4', 'C5', 'C6'] to look at lateralised motor electrodes
    tfr_betas_neutral.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_neutral_betas-tfr.h5'))
    del(tfr_betas_neutral)
    
    tfr_tstats_neutral = mne.time_frequency.AverageTFR(data = np.squeeze(model.get_tstats()[1, :, :]),
                                        info = info, times = times, freqs = freqs, nave = neut_nave)
    #tfr_tstats_neutral.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0))
    tfr_tstats_neutral.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_neutral_tstats-tfr.h5'))
    del(tfr_tstats_neutral)
    
    tfr_varcope_neutral = mne.time_frequency.AverageTFR(data = np.squeeze(model.varcopes[1,:,:]),
                                           info = info, times = times, freqs = freqs, nave = neut_nave)
    #tfr_varcope_neutral.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0), picks = ['C3', 'C4'])
    tfr_varcope_neutral.save(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_neutral_varcope-tfr.h5'))
    del(tfr_varcope_neutral)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    #cued
    tfr_betas_cued = mne.time_frequency.AverageTFR(data = np.squeeze(model.copes[2,:,:]),
                                       info = info, times = times, freqs = freqs, nave = cued_nave)
    #tfr_betas_cued.apply_baseline((None, None)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0)) #this baseline demeans the entire tfr
    #tfr_betas_cued.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0)) #can add picks=['C3', 'C4', 'C5', 'C6'] to look at lateralised motor electrodes
    tfr_betas_cued.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_cued_betas-tfr.h5'))
    del(tfr_betas_cued)
    
    tfr_tstats_cued = mne.time_frequency.AverageTFR(data = np.squeeze(model.get_tstats()[2, :, :]),
                                        info = info, times = times, freqs = freqs, nave = cued_nave)
    #tfr_tstats_cued.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0))
    tfr_tstats_cued.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_cued_tstats-tfr.h5'))
    del(tfr_tstats_cued)
    
    tfr_varcope_cued = mne.time_frequency.AverageTFR(data = np.squeeze(model.varcopes[2,:,:]),
                                           info = info, times = times, freqs = freqs, nave = cued_nave)
    #tfr_varcope_cued.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0), picks = ['C3', 'C4'])
    tfr_varcope_cued.save(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_cued_varcope-tfr.h5'))
    del(tfr_varcope_cued)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    #DT
    tfr_betas_DT = mne.time_frequency.AverageTFR(data = np.squeeze(model.copes[3,:,:]),
                                       info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_betas_DT.apply_baseline((None, None)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0)) #this baseline demeans the entire tfr
    #tfr_betas_DT.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0)) #can add picks=['C3', 'C4', 'C5', 'C6'] to look at lateralised motor electrodes
    tfr_betas_DT.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_DT_betas-tfr.h5'))
    del(tfr_betas_DT)
    
    tfr_tstats_DT = mne.time_frequency.AverageTFR(data = np.squeeze(model.get_tstats()[3, :, :]),
                                        info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_tstats_DT.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0))
    tfr_tstats_DT.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_DT_tstats-tfr.h5'))
    del(tfr_tstats_DT)
    
    tfr_varcope_DT = mne.time_frequency.AverageTFR(data = np.squeeze(model.varcopes[3,:,:]),
                                           info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_varcope_cued.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0), picks = ['C3', 'C4'])
    tfr_varcope_DT.save(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_DT_varcope-tfr.h5'))
    del(tfr_varcope_DT)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    #pside
    tfr_betas_pside = mne.time_frequency.AverageTFR(data = np.squeeze(model.copes[4,:,:]),
                                       info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_betas_pside.apply_baseline((None, None)).plot_joint(picks='eeg', topomap_args=dict(outlines='head', contours=0)) #this baseline demeans the entire tfr
    #tfr_betas_pside.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0)) #can add picks=['C3', 'C4', 'C5', 'C6'] to look at lateralised motor electrodes
    tfr_betas_pside.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_pside_betas-tfr.h5'))
    del(tfr_betas_pside)
    
    tfr_tstats_pside = mne.time_frequency.AverageTFR(data = np.squeeze(model.get_tstats()[4, :, :]),
                                        info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_tstats_pside.plot_joint(topomap_args=dict(outlines='head', contours=0), baseline=(None,None), picks=range(61))
    tfr_tstats_pside.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_pside_tstats-tfr.h5'))
    del(tfr_tstats_pside)
    
    tfr_varcope_pside = mne.time_frequency.AverageTFR(data = np.squeeze(model.varcopes[4,:,:]),
                                           info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_varcope_pside.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0), picks = ['C3', 'C4'])
    tfr_varcope_pside.save(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_pside_varcope-tfr.h5'))
    del(tfr_varcope_pside)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    
    #cued vs neutral
    tfr_betas_cvsn = mne.time_frequency.AverageTFR(data = np.squeeze(model.copes[5,:,:]),
                                       info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_betas_cvsn.apply_baseline((None, None)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0)) #this baseline demeans the entire tfr
    #tfr_betas_cvsn.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0)) #can add picks=['C3', 'C4', 'C5', 'C6'] to look at lateralised motor electrodes
    tfr_betas_cvsn.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_cvsn_betas-tfr.h5'))
    del(tfr_betas_cvsn)
    
    tfr_tstats_cvsn = mne.time_frequency.AverageTFR(data = np.squeeze(model.get_tstats()[5, :, :]),
                                        info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_tstats_cvsn.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0))
    tfr_tstats_cvsn.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_cvsn_tstats-tfr.h5'))
    del(tfr_tstats_cvsn)
    
    tfr_varcope_cvsn = mne.time_frequency.AverageTFR(data = np.squeeze(model.varcopes[5,:,:]),
                                           info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_varcope_cvsn.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0), picks = ['C3', 'C4'])
    tfr_varcope_cvsn.save(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_cvsn_varcope-tfr.h5'))
    del(tfr_varcope_cvsn)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
    #DT x probed side
    tfr_betas_DTxpside = mne.time_frequency.AverageTFR(data = np.squeeze(model.copes[6,:,:]),
                                       info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_betas_DTxpside.apply_baseline((None, None)).plot_joint(times='auto', picks='eeg', topomap_args=dict(outlines='head', contours=0)) #this baseline demeans the entire tfr
    #tfr_betas_DTxpside.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0)) #can add picks=['C3', 'C4', 'C5', 'C6'] to look at lateralised motor electrodes
    tfr_betas_DTxpside.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_DTxpside_betas-tfr.h5'))
    del(tfr_betas_DTxpside)
    
    tfr_tstats_DTxpside = mne.time_frequency.AverageTFR(data = np.squeeze(model.get_tstats()[6, :, :]),
                                        info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_tstats_DTxpside.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0))
    tfr_tstats_DTxpside.save(fname=op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_DTxpside_tstats-tfr.h5'))
    del(tfr_tstats_DTxpside)
    
    tfr_varcope_DTxpside = mne.time_frequency.AverageTFR(data = np.squeeze(model.varcopes[6,:,:]),
                                           info = info, times = times, freqs = freqs, nave = total_nave)
    #tfr_varcope_DTxpside.plot_joint(times='auto', topomap_args=dict(outlines='head', contours=0), picks = ['C3', 'C4'])
    tfr_varcope_DTxpside.save(fname = op.join(param['path'], 'glms', 'response', 'wmConfidence_'+param['subid']+'_resplocked_tfr_DTxpside_varcope-tfr.h5'))
    del(tfr_varcope_DTxpside)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    del(model)