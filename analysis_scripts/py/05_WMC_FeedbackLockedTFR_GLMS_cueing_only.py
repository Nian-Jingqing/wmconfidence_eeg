#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:51:19 2019

@author: sammirc
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:54:35 2019

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
subs = np.array([1,2,4,5,6])
subs = np.array([7,8,9,10])
#subs = np.array([8, 9, 10]) #memory error on subject 8 so need to re run
#%% only needs running if cuelocked TFR glms not already present
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    #get tfr
    
    tfr = mne.time_frequency.read_tfrs(fname=param['fblocked_tfr']); tfr=tfr[0]
#    if tfr.info['sfreq'] != 250:
#        tfr.resample(250) #resample to 250Hz so all are in the same sample rate
    
    tfr.metadata = pd.DataFrame.from_csv(path=param['fblocked_tfr_meta'], index_col=None) #read in and attach metadata

    glmdata = glm.data.TrialGLMData(data = tfr.data, time_dim=3, sample_rate=250)
    
    
    cues = tfr.metadata.cue.to_numpy()
    
    regressors = list()
    regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 0, name = 'neutral') )
    regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 1, name = 'cued')    )
    
    cues = np.where(cues==0, -1, cues) #set neutral trials to -1, cued trials to 1 for interaction terms
    absrdif = tfr.metadata.absrdif.to_numpy()
    absrdifint = np.multiply(absrdif, cues)
    
    confwidth = np.radians(tfr.metadata.confwidth.to_numpy())
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
    tfr_betas_neutral = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.copes[0,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr['cue==0'].average().nave)
    tfr_betas_neutral.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_neutral_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_neutral)
    
    tfr_tstats_neutral = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.get_tstats()[0,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr['cue==0'].average().nave)
    tfr_tstats_neutral.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_neutral_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_neutral)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    #cued trials
    tfr_betas_cued = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.copes[1,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr['cue==1'].average().nave)
    tfr_betas_cued.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_cued_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_cued)
    
    tfr_tstats_cued = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.get_tstats()[1,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr['cue==1'].average().nave)
    tfr_tstats_cued.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_cued_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_cued)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    #main effect of response accuracy
    tfr_betas_absrdif = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.copes[2,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_betas_absrdif.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_absrdif_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_absrdif)
    
    tfr_tstats_absrdif = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.get_tstats()[2,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_tstats_absrdif.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_absrdif_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_absrdif)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    #main effect of confidence width
    tfr_betas_confwidth = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.copes[3,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_betas_confwidth.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_confwidth_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_confwidth)
    
    tfr_tstats_confwidth = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.get_tstats()[3,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_tstats_confwidth.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_confwidth_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_confwidth)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    #interaction between cue and accuracy
    tfr_betas_accxcue = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.copes[4,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_betas_accxcue.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_accxcue_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_accxcue)
    
    tfr_tstats_accxcue = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.get_tstats()[4,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_tstats_accxcue.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_accxcue_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_accxcue)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    #interaction between cue and confidence
    tfr_betas_confxcue = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.copes[5,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_betas_confxcue.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_confxcue_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_confxcue)
    
    tfr_tstats_confxcue = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.get_tstats()[5,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_tstats_confxcue.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_confxcue_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_confxcue)   
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    #cued vs neutral
    tfr_betas_cvsn = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.copes[6,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_betas_cvsn.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_cvsn_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_cvsn)
    
    tfr_tstats_cvsn = mne.time_frequency.AverageTFR(info = tfr.info,
                                                      data = np.squeeze(model.get_tstats()[6,:,:,:]),
                                                      times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_tstats_cvsn.save(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_cvsn_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_cvsn)
    
    
    del(tfr)
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

    neutral = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_neutral_betas-tfr.h5')); neutral = neutral[0]
    alldata_neutral.append(neutral)
    neutral_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_neutral_tstats-tfr.h5')); neutral_t = neutral_t[0]
    alldata_neutral_t.append(neutral_t)
    
    cued = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_cued_betas-tfr.h5')); cued = cued[0]
    alldata_cued.append(cued)
    cued_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_cued_tstats-tfr.h5')); cued_t = cued_t[0]
    alldata_cued_t.append(cued_t)
    
    
    
    absrdif = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_absrdif_betas-tfr.h5')); absrdif = absrdif[0]
    alldata_absrdif.append(absrdif)
    absrdif_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_absrdif_tstats-tfr.h5')); absrdif_t = absrdif_t[0]
    alldata_absrdif_t.append(absrdif_t)
    
    confwidth = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_confwidth_betas-tfr.h5')); confwidth = confwidth[0]
    alldata_confwidth.append(confwidth)
    confwidth_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_confwidth_tstats-tfr.h5')); confwidth_t = confwidth_t[0]
    alldata_confwidth_t.append(confwidth_t)
    
    absrdif_int = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_accxcue_betas-tfr.h5')); absrdif_int = absrdif_int[0]
    alldata_absrdif_int.append(absrdif_int)
    absrdif_int_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_accxcue_tstats-tfr.h5')); absrdif_int_t = absrdif_int_t[0]
    alldata_absrdif_int_t.append(absrdif_int_t)
    
    confwidth_int = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_confxcue_betas-tfr.h5')); confwidth_int = confwidth_int[0]
    alldata_confwidth_int.append(confwidth_int)
    confwidth_int_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_confxcue_tstats-tfr.h5')); confwidth_int_t = confwidth_int_t[0]
    alldata_confwidth_int_t.append(confwidth_int_t)
    
    
    cvsn = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_cvsn_betas-tfr.h5')); cvsn = cvsn[0]
    alldata_cvsn.append(cvsn)
    cvsn_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback', 'tfr_fullglm', 'wmConfidence_'+param['subid']+'_fblocked_tfr_cvsn_tstats-tfr.h5')); cvsn_t = cvsn_t[0]
    alldata_cvsn_t.append(cvsn_t)
    

gave_neutral        = mne.grand_average(alldata_neutral)       ; gave_neutral.drop_channels(['RM'])
gave_cued           = mne.grand_average(alldata_cued)          ; gave_cued.drop_channels(['RM'])
gave_absrdif        = mne.grand_average(alldata_absrdif)       ; gave_absrdif.drop_channels(['RM'])
gave_confwidth      = mne.grand_average(alldata_confwidth)     ; gave_confwidth.drop_channels(['RM'])
gave_absrdifint     = mne.grand_average(alldata_absrdif_int)   ; gave_absrdifint.drop_channels(['RM'])
gave_confwidthint   = mne.grand_average(alldata_confwidth_int) ; gave_confwidthint.drop_channels(['RM'])
gave_cvsn           = mne.grand_average(alldata_cvsn)          ; gave_cvsn.drop_channels(['RM'])


baseline = (-0.5, -0.3)
timefreqs = {(.50, 10): (.5, 4),
             (.75, 10): (.5, 4),
             (1.0, 10): (.5, 4),
             (1.25, 10):(.5, 4)}


gave_neutral.plot_joint(topomap_args = dict(outlines='head', contours=0),
                       title = 'grand average feedback response neutral trials - betas',
                       baseline = baseline,
                       timefreqs = timefreqs)
    
gave_cued.plot_joint(topomap_args = dict(outlines='head', contours=0),
                       title = 'grand average feedback response cued trials - betas',
                       baseline = baseline,
                       timefreqs = timefreqs)

gave_absrdif.plot_joint(topomap_args = dict(outlines='head', contours=0,
                                          vmin = np.min(gave_absrdif.data)/5, 
                                          vmax = np.multiply(np.min(gave_absrdif.data)/5, -1)),
                       title = 'feedback period, main effect of error - betas',
                       baseline = baseline,
                       timefreqs = timefreqs)

gave_confwidth.plot_joint(topomap_args = dict(outlines='head', contours=0,
                                          vmin = np.min(gave_confwidth.data)/10, 
                                          vmax = np.multiply(np.min(gave_confwidth.data)/10, -1)),
                       title = 'feedback period, main effect of confidence - betas',
                       baseline = baseline,
                       timefreqs = timefreqs)


gave_absrdifint.plot_joint(topomap_args = dict(outlines='head', contours=0,
                                          vmin = np.min(gave_absrdifint.data)/5, 
                                          vmax = np.multiply(np.min(gave_absrdifint.data)/5, -1)),
                       title = 'feedback period, interaction between error and attention (cue) - betas',
                       baseline = baseline,
                       timefreqs = timefreqs)

gave_confwidthint.plot_joint(topomap_args = dict(outlines='head', contours=0,
                                          vmin = np.min(gave_confwidthint.data)/10, 
                                          vmax = np.multiply(np.min(gave_confwidthint.data)/10, -1)),
                       title = 'feedback period, interaction between confidence and attention (cue) - betas',
                       baseline = baseline,
                       timefreqs = timefreqs)



gave_cvsn.plot_joint(topomap_args = dict(outlines='head', contours=0,
                                          vmin = np.min(gave_cvsn.data)/5, 
                                          vmax = np.multiply(np.min(gave_cvsn.data)/5, -1)),
                       title = 'grand average feedback response cued vs neutral trials - betas',
                       baseline = baseline,
                       timefreqs = timefreqs)



