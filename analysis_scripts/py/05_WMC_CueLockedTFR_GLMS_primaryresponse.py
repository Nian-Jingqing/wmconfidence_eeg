#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:06:09 2019

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
#%% only needs running if cuelocked TFR glms not already present
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    #get tfr
    tfr = mne.time_frequency.read_tfrs(fname=param['cuelocked_tfr']); tfr=tfr[0]
 
    tfr.metadata = pd.DataFrame.from_csv(path=param['cuelocked_tfr_meta'], index_col=None) #read in and attach metadata
    
    glmdata = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 250)
    
    regressors = list()
    regressors.append( glm.regressors.CategoricalRegressor(category_list = tfr.metadata.cuetrig.to_numpy(), codes = 13, name = 'cued left') )
    regressors.append( glm.regressors.CategoricalRegressor(category_list = tfr.metadata.cuetrig.to_numpy(), codes = 14, name = 'cued right') )
    regressors.append( glm.regressors.ParametricRegressor(name = 'absrdif', values = tfr.metadata.absrdif.to_numpy(), preproc = 'z', num_observations = glmdata.num_observations))

    cues    = tfr.metadata.cue.to_numpy()
    cues    = np.where(tfr.metadata.cuetrig==14, -1, cues)
    absrdif = tfr.metadata.absrdif.to_numpy()
#    absrdif = sp.stats.zscore(absrdif)
    absrdif = np.multiply(absrdif, cues)
    regressors.append( glm.regressors.ParametricRegressor(name = 'absrdif x cuedside', values = absrdif, preproc = None, num_observations = glmdata.num_observations))    


    contrasts = list()
    contrasts.append( glm.design.Contrast([1,  0, 0, 0], 'cued left') )
    contrasts.append( glm.design.Contrast([0,  1, 0, 0], 'cued right') )
    contrasts.append( glm.design.Contrast([0,  0, 1, 0], 'absrdif') )
    contrasts.append( glm.design.Contrast([1, -1, 0, 0], 'left vs right') )
    contrasts.append( glm.design.Contrast([0,  0, 0, 1], 'absrdif x cuedside'))
    
    
    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
    #glmdes.plot_summary()
    
    print('\nrunning glm\n')
    model = glm.fit.OLSModel( glmdes, glmdata)

#    tfr_betas_cleft = mne.time_frequency.AverageTFR(info  = tfr.info,
#                                                    data  = np.squeeze(model.copes[0,:,:,:]),
#                                                    times = tfr.times,
#                                                    freqs = tfr.freqs,
#                                                    nave  = tfr['cuetrig==13'].average().nave 
#                                                    )
#    tfr_betas_cleft.save(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_cleft_betas-tfr.h5'), overwrite = True)
#            
#    del(globals()['tfr_betas_cleft'])
#
#
#    tfr_betas_cright = mne.time_frequency.AverageTFR(info  = tfr.info,
#                                                     data  = np.squeeze(model.copes[1,:,:,:]),
#                                                     times = tfr.times,
#                                                     freqs = tfr.freqs,
#                                                     nave  = tfr['cuetrig==14'].average().nave
#                                                     )
#    tfr_betas_cright.save(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_cright_betas-tfr.h5'), overwrite = True)

    #del(globals()['tfr_betas_cright'])
    
#    tfr_betas_absrdif = mne.time_frequency.AverageTFR(info  = tfr.info,
#                                                      data  = np.squeeze(model.copes[2, :, :, :]),
#                                                      times = tfr.times,
#                                                      freqs = tfr.freqs,
#                                                      nave  = tfr.average().nave
#                                                      )
#    tfr_betas_absrdif.save(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdif_betas-tfr.h5'), overwrite = True)
#    del(globals()['tfr_betas_absrdif'])
    
    tfr_tstats_absrdif = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                      data  = np.squeeze(model.get_tstats()[2, :, :, :]),
                                                      times = tfr.times,
                                                      freqs = tfr.freqs,
                                                      nave  = tfr.average().nave
                                                      )
    tfr_tstats_absrdif.save(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdif_tstats-tfr.h5'), overwrite = True)
    del(globals()['tfr_tstats_absrdif'])
    
    
#    tfr_betas_lvsr = mne.time_frequency.AverageTFR(info  = tfr.info,
#                                                   data  = np.squeeze(model.copes[3,:,:,:]),
#                                                   times = tfr.times,
#                                                   freqs = tfr.freqs,
#                                                   nave  = tfr.average().nave
#                                                   )
#    tfr_betas_lvsr.save(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_lvsr_betas-tfr.h5'), overwrite = True)
#
#    del(globals()['tfr_betas_lvsr'])
    
#    tfr_betas_absrdifxCue = mne.time_frequency.AverageTFR(info  = tfr.info,
#                                                          data  = np.squeeze(model.copes[4,:,:,:]),
#                                                          times = tfr.times,
#                                                          freqs = tfr.freqs,
#                                                          nave  = tfr.average().nave
#                                                          )
#    tfr_betas_absrdifxCue.save(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdifxCue_betas-tfr.h5'), overwrite = True)
#    del(globals()['tfr_betas_absrdifxCue'])
        
    tfr_tstats_absrdifxCue = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                      data  = np.squeeze(model.get_tstats()[4, :, :, :]),
                                                      times = tfr.times,
                                                      freqs = tfr.freqs,
                                                      nave  = tfr.average().nave
                                                      )
    tfr_tstats_absrdifxCue.save(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdifxCue_tstats-tfr.h5'), overwrite = True)
    del(globals()['tfr_tstats_absrdifxCue'])
    
    
    del(tfr)
    del(glmdata)
    del(glmdes)
    del(model)

#%%   if the betas are already saved ...

alldata_absrdifxCue = []
alldata_absrdifxCue_tstat = []
alldata_absrdif = []
alldata_absrdif_tstat = []
for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)


    absrdifxCue = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdifxCue_betas-tfr.h5')); absrdifxCue = absrdifxCue[0]
    alldata_absrdifxCue.append(absrdifxCue)
    absrdif = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdif_betas-tfr.h5')); absrdif = absrdif[0]
    alldata_absrdif.append(absrdif)
    
    inter_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdifxCue_tstats-tfr.h5')); inter_t = inter_t[0]
    alldata_absrdifxCue_tstat.append(inter_t)
    
    absrdif_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdif_tstats-tfr.h5')); absrdif_t = absrdif_t[0]
    alldata_absrdif_tstat.append(absrdif_t)

gave_absrdifxCue    = mne.grand_average(alldata_absrdifxCue)
gave_interaction_t  = mne.grand_average(alldata_absrdifxCue_tstat)
gave_absrdif        = mne.grand_average(alldata_absrdif)
gave_absrdif_t      = mne.grand_average(alldata_absrdif_tstat)


baseline = (-0.5, -0.3)


gave_absrdifxCue.plot_joint(topomap_args = dict(outlines='head', contours=0,
                                            vmin = np.min(gave_absrdifxCue.data)/10,
                                            vmax = np.multiply(np.min(gave_absrdifxCue.data)/10, -1)),
                       title = 'grand average interaction between cue side and absolute response deviation',
                       baseline = baseline,
                       timefreqs = {
                             (.50, 10): (.5, 4),
                             (.75, 10): (.5, 4),
                             (1.0, 10): (.5, 4),
                             (1.25, 10):(.5, 4)
                             })

gave_absrdif.drop_channels(['RM'])
gave_absrdif.plot_joint(topomap_args = dict(outlines='head', contours=0,
                                            vmin = np.min(gave_absrdif.data)/10,
                                            vmax = np.multiply(np.min(gave_absrdif.data)/10, -1)),
                       title = 'grand average main effect of absolute response deviation',
                       baseline = baseline,
                       timefreqs = {
                             (.50, 10): (.5, 4),
                             (.75, 10): (.5, 4),
                             (1.0, 10): (.5, 4),
                             (1.25, 10):(.5, 4)
                             })
    

gave_interaction_t.drop_channels(['RM'])
gave_interaction_t.plot_joint(topomap_args = dict(outlines='head', contours=0,vmin = -1.5, vmax = 1.5),
                       title = 'grand average interaction between cue side and absolute response deviation - tstats',
                       baseline = baseline,
                       timefreqs = {
                             (.50, 10): (.5, 4),
                             (.75, 10): (.5, 4),
                             (1.0, 10): (.5, 4),
                             (1.25, 10):(.5, 4)
                             })

gave_absrdif_t.drop_channels(['RM'])
gave_absrdif_t.plot_joint(topomap_args = dict(outlines='head', contours=0, vmin=-1.5, vmax=1.5),
                       title = 'grand average main effect of absolute response deviation - t stat',
                       baseline = baseline,
                       timefreqs = {
                             (.50, 10): (.5, 4),
                             (.75, 10): (.5, 4),
                             (1.0, 10): (.5, 4),
                             (1.25, 10):(.5, 4)
                             })




