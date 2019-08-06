#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:37:29 2019

@author: sammirc
"""

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
subs = np.array([9, 10]) # memory error when i got to subject 9 :(
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
    regressors.append( glm.regressors.ParametricRegressor(name = 'absrdif', values = tfr.metadata.absrdif.to_numpy(), preproc = 'z', num_observations = glmdata.num_observations) )
    regressors.append( glm.regressors.ParametricRegressor(name = 'confwidth', values = tfr.metadata.confwidth.to_numpy(), preproc = 'z', num_observations = glmdata.num_observations) )
    
    #set up interaction regressors
    cues      = tfr.metadata.cue.to_numpy() #get cue condition for all trials
    cues      = np.where(tfr.metadata.cuetrig==14, -1, cues)    #where cued right, set to -1. 1 = cued left, 0 = neutral (not modelling neutral trials)
    absrdif   = tfr.metadata.absrdif.to_numpy()                 #get absolute deviation on all trials (radians)
    confwidth = np.radians(tfr.metadata.confwidth.to_numpy())   #get width of confidence interval on all trials (set to radians)
#    absrdif = sp.stats.zscore(absrdif)
    absrdif = np.multiply(absrdif, cues)                        #create accuracy interaction term -- this is now lateralised
    confwidth = np.multiply(confwidth, cues)                    #create confidence width interaction term -- this is now lateralised
    
    
    regressors.append( glm.regressors.ParametricRegressor(name = 'absrdif x cuedside', values = absrdif, preproc = None, num_observations = glmdata.num_observations))
    regressors.append( glm.regressors.ParametricRegressor(name = 'confwidth x cuedside', values = confwidth, preproc = None, num_observations = glmdata.num_observations) )


    contrasts = list()
    contrasts.append( glm.design.Contrast([1,  0, 0, 0, 0, 0], 'cued left')             )
    contrasts.append( glm.design.Contrast([0,  1, 0, 0, 0, 0], 'cued right')            )
    contrasts.append( glm.design.Contrast([0,  0, 1, 0, 0, 0], 'absrdif')               )
    contrasts.append( glm.design.Contrast([0,  0, 0, 1, 0, 0], 'confwidth')             )
    contrasts.append( glm.design.Contrast([1, -1, 0, 0, 0, 0], 'left vs right')         )
    contrasts.append( glm.design.Contrast([0,  0, 0, 0, 1, 0], 'absrdif x cuedside')    )
    contrasts.append( glm.design.Contrast([0,  0, 0, 0, 0, 1], 'confwidth x cuedside')  )
    
    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
    #glmdes.plot_summary()
    
    print('\nrunning glm\n')
    model = glm.fit.OLSModel( glmdes, glmdata)

    #cued left average response
    tfr_betas_cleft = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                    data  = np.squeeze(model.copes[0,:,:,:]),
                                                    times = tfr.times, freqs = tfr.freqs, nave  = tfr['cuetrig==13'].average().nave)
    tfr_betas_cleft.save(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_cleft_betas-tfr.h5'), overwrite = True)
            
    del(tfr_betas_cleft)

    #cued right average response
    tfr_betas_cright = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                     data  = np.squeeze(model.copes[1,:,:,:]),
                                                     times = tfr.times, freqs = tfr.freqs, nave  = tfr['cuetrig==14'].average().nave)
    tfr_betas_cright.save(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_cright_betas-tfr.h5'), overwrite = True)

    del(tfr_betas_cright)
    
    #main effect of accuracy
    tfr_betas_absrdif = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                      data  = np.squeeze(model.copes[2, :, :, :]),
                                                      times = tfr.times, freqs = tfr.freqs, nave  = tfr.average().nave)
    tfr_betas_absrdif.save(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdif_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_absrdif)
    
    tfr_tstats_absrdif = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                      data  = np.squeeze(model.get_tstats()[2, :, :, :]),
                                                      times = tfr.times, freqs = tfr.freqs, nave  = tfr.average().nave)
    tfr_tstats_absrdif.save(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdif_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_absrdif)
    
    
    #main effect of confidence width
    tfr_betas_confwidth = mne.time_frequency.AverageTFR(info = tfr.info,
                                                        data = np.squeeze(model.copes[3,:,:,:]),
                                                        times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_betas_confwidth.save(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_confwidth_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_confwidth)
    
    tfr_tstats_confwidth = mne.time_frequency.AverageTFR(info = tfr.info,
                                                         data = np.squeeze(model.get_tstats()[3,:,:,:]),
                                                         times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_tstats_confwidth.save(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_confwidth_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_confwidth)
    
    
    
    #cued left vs right (lateralised attention)
    tfr_betas_lvsr = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                   data  = np.squeeze(model.copes[4,:,:,:]),
                                                   times = tfr.times, freqs = tfr.freqs, nave  = tfr.average().nave)
    tfr_betas_lvsr.save(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_lvsr_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_lvsr)
    
    tfr_tstats_lvsr = mne.time_frequency.AverageTFR(info = tfr.info,
                                                    data = np.squeeze(model.get_tstats()[4,:,:,:]),
                                                    times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_tstats_lvsr.save(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_lvsr_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_lvsr)
    
    
    #lateralised interaction between accuracy and cue (left vs right)
    tfr_betas_absrdifxCue = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                          data  = np.squeeze(model.copes[5,:,:,:]),
                                                          times = tfr.times, freqs = tfr.freqs, nave  = tfr.average().nave)
    tfr_betas_absrdifxCue.save(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdifxCue_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_absrdifxCue)
        
    tfr_tstats_absrdifxCue = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                           data  = np.squeeze(model.get_tstats()[5, :, :, :]),
                                                           times = tfr.times, freqs = tfr.freqs, nave  = tfr.average().nave)
    tfr_tstats_absrdifxCue.save(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdifxCue_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_absrdifxCue)
    
    #lateralised interaction between confidence width and cue (left vs right)
    tfr_betas_confwidthxCue = mne.time_frequency.AverageTFR(info = tfr.info,
                                                            data = np.squeeze(model.copes[6,:,:,:]),
                                                            times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_betas_confwidthxCue.save(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_confwidthxCue_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_confwidthxCue)
    
    tfr_tstats_confwidthxCue = mne.time_frequency.AverageTFR(info = tfr.info,
                                                             data = np.squeeze(model.get_tstats()[6,:,:,:]),
                                                             times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_tstats_confwidthxCue.save(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_confwidthxcue_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_confwidthxCue)                                             
            
    
    
    del(tfr)
    del(glmdata)
    del(glmdes)
    del(model)

#%%   if the betas are already saved ...
subs = np.array([1, 2, 4, 5, 6, 7, 8, 9, 10])

alldata_lvsr        = []
alldata_lvsr_t      = []
alldata_acc         = []
alldata_acc_t       = []
alldata_conf        = []
alldata_conf_t      = []
alldata_int_acc     = []
alldata_int_acc_t   = []
alldata_int_conf    = []
alldata_int_conf_t  = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    lvsr   = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_lvsr_betas-tfr.h5'))
    lvsr   = lvsr[0]; alldata_lvsr.append(lvsr)
    lvsr_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_lvsr_tstats-tfr.h5'))
    lvsr_t = lvsr_t[0]; alldata_lvsr_t.append(lvsr_t)
    
    acc   = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdif_betas-tfr.h5'))
    acc   = acc[0];  alldata_acc.append(acc)
    acc_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdif_tstats-tfr.h5'))
    acc_t = acc_t[0]; alldata_acc_t.append(acc_t)
    
    conf   = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_confwidth_betas-tfr.h5'))
    conf   = conf[0]; alldata_conf.append(conf)
    conf_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_confwidth_tstats-tfr.h5'))
    conf_t = conf_t[0]; alldata_conf_t.append(conf_t)
    
    inter_acc   = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdifxCue_betas-tfr.h5'))
    inter_acc   = inter_acc[0]; alldata_int_acc.append(inter_acc)
    inter_acc_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_absrdifxCue_tstats-tfr.h5'))
    inter_acc_t = inter_acc_t[0]; alldata_int_acc.append(inter_acc_t)
    
    inter_conf   = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_confwidthxCue_betas-tfr.h5'))
    inter_conf   = inter_conf[0]; alldata_int_conf.append(inter_conf)
    inter_conf_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'combined_acc_conf', 'wmConfidence_'+param['subid']+'_cuelocked_tfr_confwidthxcue_tstats-tfr.h5'))
    inter_conf_t = inter_conf_t[0]; alldata_int_conf_t.append(inter_conf_t)
    
    
gave_lvsr       = mne.grand_average(alldata_lvsr)
gave_acc        = mne.grand_average(alldata_acc)
gave_conf       = mne.grand_average(alldata_conf)
gave_inter_acc  = mne.grand_average(alldata_int_acc)
gave_inter_conf = mne.grand_average(alldata_int_conf)




#%%
baseline = (-0.5, -0.3)
timefreqs = {(.50, 10): (.5, 4),
             (.75, 10): (.5, 4),
             (1.0, 10): (.5, 4),
             (1.25, 10):(.5, 4)}

#plot cued left vs cued right
gave_lvsr.drop_channels(['RM'])
gave_lvsr.plot_joint( topomap_args = dict(outlines='head', contours=0,
                                          vmin = np.min(gave_lvsr.data)/5, 
                                          vmax = np.multiply(np.min(gave_lvsr.data)/5, -1)),
                     title = 'grand average of cued left vs right - betas',
                     baseline = baseline, timefreqs = timefreqs)

gave_acc.drop_channels(['RM'])
gave_acc.plot_joint( topomap_args = dict(outlines='head', contours=0,
                                          vmin = np.min(gave_acc.data)/10, 
                                          vmax = np.multiply(np.min(gave_acc.data)/10, -1)),
                     title = 'grand average of main effect of accuracy - betas',
                     baseline = baseline, timefreqs = timefreqs)
        

gave_conf.drop_channels(['RM'])
gave_conf.plot_joint( topomap_args = dict(outlines='head', contours=0,
                                          vmin = np.min(gave_conf.data)/5, 
                                          vmax = np.multiply(np.min(gave_conf.data)/5, -1)),
                     title = 'grand average of main effect of confidence width - betas',
                     baseline = baseline, timefreqs = timefreqs)


gave_inter_acc.drop_channels(['RM'])
gave_inter_acc.plot_joint( topomap_args = dict(outlines='head', contours=0,
                                          vmin = np.min(gave_inter_acc.data)/1, 
                                          vmax = np.multiply(np.min(gave_inter_acc.data)/1, -1)),
                     title = 'grand average of interaction between accuracy and cued side - betas',
                     baseline = baseline, timefreqs = timefreqs)


gave_inter_conf.drop_channels(['RM'])
gave_inter_conf.plot_joint( topomap_args = dict(outlines='head', contours=0,
                                          vmin = np.min(gave_inter_conf.data)/5, 
                                          vmax = np.multiply(np.min(gave_inter_conf.data)/5, -1)),
                     title = 'grand average of interaction between confidence width and cued side - betas',
                     baseline = baseline, timefreqs = timefreqs)




#%%



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




