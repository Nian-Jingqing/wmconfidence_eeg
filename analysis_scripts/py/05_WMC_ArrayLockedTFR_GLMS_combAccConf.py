#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:37:29 2019

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
#subs = np.array([9, 10]) # memory error when i got to subject 9 :(
#%% only needs running if cuelocked TFR glms not already present
alldata_ave_varcopes = []
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    #get tfr
    tfr = mne.time_frequency.read_tfrs(fname=param['arraylocked_tfr']); tfr=tfr[0]
 
    tfr.metadata = pd.DataFrame.from_csv(path=param['arraylocked_tfr_meta'], index_col=None) #read in and attach metadata
    
    glmdata = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 250)
    
    regressors = list()
    regressors.append( glm.regressors.ParametricRegressor(name = 'trials', values = np.ones(shape=(tfr.metadata.shape[0],)), preproc=None, num_observations = glmdata.num_observations))
    regressors.append( glm.regressors.ParametricRegressor(name = 'absrdif', values = tfr.metadata.absrdif.to_numpy(), preproc = 'z', num_observations = glmdata.num_observations) )
    regressors.append( glm.regressors.ParametricRegressor(name = 'confwidth', values = np.radians(np.multiply(tfr.metadata.confwidth.to_numpy(), -1)), preproc = 'z', num_observations = glmdata.num_observations) )

    contrasts = list()
    contrasts.append( glm.design.Contrast([1, 0, 0], 'average response' ) )
    contrasts.append( glm.design.Contrast([0, 1, 0], 'absrdif') )
    contrasts.append( glm.design.Contrast([0, 0, 1], 'confwidth') )
    
    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
    glmdes.plot_summary()
    
    print('\nrunning glm\n')
    model = glm.fit.OLSModel( glmdes, glmdata)

    #average array evoked response
    tfr_betas_ave = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                  data  = np.squeeze(model.copes[0,:,:,:]),
                                                  times = tfr.times, freqs = tfr.freqs, nave  = tfr.average().nave)
    tfr_betas_ave.save(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_ave_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_ave)
    
    tfr_tstats_ave = mne.time_frequency.AverageTFR(info = tfr.info,
                                                   data = np.squeeze(model.get_tstats()[0,:,:,:]),
                                                   times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_tstats_ave.save(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_ave_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_ave)
    
    tfr_varcopes_ave = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                  data  = np.squeeze(model.varcopes[0,:,:,:]),
                                                  times = tfr.times, freqs = tfr.freqs, nave  = tfr.average().nave)
    tfr_varcopes_ave.save(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_ave_varcopes-tfr.h5'), overwrite = True)
#    alldata_ave_varcopes.append(tfr_varcopes_ave)
    del(tfr_varcopes_ave)
    
    
    #association with error/accuracy (absrdif)
    tfr_betas_absrdif = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                  data  = np.squeeze(model.copes[1,:,:,:]),
                                                  times = tfr.times, freqs = tfr.freqs, nave  = tfr.average().nave)
    tfr_betas_absrdif.save(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_absrdif_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_absrdif)
    
    tfr_tstats_absrdif = mne.time_frequency.AverageTFR(info = tfr.info,
                                                   data = np.squeeze(model.get_tstats()[1,:,:,:]),
                                                   times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_tstats_absrdif.save(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_absrdif_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_absrdif)
    
    tfr_varcopes_absrdif = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                  data  = np.squeeze(model.varcopes[1,:,:,:]),
                                                  times = tfr.times, freqs = tfr.freqs, nave  = tfr.average().nave)
    tfr_varcopes_absrdif.save(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_absrdif_varcopes-tfr.h5'), overwrite = True)
#    alldata_ave_varcopes.append(tfr_varcopes_ave)
    del(tfr_varcopes_absrdif)
    
    #association with confidence (confwidth)
    tfr_betas_confwidth = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                  data  = np.squeeze(model.copes[2,:,:,:]),
                                                  times = tfr.times, freqs = tfr.freqs, nave  = tfr.average().nave)
    tfr_betas_confwidth.save(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_confwidth_betas-tfr.h5'), overwrite = True)
    del(tfr_betas_confwidth)
    
    tfr_tstats_confwidth = mne.time_frequency.AverageTFR(info = tfr.info,
                                                   data = np.squeeze(model.get_tstats()[2,:,:,:]),
                                                   times = tfr.times, freqs = tfr.freqs, nave = tfr.average().nave)
    tfr_tstats_confwidth.save(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_confwidth_tstats-tfr.h5'), overwrite = True)
    del(tfr_tstats_confwidth)
    
    tfr_varcopes_confwidth = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                  data  = np.squeeze(model.varcopes[2,:,:,:]),
                                                  times = tfr.times, freqs = tfr.freqs, nave  = tfr.average().nave)
    tfr_varcopes_confwidth.save(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_confwidth_varcopes-tfr.h5'), overwrite = True)
#    alldata_ave_varcopes.append(tfr_varcopes_ave)
    del(tfr_varcopes_confwidth)
    
    del(tfr)
    del(glmdata)
    del(glmdes)
    del(model)

#%%   if the betas are already saved ...
subs = np.array([1, 2, 4, 5, 6, 7, 8, 9, 10])

alldata_ave   = []; alldata_ave_t   = []
alldata_error = []; alldata_error_t = []
alldata_conf  = []; alldata_conf_t = []
alldata_varcopes = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    ave   = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_ave_betas-tfr.h5'))
    ave   = ave[0]; alldata_ave.append(ave)
    ave_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_ave_tstats-tfr.h5'))
    ave_t = ave_t[0]; alldata_ave_t.append(ave_t)
    
    ave_vcopes = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_ave_varcopes-tfr.h5'))
    ave_vcopes = ave_vcopes[0]; alldata_varcopes.append(ave_vcopes)
    
    acc   = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_absrdif_betas-tfr.h5'))
    acc   = acc[0];  alldata_error.append(acc)
    acc_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_absrdif_tstats-tfr.h5'))
    acc_t = acc_t[0]; alldata_error_t.append(acc_t)
    
    conf   = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_confwidth_betas-tfr.h5'))
    conf   = conf[0]; alldata_conf.append(conf)
    conf_t = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'array', 'wmConfidence_'+param['subid']+'_arraylocked_tfr_confwidth_tstats-tfr.h5'))
    conf_t = conf_t[0]; alldata_conf_t.append(conf_t)

baseline = (-0.5, -0.3)
timefreqs = {(.50, 10): (.5, 4),
             (.75, 10): (.5, 4)}


    
gave_ave      = mne.grand_average(alldata_ave)
gave_error    = mne.grand_average(alldata_error)
gave_conf     = mne.grand_average(alldata_conf)
gave_varcopes = mne.grand_average(alldata_varcopes)

gave_ave_t   = mne.grand_average(alldata_ave_t)
gave_error_t = mne.grand_average(alldata_error_t)
gave_conf_t  = mne.grand_average(alldata_conf_t)


#%%
baseline = (-0.5, -0.3)

#for i in range(len(alldata_ave_varcopes)):
#    tmp = deepcopy(alldata_ave_varcopes[i])
#    tmp.drop_channels(['RM'])
#    tmp.plot_joint(topomap_args = dict(outlines='head', contours=0),
#                        title = 'subject %d varcopes'%i)
    

gave_ave.drop_channels(['RM'])
gave_ave_t.drop_channels(['RM'])
gave_varcopes.drop_channels(['RM'])


#gave_ave_varcopes.data = np.sqrt(gave_ave_varcopes.data)
baseline = (-.3, -.1)
baseline = (-.5, -.3)
gave_ave.plot_joint(topomap_args = dict(outlines='head', contours=0),baseline=baseline,
                     title = 'grand average response to array - betas', timefreqs = timefreqs)

gave_ave_t.plot_joint(topomap_args = dict(outlines='head', contours=0),baseline=baseline,
                     title = 'grand average response to array - tstats', timefreqs = timefreqs)


gave_varcopes.plot_joint(topomap_args = dict(outlines='head', contours=0),baseline=None,
                         title = 'grand average response to array - varcopes', timefreqs = timefreqs)


gave_ave_tovert = deepcopy(gave_ave_t)
gave_ave_toverb = deepcopy(gave_ave)

#need to run a ttest over the t-values rather than grand average here
tmp = np.empty(shape = (len(alldata_ave_t), 62, 39, 375 ))
for i in range(len(subs)):
    tmp[i,:,:] = alldata_ave_t[i].data
tovert = sp.stats.ttest_1samp(tmp,popmean=0, axis = 0)
gave_ave_tovert.data = tovert[0]

gave_ave_tovert.plot_joint(topomap_args = dict(outlines='head', contours=0),baseline=baseline,
                           title = 'grand average response - t over tstats', timefreqs = timefreqs)

tmp = np.empty(shape = (len(alldata_ave), 62, 39, 375 ))
for i in range(len(subs)):
    tmp[i,:,:] = alldata_ave[i].data
tovert = sp.stats.ttest_1samp(tmp,popmean=0, axis = 0)
gave_ave_toverb.data = tovert[0]

gave_ave_toverb.plot_joint(topomap_args = dict(outlines='head', contours=0),baseline=baseline,
                           title = 'grand average response - t over betas', timefreqs = timefreqs)






#%%
baseline = (-0.5, -0.3)
timefreqs = {(-.15, 10):(.3, 4), #this is a prestimulus window that looks at the 300mis pre array (not in the baseline window)
             (.50, 10): (.5, 4),
             (.75, 10): (.5, 4)}

#alldata_ave[9].apply_baseline(baseline = baseline).plot_joint(timefreqs = timefreqs, topomap_args=dict(outlines='head', contours=0))


gave_ave.drop_channels(['RM'])
gave_error.drop_channels(['RM'])
gave_conf.drop_channels(['RM'])

gave_ave_t.drop_channels(['RM'])
gave_error_t.drop_channels(['RM'])
gave_conf_t.drop_channels(['RM'])

#plot average array evoked response
gave_ave.apply_baseline(baseline=baseline).plot_joint( topomap_args = dict(outlines='head', contours=0,
                                          vmin = np.min(gave_ave.data)/1, 
                                          vmax = np.multiply(np.min(gave_ave.data)/1, -1)),
                     title = 'grand average response to array - betas', timefreqs = timefreqs)

gave_error.apply_baseline(baseline=baseline).plot_joint( topomap_args = dict(outlines='skirt', contours=0,
                                          vmin = np.min(gave_error.data)/1, 
                                          vmax = np.multiply(np.min(gave_error.data)/1, -1)),
                     title = 'grand average of main effect of error - betas',
                     baseline = baseline, timefreqs = timefreqs)
        

gave_conf.apply_baseline(baseline=baseline).plot_joint( topomap_args = dict(outlines='skirt', contours=0,
                                          vmin = np.min(gave_conf.data)/5, 
                                          vmax = np.multiply(np.min(gave_conf.data)/5, -1)),
                     title = 'grand average of main effect of confidence width - betas',
                     baseline = baseline, timefreqs = timefreqs)




#plot average array evoked response
gave_ave_t.apply_baseline(baseline=baseline).plot_joint( topomap_args = dict(outlines='head', contours=0),
                     title = 'grand average response to array - tstats',
                     timefreqs = timefreqs)

gave_error_t.apply_baseline(baseline=baseline).plot_joint( topomap_args = dict(outlines='skirt', contours=0),
                     title = 'grand average of main effect of error - tstats',
                     baseline = baseline, timefreqs = timefreqs)
        

gave_conf_t.apply_baseline(baseline=baseline).plot_joint( topomap_args = dict(outlines='head', contours=0),
                     title = 'grand average of main effect of confidence width - tstats',
                     baseline = baseline, timefreqs = timefreqs)
#%% to do t-tests over tstats

#need to run a ttest over the t-values rather than grand average here
tmp = np.empty(shape = (len(alldata_ave_t), 62, 39, 375 ))
for i in range(len(subs)):
    tmp[i,:,:] = alldata_ave_t[i].data
tovert = sp.stats.ttest_1samp(tmp,popmean=0, axis = 0)
gave_ave_t.data = tovert[0]

gave_ave_t.apply_baseline(baseline=baseline).plot_joint( topomap_args = dict(outlines='head', contours=0),
                     title = 'grand average response - t over tstats', timefreqs = timefreqs)


gave_error_t.plot_joint( topomap_args = dict(outlines='head', contours=0,
                                             vmin = -3, vmax = +3),
                     title = 'grand average of main effect of error - t over tstats',
                     baseline = baseline, timefreqs = timefreqs)




tmp = np.empty(shape = (len(alldata_error_t), 62, 39, 375 ))
for i in range(len(subs)):
    tmp[i,:,:] = alldata_error_t[i].data
tovert = sp.stats.ttest_1samp(tmp,popmean=0, axis = 0)
gave_error_t.data = tovert[0]

gave_error.plot_joint( topomap_args = dict(outlines='head', contours=0),
                     title = 'grand average of main effect of error - t over betas',
                     baseline = baseline, timefreqs = timefreqs)


gave_error_t.plot_joint( topomap_args = dict(outlines='head', contours=0,
                                             vmin = -3, vmax = +3),
                     title = 'grand average of main effect of error - t over tstats',
                     baseline = baseline, timefreqs = timefreqs)


tmp = np.empty(shape = (len(alldata_conf), 62, 39, 375))
for i in range(len(subs)):
    tmp[i,:,:] = alldata_conf[i].data
tovert = sp.stats.ttest_1samp(tmp,popmean=0, axis = 0)
gave_conf.data = tovert[0]

gave_conf.plot_joint( topomap_args = dict(outlines='head', contours=0),
                     title = 'grand average of main effect of confidence width - t over betas',
                     baseline = baseline, timefreqs = timefreqs)

tmp = np.empty(shape = (len(alldata_conf_t), 62, 39, 375))
for i in range(len(subs)):
    tmp[i,:,:] = alldata_conf_t[i].data
tovert = sp.stats.ttest_1samp(tmp,popmean=0, axis = 0)
gave_conf_t.data = tovert[0]

gave_conf_t.plot_joint( topomap_args = dict(outlines='head', contours=0),
                     title = 'grand average of main effect of confidence width - t over tstats',
                     baseline = baseline, timefreqs = timefreqs)






