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
#    if tfr.info['sfreq'] != 250:
#        tfr.resample(250) #resample to 250Hz so all are in the same sample rate
    
    tfr.metadata = pd.DataFrame.from_csv(path=param['cuelocked_tfr_meta'], index_col=None) #read in and attach metadata
    
    #separate into cued left and right    
    #tf_cleft  = tfr['cuetrig==13'].drop_channels(['RM'])
    #tf_cright = tfr['cuetrig==14'].drop_channels(['RM'])
    #po8 = mne.pick_channels(tf_cleft.ch_names, ['PO8'])

    glmdata = glm.data.TrialGLMData(data = tfr.data, time_dim=3, sample_rate=250)
    
    regressors = list()
    regressors.append( glm.regressors.CategoricalRegressor(category_list=tfr.metadata.cuetrig.to_numpy(), codes = 13, name = 'cued left') )
    regressors.append( glm.regressors.CategoricalRegressor(category_list=tfr.metadata.cuetrig.to_numpy(), codes = 14, name = 'cued right') )
    regressors.append( glm.regressors.ParametricRegressor(name = 'DT', values = tfr.metadata.DT.to_numpy(), preproc = 'z', num_observations = glmdata.num_observations))

    cues = tfr.metadata.cue.to_numpy()
    cues = np.where(tfr.metadata.cuetrig==14, -1, cues)
    DTcues = np.multiply(tfr.metadata.DT.to_numpy(), cues)
    regressors.append( glm.regressors.ParametricRegressor(name = 'DT x cuedside', values = DTcues, preproc = 'z', num_observations = glmdata.num_observations))    


    contrasts = list()
    contrasts.append( glm.design.Contrast([1,  0, 0, 0], 'cued left') )
    contrasts.append( glm.design.Contrast([0,  1, 0, 0], 'cued right') )
    contrasts.append( glm.design.Contrast([0,  0, 1, 0], 'DT') )
    contrasts.append( glm.design.Contrast([1, -1, 0, 0], 'left vs right') )
    contrasts.append( glm.design.Contrast([0,  0, 0, 1], 'DT x cuedside'))
    
    
    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
    #glmdes.plot_summary()
    
    model = glm.fit.OLSModel( glmdes, glmdata)

    tfr_betas_cleft = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                    data  = np.squeeze(model.copes[0,:,:,:]),
                                                    times = tfr.times,
                                                    freqs = tfr.freqs,
                                                    nave  = tfr['cuetrig==13'].average().nave 
                                                    )
    tfr_betas_cleft.save(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_cleft_betas-tfr.h5'), overwrite = True)
            
    del(globals()['tfr_betas_cleft'])


    tfr_betas_cright = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                     data  = np.squeeze(model.copes[1,:,:,:]),
                                                     times = tfr.times,
                                                     freqs = tfr.freqs,
                                                     nave  = tfr['cuetrig==14'].average().nave
                                                     )
    tfr_betas_cright.save(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_cright_betas-tfr.h5'), overwrite = True)

    del(globals()['tfr_betas_cright'])
    
    tfr_betas_DT    = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                    data  = np.squeeze(model.copes[2, :, :, :]),
                                                    times = tfr.times,
                                                    freqs = tfr.freqs,
                                                    nave  = tfr.average().nave
                                                    )
    tfr_betas_DT.save(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_DT_betas-tfr.h5'), overwrite = True)
    del(globals()['tfr_betas_DT'])
    
    
    tfr_betas_lvsr = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                   data  = np.squeeze(model.copes[3,:,:,:]),
                                                   times = tfr.times,
                                                   freqs = tfr.freqs,
                                                   nave  = tfr.average().nave
                                                   )
    tfr_betas_lvsr.save(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_lvsr_betas-tfr.h5'), overwrite = True)

    del(globals()['tfr_betas_lvsr'])
    
    tfr_betas_DTxCue = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                      data  = np.squeeze(model.copes[4,:,:,:]),
                                                      times = tfr.times,
                                                      freqs = tfr.freqs,
                                                      nave  = tfr.average().nave
                                                      )
    tfr_betas_DTxCue.save(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_DTxCue_betas-tfr.h5'), overwrite = True)

    del(globals()['tfr_betas_DTxCue'])
    
    del(tfr)
    del(glmdata)
    del(glmdes)
    del(model)

#%%   if the betas are already saved ...

alldata_lvsr = []
alldata_DTxCue = []
alldata_DT = []
for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)

    lvsr = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_lvsr_betas-tfr.h5')); lvsr = lvsr[0]
    alldata_lvsr.append(lvsr)
    DTxCue = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_DTxCue_betas-tfr.h5')); DTxCue = DTxCue[0]
    alldata_DTxCue.append(DTxCue)
    DT = mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_tfr_DT_betas-tfr.h5')); DT = DT[0]
    alldata_DT.append(DT)

gave_lvsr   = mne.grand_average(alldata_lvsr)
gave_DTxCue = mne.grand_average(alldata_DTxCue)
gave_DT     = mne.grand_average(alldata_DT)


baseline = (-0.5, -0.3)

gave_lvsr.plot_joint(topomap_args = dict(outlines = 'head', contours=0),
                     title = 'cued left vs right grand average betas',
                     baseline=baseline,
                     timefreqs = {
                             (.50, 10): (.5, 4),
                             (.75, 10): (.5, 4),
                             (1.0, 10): (.5, 4),
                             (1.25, 10):(.5, 4)
                             })

gave_DTxCue.plot_joint(topomap_args = dict(outlines='head', contours=0),
                       title = 'grand average interaction between cue and reaction time',
                       baseline = baseline,
                       timefreqs = {
                             (.50, 10): (.5, 4),
                             (.75, 10): (.5, 4),
                             (1.0, 10): (.5, 4),
                             (1.25, 10):(.5, 4)
                             })
    
gave_DT.plot_joint(topomap_args = dict(outlines='head', contours=0),
                       title = 'grand average main effect of decision time',
                       baseline = baseline,
                       timefreqs = {
                             (.50, 10): (.5, 4),
                             (.75, 10): (.5, 4),
                             (1.0, 10): (.5, 4),
                             (1.25, 10):(.5, 4)
                             })



#%%

gave_lvsr.plot_joint(topomap_args = dict(outlines = 'head', contours=0),
                     title = 'cued left vs right grand average betas',
                     baseline=baseline,
                     timefreqs = {
                             (.50, 10): (.5, 4),
                             (.75, 10): (.5, 4),
                             (1.0, 10): (.5, 4),
                             (1.25, 10):(.5, 4)
                             })



#now have some structures that have all current subjects so we can do some grand averaging
neut_gave = mne.grand_average(neutral)
cued_gave = mne.grand_average(cued)
cvsn_gave = mne.grand_average(cuedvsneutral)
cdif_int_gave = mne.grand_average(confdiffint)

fig = cvsn_gave.plot_joint(topomap_args = dict(outlines = 'head'),
                     ts_args = dict(hline = [0]),
                     title = 'cued vs neutral grand average')
axes = fig.get_axes()
axes[0].axvline(x=0  , ymin = -3, ymax = 3, linestyle = '--', linewidth = .5, color = 'k')
axes[0].axvline(x=0.5, ymin = -3, ymax = 3, linestyle = '--', linewidth = .5, color = 'k')


fig = cdif_int_gave.plot_joint(topomap_args = dict(outlines = 'head'),
                     ts_args = dict(hline = [0]),
                     title = 'confdiff x cue interaction grand average')
axes = fig.get_axes()
axes[0].axvline(x=0  , ymin = -3, ymax = 3, linestyle = '--', linewidth = .5, color = 'k')
axes[0].axvline(x=0.5, ymin = -3, ymax = 3, linestyle = '--', linewidth = .5, color = 'k')

    


























#%%
model = glm.fit.OLSModel( glmdes, glmdata )

#i'm pretty sure that here the middle plot is plotting the beta coefficients across the length of the time series to show where the effect starts to emerge (which is awesome)
plt.figure()
plt.subplot(311)
plt.plot(timerange, model.copes.T,lw=1) 
plt.legend(glmdes.contrast_names, loc = 'upper left', ncol=2)
plt.title('COPEs')
plt.subplot(312)
plt.plot(timerange, model.get_tstats().T, lw=1)
plt.axhline(y=0, ls='dashed', color='k', lw=1)
plt.axhline(y=2.58, ls='dashed', color = '#bdbdbd') #2.58 tstat line for significance
plt.axhline(y=-2.58, ls='dashed', color = '#bdbdbd') #2.58 tstat line for significance
plt.legend(glmdes.contrast_names, loc = 'upper left', ncol=2)
plt.title('t-stats')
plt.subplot(313)
plt.plot(timerange, np.nanmean(cued_fb_epoch['ave_p'],0),    label = 'cued'   , lw = 1, color = '#3182bd') #blue for cued
plt.plot(timerange, np.nanmean(neutral_fb_epoch['ave_p'],0), label = 'neutral', lw = 1, color = '#bdbdbd') #grey for neutral
plt.axvline(x = 0.0, lw = 1, ls = 'dashed', color = 'k')
plt.axvline(x = 0.5, lw = 1, ls = 'dashed', color = 'k', label = 'feedback offset')
plt.axvline(x = 1.5, lw = 1, ls = 'dashed', color = 'r', label = 'minimum onset of next trial')
plt.legend()
plt.title('pupil dilation rel. to feedback onset')
plt.show()


plt.figure()
plt.plot(timerange,model.fstats.T)
plt.legend(glmdes.ftest_names)
plt.title('FTests')