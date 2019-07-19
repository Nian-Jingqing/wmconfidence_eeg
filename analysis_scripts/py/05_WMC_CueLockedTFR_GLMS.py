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


subs = np.array([1,2,4,5,6,7])

alldata_left = []
alldata_right = []
alldata_lvsr = []

count=1
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    #get tfr
    
    tfr = mne.time_frequency.read_tfrs(fname=param['cuelocked_tfr']); tfr=tfr[0]
    #tfr.metadata = pd.DataFrame.from_csv(path=param['cuelocked_tfr_meta']) #read in and attach metadata
    
    #separate into cued left and right    
    tf_cleft  = tfr['cuetrig==13'].drop_channels(['RM'])
    tf_cright = tfr['cuetrig==14'].drop_channels(['RM'])
    po8 = mne.pick_channels(tf_cleft.ch_names, ['PO8'])

    glmdata = glm.data.TrialGLMData(data = tfr.data, time_dim=3, sample_rate=250)
    
    regressors = list()
    regressors.append( glm.regressors.CategoricalRegressor(category_list=tfr.metadata.cuetrig.to_numpy(), codes = 13, name = 'cued left') )
    regressors.append( glm.regressors.CategoricalRegressor(category_list=tfr.metadata.cuetrig.to_numpy(), codes = 14, name = 'cued right') )

    cues = tfr.metadata.cue.to_numpy()
    cues=np.where(tfr.metadata.cuetrig==14, -1, cues)
    absrdifcues=np.multiply(tfr.metadata.DT.to_numpy(), cues)
    regressors.append( glm.regressors.ParametricRegressor(name='DT x cuedside', values=absrdifcues, preproc='z', num_observations=glmdata.num_observations))    


    contrasts = list()
    contrasts.append( glm.design.Contrast([1,  0, 0], 'cued left') )
    contrasts.append( glm.design.Contrast([0,  1, 0], 'cued right') )
    contrasts.append( glm.design.Contrast([1, -1, 0], 'left vs right') )
    contrasts.append( glm.design.Contrast([0,  0, 1], 'DT x cuedside'))
    
    
    glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
    #glmdes.plot_summary()
    
    model = glm.fit.OLSModel( glmdes, glmdata)

#    plt.subplot(2,3,1)
#    plt.title('tfr of PO8 cued left trials, betas not power')
#    plt.contourf(tf_cleft.times, tf_cleft.freqs, np.squeeze(model.copes[0,po8,:,:]), cmap = 'RdBu_r', levels=64)
#    plt.colorbar()
#    
#    plt.subplot(2,3,2)
#    plt.title('tfr of PO8 cued right trials, betas not power')
#    plt.contourf(tf_cright.times, tf_cright.freqs, np.squeeze(model.copes[1,po8,:,:]), cmap='RdBu_r', levels=64)
#    plt.colorbar()
#    
#    plt.subplot(2,3,3)
#    plt.title('tfr of PO8 lvsr, betas not power')
#    plt.contourf(tfr.times, tfr.freqs, np.squeeze(model.copes[2,po8,:,:]), cmap='RdBu_r', levels=64)
#    plt.colorbar()
#    
#    plt.subplot(2,3,4)
#    plt.title('tfr of PO8 cued left trials, tstats not power')
#    plt.contourf(tf_cleft.times, tf_cleft.freqs, np.squeeze(model.get_tstats()[0,po8,:,:]), cmap = 'RdBu_r', levels=64)
#    plt.colorbar()
#    
#    plt.subplot(2,3,5)
#    plt.title('tfr of PO8 cued right trials, tstats not power')
#    plt.contourf(tf_cright.times, tf_cright.freqs, np.squeeze(model.get_tstats()[1,po8,:,:]), cmap='RdBu_r', levels=64)
#    plt.colorbar()
#    
#    plt.subplot(2,3,6)
#    plt.title('tfr of PO8 lvsr, tstats not power')
#    plt.contourf(tfr.times, tfr.freqs, np.squeeze(model.get_tstats()[2,po8,:,:]), cmap='RdBu_r', levels=64)
#    plt.colorbar()
#    plt.suptitle('subject '+str(i))
    
    plt.subplot(2,3,count)
    plt.contourf(tfr.times, tfr.freqs, np.squeeze(model.copes[3,po8,:,:]), cmap='RdBu_r', levels=64)
    plt.title('tfr of PO8, interaction between cued side and absrdif - subject'+str(i))
    plt.colorbar()
    count+=1

    tfr_betas_cleft = mne.time_frequency.AverageTFR(info  = tfr.info,
                                                    data  = np.squeeze(model.copes[[0,:,:,:]),
                                                    times = tf_cleft.times,
                                                    freqs = tf_cleft.freqs,
                                                    nave  = tf_cleft.average().nave 
                                                    )
    tfr_betas_cright = mne.time_frequency.AverageTFR(info=tfr.info,
                                                     data=np.squeeze(model.copes[[1,:,:,:]),
                                                     times=tf_cright.times,
                                                     freqs=tf_cright.freqs,
                                                     nave=tf_cright.average().nave
                                                     )
    tfr_betas_lvsr = mne.time_frequency.AverageTFR(info=tfr.info,
                                                   data=np.squeeze(model.copes[[2,:,:,:]),
                                                   times=tfr.times,
                                                   freqs=tfr.freqs,
                                                   nave=tfr.average().nave
                                                   )
    
    tfr_betas_absrdif = mne.time_frequency.AverageTFR(info=tfr.info,
                                                   data=np.squeeze(model.copes[3,:,:,:]),
                                                   times=tfr.times,
                                                   freqs=tfr.freqs,
                                                   nave=tfr.average().nave
                                                   )
    
    
    visright= mne.pick_channels(tfr_betas_lvsr.ch_names, ['PO8', 'O2', 'PO4', 'P4', 'P6', 'P8'])
    
    baseline=(-0.3,0); baseline=(None,None)
    tfr_betas_cleft.plot_joint(baseline=baseline,timefreqs = {(.75, 10):(.5, 4)}, title='cued left')
    tfr_betas_cright.plot_joint(baseline=baseline,timefreqs= {(.75, 10):(.5, 4)}, title='cued right')
    tfr_betas_lvsr.plot(picks=visright,combine='mean', cmap='RdBu_r')
    tfr_betas_lvsr.plot_joint(timefreqs= {(.75, 10):(.5, 4)}, title='lvsr',baseline=baseline)#, picks=visright, combine='mean')
    
    tfr_betas_absrdif.plot_joint(title='absrdif', cmap='RdBu_r')
    
    

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