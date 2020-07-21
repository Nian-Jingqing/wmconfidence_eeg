#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:14:22 2019

@author: sammirc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:30:29 2019

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
from copy import deepcopy
from scipy import stats
from scipy import ndimage


sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, runclustertest_epochs

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

figpath = op.join(wd, 'figures', 'eeg_figs', 'feedbacklocked', 'epochs_glm5')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm
nsubs = subs.size


contrasts = ['defcorrect', 'justcorrect', 'incorrect',
             'errdefcorrect', 'errjustcorrect', 'errincorrect',
             'confdefcorrect', 'confjustcorrect', 'confincorrect',
             'incorrvsdef', 'incorrvsjust', 'justvsdef',
             'errorincorrvsdef', 'errorincorrvsjust', 'errorjustvsdef',
             'confincorrvsdef', 'confincorrvsjust', 'confjustvsdef']

data = dict()
data_t = dict()
for i in contrasts:
    data[i] = []
    data_t[i] = []
for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    for name in contrasts:
        data[name].append(   mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm5', 'wmc_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_betas-ave.fif'))[0])
        data_t[name].append( mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm5', 'wmc_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_tstats-ave.fif'))[0])

#drop right mastoid from literally everything here lol its not useful anymore
for cope in data.keys():
    for i in range(subs.size):
        data[cope][i]   = data[cope][i].drop_channels(['RM'])#.set_eeg_reference(ref_channels='average')
        data_t[cope][i] = data_t[cope][i].drop_channels(['RM'])#.set_eeg_reference(ref_channels='average')
#%%
        
#gave_pside = mne.grand_average(data['pside']); #gave_pside.data = toverparam(data['pside'])
#gave_pside.plot_joint(title = 'probed side (left vs right)', picks = 'eeg', topomap_args = dict(outlines='head', contours = 0))
#gave_pside.plot(picks = ['P7', 'P8'], window_title = 'probed left vs right', spatial_colors=True)

#gave_pleft = mne.grand_average(data['probedleft']); #gave_pleft.data = toverparam(data['probedleft'])
#gave_pleft.plot_joint(title = 'probed left', picks = 'eeg', topomap_args = dict(outlines='head', contours = 0))
#gave_pleft.plot(picks = ['P7', 'P8'], spatial_colors = True, window_title = 'probed left')
#
#gave_pright = mne.grand_average(data['probedright']); #gave_pright.data = toverparam(data['probedright'])
#gave_pright.plot_joint(title = 'probed right', picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0))
#gave_pright.plot(picks = ['P7', 'P8'], spatial_colors = True, window_title = 'probed right')
#        
#gave_corr = mne.grand_average(data['corr'])
#gave_incorr = mne.grand_average(data['incorr'])
#gave_ivsc = mne.grand_average(data['incorrvscorr'])
#mne.viz.plot_sensors(data['incorrect'][0].info, show_names=True)

for channel in ['FCz', 'Cz', 'CPz', 'Pz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    defscorrect = data['defcorrect'],
                    justcorrect   = data['justcorrect'],
                    incorrect   = data['incorrect']),
            colors = dict(
                    defscorrect = '#fc8d59',
                    justcorrect   = '#91cf60',
                    incorrect   = '#91bfdb'
                    ),
                    ci = .68, show_legend = 'upper right',
                    picks = channel, show_sensors = False,#ylim = dict(eeg=[-4,4]),
                    truncate_xaxis = False)
#%%
for channel in ['FCz', 'Cz', 'CPz', 'Pz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    incorrvsdef = data['incorrvsdef'],
                    incorrvsjust   = data['incorrvsjust'],
                    justvsdef   = data['justvsdef']),
            colors = dict(
                    incorrvsdef = '#fc8d59',
                    incorrvsjust   = '#91cf60',
                    justvsdef   = '#91bfdb'
                    ),
                    ci = .68, show_legend = 'upper right',
                    picks = channel, show_sensors = False,#ylim = dict(eeg=[-4,4]),
                    truncate_xaxis = False)          
#%%
gave_incorrvsdef  = mne.grand_average(data['incorrvsdef'])
gave_incorrvsjust = mne.grand_average(data['incorrvsjust'])
gave_justvsdef    = mne.grand_average(data['justvsdef'])
alltimes = gave_incorrvsdef.times

gave_incorrvsdef.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0), title = 'incorrect vs definitely correct', times=np.arange(0.1, 0.6,0.05))       
gave_incorrvsjust.plot_joint(picks='eeg', topomap_args = dict(outlines='head', contours = 0), title = 'incorrect vs just correct', times=np.arange(0.1, 0.6,0.05))
gave_justvsdef.plot_joint(picks='eeg', topomap_args = dict(outlines='head', contours = 0), title = 'just correct vs definitely correct', times=np.arange(0.1, 0.6,0.05))            

#%%
#nonpara cluster t test to see where diff is significant
tmin, tmax = 0, 1
channel = 'FCz'
t_ivsd, clu_ivsd, clupv_ivsd, H0_ivsd = runclustertest_epochs(data = data,
                                                          channels = [channel],
                                                          contrast_name = 'incorrvsdef',
                                                          tmin = tmin, tmax = tmax,
                                                          gauss_smoothing = None, out_type = 'indices',
                                                          n_permutations = 'Default')
masks_ivsd = np.asarray(clu_ivsd)[clupv_ivsd <= 0.05]


t_ivsj, clu_ivsj, clupv_ivsj, H0_ivsj = runclustertest_epochs(data = data,
                                                          channels = [channel],
                                                          contrast_name = 'incorrvsjust',
                                                          tmin = tmin, tmax = tmax,
                                                          gauss_smoothing = None, out_type = 'indices',
                                                          n_permutations = 'Default')
masks_ivsj = np.asarray(clu_ivsj)[clupv_ivsj <= 0.05]

t_jvsd, clu_jvsd, clupv_jvsd, H0_jvsd = runclustertest_epochs(data = data,
                                                          channels = [channel],
                                                          contrast_name = 'justvsdef',
                                                          tmin = tmin, tmax = tmax,
                                                          gauss_smoothing = None, out_type = 'indices',
                                                          n_permutations = 'Default')
masks_jvsd = np.asarray(clu_jvsd)[clupv_jvsd <= 0.05]


fig = plt.figure()
ax = plt.axes()
ferntimes = deepcopy(gave_incorrvsdef).crop(tmin=tmin, tmax=tmax).times
mne.viz.plot_compare_evokeds(
            evokeds = dict(ivsd = data['incorrvsdef'], ivsj   = data['incorrvsjust'], jvsd   = data['justvsdef']),
            colors  = dict(ivsd = '#fc8d59',           ivsj   = '#91cf60',            jvsd   = '#91bfdb'),
                    ci = .68, show_legend = 'upper right',
                    picks = channel, show_sensors = False, truncate_xaxis = False, axes = ax)
ax.set_title('feedback evoked response at electrode ' + channel)
ax.hlines(y = 0, linestyles = 'dashed', color = '#000000', lw = .75, xmin = alltimes.min(), xmax = alltimes.max())
ax.vlines(x = 0, linestyles = 'dashed', color = '#000000', lw = .75, ymin = -3, ymax = 14)
ax.set_ylabel('t-value')
ax.set_xlabel('Time relative to feedback onset (s)')
ax.set_ylim([-5.6, None])

for mask in masks_ivsd:
    masktimes = mask[1]
    ax.hlines(y = -5,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#fc8d59', alpha = .5) #plot significant time points for difference effect
for mask in masks_ivsj:
    masktimes = mask[1]
    ax.hlines(y = -5.2,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#91cf60', alpha = .5) #plot significant time points for difference effect            
for mask in masks_jvsd:
    masktimes = mask[1]
    ax.hlines(y = -5.4,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#91bfdb', alpha = .5) #plot significant time points for difference effect

#%%

fig = gave_incorrvsdef.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0), title = 'incorrect vs definitely correct', times=np.arange(0.1, 0.6,0.05))       
ax = fig.axes[0]
for mask in masks_ivsd:
    masktimes = mask[1]
    ax.hlines(y = -5,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#fc8d59', alpha = .5) #plot significant time points for difference effect

fig = gave_incorrvsjust.plot_joint(picks='eeg', topomap_args = dict(outlines='head', contours = 0), title = 'incorrect vs just correct', times=np.arange(0.1, 0.6,0.05))
ax = fig.axes[0]
for mask in masks_ivsj:
    masktimes = mask[1]
    ax.hlines(y = -5,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#fc8d59', alpha = .5) #plot significant time points for difference effect


fig = gave_justvsdef.plot_joint(picks='eeg', topomap_args = dict(outlines='head', contours = 0), title = 'just correct vs definitely correct', times=np.arange(0.1, 0.6,0.05))            
ax = fig.axes[0]
for mask in masks_jvsd:
    masktimes = mask[1]
    ax.hlines(y = -3.5,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#fc8d59', alpha = .5) #plot significant time points for difference effect
       
             
#%%
gave_errdefcorr  = mne.grand_average(data['errdefcorrect'])
gave_errjustcorr = mne.grand_average(data['errjustcorrect'])
gave_errincorr   = mne.grand_average(data['errincorrect'])  
              

gave_errdefcorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'error ~ definitely correct',
                            times=np.arange(0.1, 0.6,0.05))       
              
gave_errjustcorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'error ~ just correct',
                            times=np.arange(0.1, 0.6,0.05))  

gave_errincorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'error ~ incorrect',
                            times=np.arange(0.1, 0.6,0.05))  

fig, axes = plt.subplots(4,1)
fig.suptitle('error regressor')
count = 0
for channel in ['FCz', 'Cz', 'CPz', 'Pz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    defscorrect = data['errdefcorrect'],
                    justcorrect   = data['errjustcorrect'],
                    incorrect   = data['errincorrect']),
            colors = dict(
                    defscorrect = '#fc8d59',
                    justcorrect   = '#91cf60',
                    incorrect   = '#91bfdb'
                    ),
                    ci = .68, show_legend = 'lower right',
                    picks = channel, show_sensors = False,#ylim = dict(eeg=[-4,4]),
                    truncate_xaxis = False,
                    axes = axes[count])
    count += 1
              
          
fig, axes = plt.subplots(4,1)
fig.suptitle('error regressor, differences between trial types')
count = 0
for channel in ['FCz', 'Cz', 'CPz', 'Pz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    ivsd = data['errorincorrvsdef'],
                    ivsj   = data['errorincorrvsjust'],
                    jvsd   = data['errorjustvsdef']),
            colors = dict(
                    ivsd = '#fc8d59',
                    ivsj   = '#91cf60',
                    jvsd   = '#91bfdb'
                    ),
                    ci = .68, show_legend = 'lower right',
                    picks = channel, show_sensors = False,#ylim = dict(eeg=[-4,4]),
                    truncate_xaxis = False,
                    axes = axes[count])
    count += 1    


gave_errivsd = mne.grand_average(data['errorincorrvsdef'])
gave_errivsd.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'error ~ incorrect vs definitely correct',
                            times=np.arange(0.1, 0.6,0.05))

gave_errivsj = mne.grand_average(data['errorincorrvsjust'])
gave_errivsj.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'error ~ incorrect vs just correct',
                            times=np.arange(0.1, 0.6,0.05))


gave_errjvsd = mne.grand_average(data['errorjustvsdef'])
gave_errjvsd.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'error ~ just vs definitely correct',
                            times=np.arange(0.1, 0.6,0.05))
#%%
#cluster stats on these regressor differences between conditions
tmin, tmax = 0, 0.75
channel = 'FCz'
t_errivsd, clu_errivsd, clupv_errivsd, _ = runclustertest_epochs(data = data,
                                                                 contrast_name = 'errorincorrvsdef',
                                                                 channels = [channel],
                                                                 tmin = tmin, tmax = tmax,
                                                                 gauss_smoothing = None, out_type = 'indices')
masks_errivsd = np.asarray(clu_errivsd)[clupv_errivsd <= 0.05]

t_errivsj, clu_errivsj, clupv_errivsj, _ = runclustertest_epochs(data = data,
                                                                 contrast_name = 'errorincorrvsjust',
                                                                 channels = [channel],
                                                                 tmin = tmin, tmax = tmax,
                                                                 gauss_smoothing = None, out_type = 'indices')
masks_errivsj = np.asarray(clu_errivsj)[clupv_errivsj <= 0.05]

t_errjvsd, clu_errjvsd, clupv_errjvsd, _ = runclustertest_epochs(data = data,
                                                                 contrast_name = 'errorjustvsdef',
                                                                 channels = [channel],
                                                                 tmin = tmin, tmax = tmax,
                                                                 gauss_smoothing = None, out_type = 'indices')
masks_errjvsd = np.asarray(clu_errjvsd)[clupv_errjvsd <= 0.05]


fig = plt.figure()
ax = plt.axes()
ferntimes = deepcopy(gave_incorrvsdef).crop(tmin=tmin, tmax=tmax).times
mne.viz.plot_compare_evokeds(
            evokeds = dict(ivsd = data['errorincorrvsdef'], ivsj   = data['errorincorrvsjust'], jvsd   = data['errorjustvsdef']),
            colors  = dict(ivsd = '#fc8d59',           ivsj   = '#91cf60',            jvsd   = '#91bfdb'),
                    ci = .68, show_legend = 'upper right', ylim = dict(eeg = [-10,10]),
                    picks = channel, show_sensors = False, truncate_xaxis = False, axes = ax)
ax.set_title('error regressor ~ feedback evoked response at electrode ' + channel)
ax.hlines(y = 0, linestyles = 'dashed', color = '#000000', lw = .75, xmin = alltimes.min(), xmax = alltimes.max())
ax.vlines(x = 0, linestyles = 'dashed', color = '#000000', lw = .75, ymin = -3, ymax = 14)
ax.set_ylabel('average beta value (AU)')
ax.set_xlabel('Time relative to feedback onset (s)')
ax.set_ylim([-10, 10])
#ax.set_xlim([-0.2, 0.75])

for mask in masks_errivsd:
    masktimes = mask[1]
    ax.hlines(y = -8,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#fc8d59', alpha = .5) #plot significant time points for difference effect
for mask in masks_errivsj:
    masktimes = mask[1]
    ax.hlines(y = -8.3,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#91cf60', alpha = .5) #plot significant time points for difference effect            
for mask in masks_errjvsd:
    masktimes = mask[1]
    ax.hlines(y = -8.6,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#91bfdb', alpha = .5) #plot significant time points for difference effect

#%%

fig = gave_errivsd.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'error ~ incorrect vs definitely correct',
                            times=np.arange(0.1, 0.6,0.05))
ax = fig.axes[0]
for mask in masks_errivsd:
    masktimes = mask[1]
    ax.hlines(y = -2.2,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#fc8d59', alpha = .5) #plot significant time points for difference effect


fig = gave_errivsj.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'error ~ incorrect vs just correct',
                            times=np.arange(0.1, 0.6,0.05))
ax = fig.axes[0]
for mask in masks_errivsj:
    masktimes = mask[1]
    ax.hlines(y = -6,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#fc8d59', alpha = .5) #plot significant time points for difference effect


fig = gave_errjvsd.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'error ~ just vs definitely correct',
                            times=np.arange(0.1, 0.6,0.05))             
ax = fig.axes[0]
for mask in masks_errjvsd:
    masktimes = mask[1]
    ax.hlines(y = -8,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#fc8d59', alpha = .5) #plot significant time points for difference effect


       

 
#%%
gave_confdefcorr  = mne.grand_average(data['confdefcorrect'])
gave_confjustcorr = mne.grand_average(data['confjustcorrect'])
gave_confincorr   = mne.grand_average(data['confincorrect'])  


gave_confdefcorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'confidence ~ definitely correct',
                            times=np.arange(0.1, 0.6,0.05))       
              
gave_confjustcorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'confidence ~ just correct',
                            times=np.arange(0.1, 0.6,0.05))  

gave_confincorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'confidence ~ incorrect',
                            times=np.arange(0.1, 0.6,0.05))  

fig, axes = plt.subplots(4,1)
fig.suptitle('confidence regressor')
count = 0
for channel in ['FCz', 'Cz', 'CPz', 'Pz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    defscorrect = data['confdefcorrect'],
                    justcorrect   = data['confjustcorrect'],
                    incorrect   = data['confincorrect']),
            colors = dict(
                    defscorrect = '#fc8d59',
                    justcorrect   = '#91cf60',
                    incorrect   = '#91bfdb'
                    ),
                    ci = .68, show_legend = 'upper right',
                    picks = channel, show_sensors = False,#ylim = dict(eeg=[-4,4]),
                    truncate_xaxis = False,
                    axes = axes[count])
    count+=1

dat2use = deepcopy(data_t)
tmin, tmax = 0, 0.75
channel = 'FCz'
t_confi, clu_confi, clupv_confi, _ = runclustertest_epochs(data = dat2use,
                                                        channels = [channel],
                                                        contrast_name = 'confincorrect',
                                                        tmin = tmin, tmax = tmax,
                                                        gauss_smoothing = None, out_type = 'indices',
                                                        n_permutations = 'Default')
masks_confi = np.asarray(clu_confi)[clupv_confi <= 0.05]    
    
t_confd, clu_confd, clupv_confd, _ = runclustertest_epochs(data     = dat2use,
                                                        channels = [channel],
                                                        contrast_name = 'confdefcorrect',
                                                        tmin = tmin, tmax = tmax,
                                                        gauss_smoothing = None, out_type = 'indices',
                                                        n_permutations = 'Default')
masks_confd = np.asarray(clu_confd)[clupv_confd <= 0.05]  

t_confj, clu_confj, clupv_confj, _ = runclustertest_epochs(data = dat2use,
                                                        channels = [channel],
                                                        contrast_name = 'confjustcorrect',
                                                        tmin = tmin, tmax = tmax,
                                                        gauss_smoothing = None, out_type = 'indices',
                                                        n_permutations = 'Default')
masks_confj = np.asarray(clu_confj)[clupv_confj <= 0.05]         
    
#really need to sort out this figure, stop relying on plot_compare_evokeds
fig = plt.figure()
ax = plt.axes()
mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    defscorrect   = dat2use['confdefcorrect'],
                    justcorrect   = dat2use['confjustcorrect'],
                    incorrect     = dat2use['confincorrect']),
            colors = dict(
                    defscorrect   = '#fc8d59',
                    justcorrect   = '#91cf60',
                    incorrect     = '#91bfdb'
                    ),
                    ci = .68, show_legend = 'upper right',
                    picks = channel, show_sensors = False,ylim = dict(eeg=[-2e6,2e6]),
                    truncate_xaxis = False, axes = ax)            
            
ax.set_title('confidence regressor ~ feedback evoked response at electrode '+channel)
ax.hlines(y = 0, linestyles = 'dashed', color = '#000000', lw = .75, xmin = alltimes.min(), xmax = alltimes.max())
ax.vlines(x = 0, linestyles = 'dashed', color = '#000000', lw = .75, ymin = -3, ymax = 14)
ax.set_ylabel('average tstat (AU)')
ax.set_xlabel('Time relative to feedback onset (s)')
#ax.set_ylim([-8, None])

for mask in masks_confd:
    masktimes = mask[1]
    ax.hlines(y = -15e5,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#fc8d59', alpha = .5) #plot significant time points for difference effect
for mask in masks_confj:
    masktimes = mask[1]
    ax.hlines(y = -15.2e5,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#91cf60', alpha = .5) #plot significant time points for difference effect            
for mask in masks_confi:
    masktimes = mask[1]
    ax.hlines(y = -15.4e5,
              xmin = ferntimes[masktimes].min(),
              xmax = ferntimes[masktimes].max(),
              lw = 5, color = '#91bfdb', alpha = .5) #plot significant time points for difference effect
             
#%%    
gave_confdefcorr  = mne.grand_average(data['confdefcorrect'])
gave_confjustcorr = mne.grand_average(data['confjustcorrect'])
gave_confincorr   = mne.grand_average(data['confincorrect'])  


gave_confdefcorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'confidence ~ definitely correct',
                            times=np.arange(0.1, 0.6,0.05))       
              
gave_confjustcorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'confidence ~ just correct',
                            times=np.arange(0.1, 0.6,0.05))  

gave_confincorr.plot_joint(picks = 'eeg', topomap_args = dict(outlines = 'head', contours = 0),
                            title = 'confidence ~ incorrect',
                            times=np.arange(0.1, 0.6,0.05))  

fig, axes = plt.subplots(4,1)
fig.suptitle('confidence regressor')
count = 0
for channel in ['FCz', 'Cz', 'CPz', 'Pz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    defscorrect = data['confdefcorrect'],
                    justcorrect   = data['confjustcorrect'],
                    incorrect   = data['confincorrect']),
            colors = dict(
                    defscorrect = '#fc8d59',
                    justcorrect   = '#91cf60',
                    incorrect   = '#91bfdb'
                    ),
                    ci = .68, show_legend = 'upper right',
                    picks = channel, show_sensors = False,#ylim = dict(eeg=[-4,4]),
                    truncate_xaxis = False,
                    axes = axes[count])
    count+=1
    
fig, axes = plt.subplots(4,1)
fig.suptitle('confidence regressor, differences between trial types')
count = 0
for channel in ['FCz', 'Cz', 'CPz', 'Pz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    ivsd = data['confincorrvsdef'],
                    ivsj   = data['confincorrvsjust'],
                    jvsd   = data['confjustvsdef']),
            colors = dict(
                    ivsd = '#fc8d59',
                    ivsj   = '#91cf60',
                    jvsd   = '#91bfdb'
                    ),
                    ci = .68, show_legend = 'lower right',
                    picks = channel, show_sensors = False,#ylim = dict(eeg=[-4,4]),
                    truncate_xaxis = False,
                    axes = axes[count])
    count += 1  
  
  