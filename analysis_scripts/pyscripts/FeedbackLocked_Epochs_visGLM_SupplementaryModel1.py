# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:29:02 2021

@author: sammi
"""

import numpy as np
import scipy as sp
from scipy import signal
import pandas as pd
import mne
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy import stats

# sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
# from wmConfidence_funcs import get_subject_info_wmConfidence
# from wmConfidence_funcs import gesd, plot_AR, toverparam, smooth, runclustertest_epochs
sys.path.insert(0, 'C:\\Users\\sammi\\Desktop\\Experiments\\DPhil\\wmConfidence\\analysis_scripts')#because working from laptop to make this script
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, smooth, runclustertest_epochs

# wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd

os.chdir(wd)
figpath = op.join(wd,'figures', 'eeg_figs', 'fblocked', 'supp_glm3')


subs = np.array([4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22,24, 25, 26])

contrasts = ['correct', 'incorrect', 'errorcorrect', 'errorincorrect', 'confcorrect', 'confincorrect', 'pside',
             'incorrvscorr', 'errorincorrvscorr', 'confincorrvscorr',
             'grandmean', 'error', 'conf', 'neutral', 'cued', 'cuedvsneutral']

data = dict()
for i in contrasts:
    data[i] = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub) #_baselined
    
    param = {}
    param['path'] = 'C:/Users/sammi/Desktop/Experiments/DPhil/wmConfidence/data'
    param['subid'] = 's%02d'%(i)
    sub = dict(loc = 'windows', id = i)
    
    for name in contrasts:
        data[name].append(   mne.read_evokeds(fname = op.join(param['path'], 'glms', 'feedback', 'supp_glm3', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tl_' + name + '_betas-ave.fif'))[0])        

#%%
gave = mne.grand_average(data['grandmean']); times = gave.times;
fig = plt.figure(figsize = (9,9))
ax = fig.add_subplot(111)
gave.plot_sensors(show_names=True, axes = ax)
#fig.savefig(fname = op.join(figpath, 'sensor_locations.eps'), format = 'eps', dpi = 300)
#fig.savefig(fname = op.join(figpath, 'sensor_locations.pdf'), format = 'pdf', dpi = 300)
del(gave)
plt.close()


#%%

#firstly, just look at the scalp topographies (joint plot) for the evoked response in the relevant conditions
mne.grand_average(deepcopy(data['correct'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'correct trials')
mne.grand_average(deepcopy(data['incorrect'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'incorrect trials')
mne.grand_average(deepcopy(data['incorrvscorr'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'incorrect vs correct trials')
#%%
#first pass look at scalp topographies (joint plot) for neutral/cued trials and their difference
mne.grand_average(deepcopy(data['neutral'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'neutral trials')
mne.grand_average(deepcopy(data['cued'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'cued trials')
mne.grand_average(deepcopy(data['cuedvsneutral'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'cued vs neutral trials')
#%%

# now for channels FCz, Fz, Cz, we want to plot the ERPs in the different trial types

for channel in ['Fz', 'FCz', 'Cz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    neutral    = data['neutral'],
                    cued  = data['cued'],
                    difference = data['cuedvsneutral']),
            colors = dict(
                    neutral = '#636363',
                    cued = '#3182bd',
                    difference = '#000000'),
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'electrode: '+channel, truncate_xaxis = False)
    
    
#plot some posterior channels also incase there are posterior differences based on cueing
for channel in ['Pz', 'POz', 'Oz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    neutral    = data['neutral'],
                    cued  = data['cued'],
                    difference = data['cuedvsneutral']),
            colors = dict(
                    neutral = '#636363',
                    cued = '#3182bd',
                    difference = '#000000'),
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'electrode: '+channel, truncate_xaxis = False)    

# this is reassuring there arent differences in ERP shape based on cueing at either the frontal electrodes of interest
# or at posterior visual (central because central stimulus) electrodes

    
tmin, tmax = 0,1 #set time window for the clustering

#gonna store things in a dictionary so we dont need to constantly rerun
t_ern, clu_ern, clupv_ern, h0_ern = dict(), dict(), dict(), dict()
masks_ern = dict()
clutimes = deepcopy(data['grandmean'][0].crop(tmin = tmin, tmax = tmax).times)
for channel in ['FCz', 'Cz','Pz', 'POz', 'Oz']:
    t_ern[channel], clu_ern[channel], clupv_ern[channel], h0_ern[channel] = runclustertest_epochs(data = data,
                                                                                                  contrast_name = 'cuedvsneutral',
                                                                                                  channels = [channel],
                                                                                                  tmin = tmin, tmax = tmax,
                                                                                                  gauss_smoothing = None,
                                                                                                  out_type = 'indices', n_permutations = 5000)
    masks_ern[channel] = np.asarray(clu_ern[channel])[clupv_ern[channel] < 0.05]

#no clusters survive permutation testing at any electrode, for the difference between cued and neutral
#given this fact, having these regressors (neutral and cued categorical regressors) in the model
# only increases model complexity without better explaining the data in a meaningful way, so we can remove them
#%%
    
#first pass look at scalp topographies (joint plot) for neutral/cued trials and their difference
mne.grand_average(deepcopy(data['pside'])).plot_joint(times = np.arange(0.05, 0.7, 0.1), title = 'probed side (lvsr)')

#plot some posterior channels also incase there are posterior differences based on cueing
for channel in ['PO3', 'PO7', 'O1']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    pside    = data['pside']),
            colors = dict(
                    pside = '#636363'),
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'electrode: '+channel, truncate_xaxis = False)    

for channel in ['Fz', 'FCz', 'Cz']:
    mne.viz.plot_compare_evokeds(
            evokeds = dict(
                    pside    = data['pside']),
            colors = dict(
                    pside = '#636363'),
            legend = 'upper right', picks = channel, ci = .68, #show standard error of the ERP at the channel
            show_sensors = False, title = 'electrode: '+channel, truncate_xaxis = False)    


    
tmin, tmax = 0,1 #set time window for the clustering

#gonna store things in a dictionary so we dont need to constantly rerun
t_ern, clu_ern, clupv_ern, h0_ern = dict(), dict(), dict(), dict()
masks_ern = dict()
clutimes = deepcopy(data['grandmean'][0].crop(tmin = tmin, tmax = tmax).times)
for channel in ['FCz', 'Cz','PO3', 'PO7', 'O1']:
    t_ern[channel], clu_ern[channel], clupv_ern[channel], h0_ern[channel] = runclustertest_epochs(data = data,
                                                                                                  contrast_name = 'pside',
                                                                                                  channels = [channel],
                                                                                                  tmin = tmin, tmax = tmax,
                                                                                                  gauss_smoothing = None,
                                                                                                  out_type = 'indices', n_permutations = 5000)
    masks_ern[channel] = np.asarray(clu_ern[channel])[clupv_ern[channel] < 0.05]

#no clusters survive permutation testing at any electrode, for the difference between whether the probed item was on the left or right
#given this fact, having this regressor (left vs right probed) in the model
# only increases model complexity without better explaining the data in a meaningful way, so we can remove it
    
