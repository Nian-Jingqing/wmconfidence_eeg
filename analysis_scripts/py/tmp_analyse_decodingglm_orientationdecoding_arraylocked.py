#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:02:47 2020

@author: sammirc
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

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, smooth, runclustertest_epochs

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

subs = np.array([4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22,24, 25, 26])

contrasts = ['grandmean', 'error']

srate=500; ntimes = srate*2



data   = np.zeros(shape = (len(subs), len(contrasts), ntimes)) * np.nan
data_t = np.zeros(shape = (len(subs), len(contrasts), ntimes)) * np.nan
   

for i in range(len(subs)):
    for contrast in range(len(contrasts)):
        isub = subs[i]
        icontrast = contrasts[contrast]
            
        data[i,contrast,:]   = np.load(op.join(wd, 'data/decoding/arraylocked/glm2', 's%02d_orientation_decodingglm_%s_beta.npy'%(isub, icontrast)))
        data_t[i,contrast,:] = np.load(op.join(wd, 'data/decoding/arraylocked/glm2', 's%02d_orientation_decodingglm_%s_tstat.npy'%(isub, icontrast)))

times = np.load(op.join(wd, 'data/decoding/arraylocked/glm2/decoding_data_times.npy'))



#%% look at the betas here
gmean = np.squeeze(data[:,0,:])
error = np.squeeze(data[:,1,:])


plotmean_gmean = np.mean(gmean,0)
plotsem_gmean  = sp.stats.sem(gmean, 0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times, plotmean_gmean, color = '#636363', label = 'grandmean decoding - ave betas')
ax.fill_between(times, plotmean_gmean-plotsem_gmean, plotmean_gmean+plotsem_gmean, alpha = .3, color = '#636363')
ax.hlines(1/12, xmin=times.min(), xmax = times.max(), linestyles = 'dashed', color = '#000000', lw = .5)
          
          
plotmean_error = np.mean(error,0)
plotsem_error  = sp.stats.sem(error,0)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times, plotmean_error, color = '#636363', label = 'error - ave betas')
ax.fill_between(times, plotmean_error-plotsem_error, plotmean_error+plotsem_error, alpha = .3, color = '#636363')
ax.hlines(0, xmin=times.min(), xmax = times.max(), linestyles = 'dashed', color = '#000000', lw = .5)
          
          
#%% look at tstats here

gmean_t = np.squeeze(data_t[:,0,:])
error_t = np.squeeze(data_t[:,1,:])
confidence_t = np.squeeze(data_t[:,2,:])


plotmean_gmean_t = np.mean(gmean_t,0)
plotsem_gmean_t  = sp.stats.sem(gmean_t, 0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times, plotmean_gmean_t, color = '#636363', label = 'grandmean decoding - ave tstat')
ax.fill_between(times, plotmean_gmean_t-plotsem_gmean_t, plotmean_gmean_t+plotsem_gmean_t, alpha = .3, color = '#636363')
ax.hlines(0, xmin=times.min(), xmax = times.max(), linestyles = 'dashed', color = '#000000', lw = .5)
         
          
plotmean_confidence_t = np.mean(confidence_t,0)
plotsem_confidence_t  = sp.stats.sem(confidence_t,0)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times, plotmean_confidence_t, color = '#636363', label = 'confidence - ave tstat')
ax.fill_between(times, plotmean_confidence_t-plotsem_confidence_t, plotmean_confidence_t+plotsem_confidence_t, alpha = .3, color = '#636363')
ax.hlines(0, xmin=times.min(), xmax = times.max(), linestyles = 'dashed', color = '#000000', lw = .5)
          
          
plotmean_error_t = np.mean(error_t,0)
plotsem_error_t  = sp.stats.sem(error_t,0)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(times, plotmean_error_t, color = '#636363', label = 'error - ave tstat')
ax.fill_between(times, plotmean_error_t-plotsem_error_t, plotmean_error_t+plotsem_error_t, alpha = .3, color = '#636363')
ax.hlines(0, xmin=times.min(), xmax = times.max(), linestyles = 'dashed', color = '#000000', lw = .5)      
          
          
          
          
          
          
          
          

