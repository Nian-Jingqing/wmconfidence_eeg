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

# sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri/analysis_scripts')
# sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
sys.path.insert(0, 'C:\\Users\\sammi\\Desktop\\Experiments\\DPhil\\wmConfidence\\analysis_scripts')#because working from laptop to make this script

from wmconfidence_funcs import get_subject_info_wmConfidence
from wmconfidence_funcs import gesd, plot_AR, toverparam, smooth, runclustertest_epochs

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
wd = 'C:/Users/sammi/Desktop/Experiments/DPhil/wmConfidence' # windows laptop wd
os.chdir(wd)



glmnum = 1
subs = np.array([4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22,24, 25, 26])

contrasts = ['grandmean', 'neutral', 'cued',
             'neutralleft', 'cuedleft', 'neutralright', 'cuedright',
             'cuedvsneutral', 'clvsn', 'crvsn', 'clvsr', 'nlvsr']


figpath = op.join(wd,'figures', 'eeg_figs', 'cuelocked', 'epochs_glm'+str(glmnum))

laplacian = False
if laplacian:
    lapstr = 'laplacian_'
else:
    lapstr = ''


data = dict()
data_t = dict()
for i in contrasts:
    data[i] = []
    data_t[i] = []

for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    
    param = {}
    param['path'] = 'C:/Users/sammi/Desktop/Experiments/DPhil/wmConfidence/data'
    param['subid'] = 's%02d'%(i)
    sub = dict(loc = 'windows', id = i)

    for name in contrasts:
        data[name].append(   mne.read_evokeds(fname = op.join(param['path'], 'glms', 'cuelocked', 'epochs_glm'+str(glmnum), 'wmc_' + param['subid'] + '_cuelocked_tl_' + lapstr + name + '_betas-ave.fif'))[0])        
        data_t[name].append( mne.read_evokeds(fname = op.join(param['path'], 'glms', 'cuelocked', 'epochs_glm'+str(glmnum), 'wmc_' + param['subid'] + '_cuelocked_tl_' + lapstr + name + '_tstats-ave.fif'))[0])        

# crop to view just cue period
for i in range(subs.size):
 for name in contrasts:
     data[name][i].crop(tmin = -0.3, tmax = 1.25)
     data_t[name][i].crop(tmin = -0.3, tmax = 1.25)     
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
##note here, the actual delay between offset of the cue and onset of the probe is ONE SECOND, NOT 1.5 SECONDS.
#I think that the script on the eeglab is different to in my local data, because it says 1.5s there but the saved behavioural data has delay2 as 1

mne.grand_average(deepcopy(data['grandmean'])).plot_joint(times = np.arange(0.05, 0.7, 0.05), title = 'grand mean ERP')
mne.grand_average(deepcopy(data['neutralleft'])).plot_joint(times = np.arange(0.05, 0.7, 0.05), title = 'neutral left')
mne.grand_average(deepcopy(data['cuedleft'])).plot_joint(times = np.arange(0.05, 0.7, 0.05), title = 'cued right')
mne.grand_average(deepcopy(data['neutralright'])).plot_joint(times = np.arange(0.05, 0.7, 0.05), title = 'neutral right')
mne.grand_average(deepcopy(data['cuedright'])).plot_joint(times = np.arange(0.05, 0.7, 0.05), title = 'cued right')
#%%


#look at the jointplots for some of the contrasts of interest
for contrast in ['cuedleft', 'cuedright', 'clvsr']:
    mne.grand_average(deepcopy(data[contrast])).plot_joint(times = np.arange(0.05, 0.7, 0.05),
                                                          topomap_args = dict(contours=0),
                                                          title = contrast)
#%%


#for cued left vs right (e.g. normal left - right subtraction in classic ERP literature)
#some time periods are associated with different components
#plot topographies for these time points
#this is all initially based off nick's paper: "Temporal Dynamics of Attention during Encoding versus Maintenance of Working Memory: Complementary Views from Event-related Potentials and Alpha-band Oscillations" -- Myers et al. (2015)

clvsr  = mne.grand_average(deepcopy(data)['clvsr'])
cleft  = mne.grand_average(deepcopy(data)['cuedleft'])
cright = mne.grand_average(deepcopy(data)['cuedright']) 
clvsr.plot_joint()


#rhs channels are PO8 and O2
#lhs channels are PO7 and O1


clvsr_po7 = np.squeeze(deepcopy(clvsr).pick_channels(['PO7']).data)
clvsr_po8 = np.squeeze(deepcopy(clvsr).pick_channels(['PO8']).data)



# N1   -- 150-180ms (mid = 165ms) post cue (PO7/8)

clvsr.plot_topomap(times   = .165, # plot this time point
                  average = .030,  #average over this time width around it (half this value either side)
                  contours = 0, res = 300)

#get the single subject data for the channels we're looking at
smooth_singlesubs = True
allpo7 = np.empty(shape = (subs.size, times.size))
allpo8 = np.empty(shape = (subs.size, times.size))
for i in range(subs.size):
    tmp_po7 = deepcopy(data)['clvsr'][i].pick_channels(['PO7']).data
    tmp_po8 = deepcopy(data)['clvsr'][i].pick_channels(['PO8']).data
    
    allpo7[i,:] = np.squeeze(tmp_po7)
    allpo8[i,:] = np.squeeze(tmp_po8)

for smoothing in [None, 3, 5, 15]:
    
    if smooth_singlesubs and smoothing != None:
        # smoothing = 3
        allpo7_smoothed = sp.ndimage.gaussian_filter1d(deepcopy(allpo7), sigma = smoothing)
        allpo8_smoothed = sp.ndimage.gaussian_filter1d(deepcopy(allpo8), sigma = smoothing)
    elif smooth_singlesubs and smoothing == None:
        allpo7_smoothed = deepcopy(allpo7)
        allpo8_smoothed = deepcopy(allpo8)

    
    plot_po7_mean = np.nanmean(allpo7_smoothed, axis = 0)
    plot_po7_sem  = sp.stats.sem(allpo7_smoothed, axis = 0)
    
    plot_po8_mean = np.nanmean(allpo8_smoothed, axis = 0)
    plot_po8_sem  = sp.stats.sem(allpo8_smoothed, axis = 0)
    
    diff_mean = np.nanmean(np.subtract(allpo7_smoothed, allpo8_smoothed), axis = 0)
    diff_sem  = sp.stats.sem(np.subtract(allpo7_smoothed, allpo8_smoothed), axis = 0)
    
    #66c2a5 -- green
    #fc8d62 -- orange
    #8da0cb -- blue
    
    fig = plt.figure(figsize = (12, 8))
    fig.suptitle('smoothing = %s'%str(smoothing))
    ax = fig.subplots(1)
    ax.plot(times, plot_po7_mean, lw = 1.5, color = '#66c2a5', label = 'PO7')
    ax.plot(times, plot_po8_mean, lw = 1.5, color = '#fc8d62', label = 'PO8')
    ax.fill_between(times, plot_po7_mean-plot_po7_sem, plot_po7_mean+plot_po7_sem, color = '#66c2a5', alpha = .2)
    ax.fill_between(times, plot_po8_mean-plot_po8_sem, plot_po8_mean+plot_po8_sem, color = '#fc8d62', alpha = .2)
    ax.plot(times, diff_mean, lw = 1.5, color = '#bdbdbd', label = 'difference')
    ax.fill_between(times, diff_mean-diff_sem, diff_mean+diff_sem, color = '#bdbdbd', alpha = .2)
    ax.hlines(y = 0, xmin = times.min(), xmax = times.max(), linestyles = 'dashed', colors = '#000000')
    ax.vlines(x = 0, ymin = -10e-7, ymax = 14e-7, linestyles = 'dashed', colors = '#000000')
    ax.legend()
    ax.set_ylim([-10e-7, 15e-7])
    
    
    
# N2pc -- 240-300ms  (mid = 270) post cue (PO7/8)

clvsr.plot_topomap(times   = .270, # plot this time point
                  average  = .060,  #average over this time width around it (half this value either side)
                  contours = 0, res = 300)




# EDAN -- 250-350ms  (mid = 300) post cue (PO7/8 & O1/2)

clvsr.plot_topomap(times   = .300, # plot this time point
                  average  = .100,  #average over this time width around it (half this value either side)
                  contours = 0, res = 300)

# ADAN -- 300-500ms  (mid = 400) post cue (FC3/4 & C3/C4)
clvsr.plot_topomap(times   = .400, # plot this time point
                  average  = .200,  #average over this time width around it (half this value either side)
                  contours = 0, res = 300)


# LDAP -- 750-1000ms (mid = 875) post cue (PO7/8 & O1/2)

clvsr.plot_topomap(times   = .875, # plot this time point
                  average  = .250,  #average over this time width around it (half this value either side)
                  contours = 0, res = 300)

# 
# 





