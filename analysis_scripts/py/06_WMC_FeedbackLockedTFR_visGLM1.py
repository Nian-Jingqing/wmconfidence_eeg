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

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam, flip_tfrdata, runclustertest_tfr

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

figpath = op.join(wd, 'figures', 'eeg_figs', 'fblocked', 'tfrglm1')

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26]) #subject 2 actually only has 72 trials in total, not really a lot so exclude atm
nsubs = subs.size



contrasts = ['grandmean', 'pside', 'error', 'conf',
             'correct',      'incorrect',      'incorrvscorr',
             'errorcorrect', 'errorincorrect', 'errorincorrvscorr',
             'confcorrect',  'confincorrect',  'confincorrvscorr']

laplacian = False
if laplacian:
    lapstr = 'laplacian_'
else:
    lapstr = ''

data = dict()
data_baselined = dict()
data_t = dict()
data_baselined_t = dict()
for i in contrasts:
    data[i] = []
    data_baselined[i] = []
    data_t[i] = []
    data_baselined_t[i] = []

for i in subs:
    print('\n\ngetting subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    for name in contrasts:
        data[name].append(mne.time_frequency.read_tfrs(            fname = op.join(param['path'], 'glms', 'feedback','tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ lapstr + name + '_betas-tfr.h5'))[0])
        data_t[name].append(mne.time_frequency.read_tfrs(          fname = op.join(param['path'], 'glms', 'feedback','tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ lapstr + name + '_tstats-tfr.h5'))[0])
        data_baselined[name].append(mne.time_frequency.read_tfrs(  fname = op.join(param['path'], 'glms', 'feedback','tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ lapstr + name + '_betas_baselined-tfr.h5'))[0])
        data_baselined_t[name].append(mne.time_frequency.read_tfrs(fname = op.join(param['path'], 'glms', 'feedback','tfr_fullglm', 'wmConfidence_' + param['subid'] + '_feedbacklocked_tfr_'+ lapstr + name + '_tstats_baselined-tfr.h5'))[0])


#%%
alltimes = mne.grand_average(data['grandmean']).times
allfreqs = mne.grand_average(data['grandmean']).freqs

timefreqs_alpha = {(.4, 10):(.4, 4),
                   (.6, 10):(.4, 4),
                   (.8, 10):(.4, 4),
                   (1., 10):(.4, 4),
                   (1.2, 10):(.4, 4)}
timefreqs_theta = {(.4, 6):(.4, 4),
                   (.6, 6):(.4, 4),
                   (.8, 6):(.4, 4),
                   (1., 6):(.4, 4),
                   (1.2, 6):(.4, 4)}

frontal_chans = ['Fz', 'AFz', 'F1', 'F2', 'FCz']

#%%
#here we're going to go through and plot the conditions in separate cells, and do stats there too
stat = 'tstat'
baselined = True
contrast = 'grandmean'
plot_t = True

vmin = dict()
vmin['beta'] = -5e-10
vmin['tstat'] = -3

for contrast in ['grandmean', 'incorrect', 'correct', 'incorrvscorr']:
    if 'vs' in contrast:
        baselined = False
    
    if stat == 'beta' and not baselined:
        dat2use = deepcopy(data)
    elif stat == 'beta' and baselined:
        dat2use = deepcopy(data_baselined)
    elif stat == 'tstat' and not baselined:
        dat2use = deepcopy(data_t)
    elif stat == 'tstat' and baselined:
        dat2use = deepcopy(data_baselined_t)
    
    gave = mne.grand_average(dat2use[contrast])
    
    
    if plot_t:
        gave.data = toverparam(dat2use[contrast])
    
    gave.plot_joint(topomap_args = dict(contours =0, vmin = vmin['tstat'], vmax = vmin['tstat']*-1), title = contrast,
                    timefreqs = timefreqs_theta, vmin = vmin['tstat'], vmax = vmin['tstat']*-1)
#channels to focus on for theta: FCz, FC1, FC2, Cz, Fz


for channel in ['FCz','Cz', 'Fz']:
    gave.plot(picks = channel, vmin = -3, vmax = 3)


#get the time frequency for incorrect vs correct and do some cluster stats on it to see if the theta effect comes out as significant
    

contrast = 'incorrvscorr'
channel = ['FCz']
stat = 'tstat'
tmin = 0
tmax = 1
t_cope, clu_cope, clupv_cope, _ = runclustertest_tfr(data = dat2use,
                                                     contrast_name = contrast,
                                                     channels = ['FCz'],
                                                     contra_channels = None,
                                                     ipsi_channels   = None,
                                                     tmin = tmin, tmax = tmax, out_type = 'mask',
                                                     n_permutations = 5000)
masks_cope = np.asarray(clu_cope)[clupv_cope <= 0.05]
clutimes   = deepcopy(dat2use[contrast][0]).crop(tmin = tmin, tmax = tmax).times

plotdata = mne.grand_average(deepcopy(dat2use[contrast]));
plot_t = True
if plot_t:
    plotdata.data = toverparam(deepcopy(dat2use[contrast]))

tfdata = np.squeeze(plotdata.pick_channels(['FCz']).data)

fig = plt.figure(figsize = (8,4))
ax = fig.subplots(1)
tfplot = ax.imshow(tfdata,
                   cmap = 'RdBu_r', aspect = 'auto', vmin = -3, vmax = 3, interpolation = 'gaussian',
                   origin = 'lower', extent = (alltimes.min(), alltimes.max(), allfreqs.min(), allfreqs.max()))
ax.vlines([0, 0.5], ymin = allfreqs.min(), ymax = allfreqs.max(), linestyles = 'dashed', color = '#000000', lw = 3)
fig.colorbar(tfplot, ax = ax)
for mask in masks_cope:
    bigmask = np.kron(mask, np.ones((10,10)))
    ax.contour(bigmask, color = '#000000', linewidths = .75, antialiased = False,
               extent = (clutimes.min(), clutimes.max(), allfreqs.min(), allfreqs.max()))
ax.set_ylabel('Frequency (Hz)')
ax.set_xlabel('Time relative to feedback onset (s)')
ax.set_title('feedback induced response for incorrect - correct trials, stat = %s'%(stat))
plt.tight_layout()

#now see if we can get a jointplot out
jplot = mne.grand_average(dat2use[contrast]); jplot.data = toverparam(dat2use[contrast])
jplot.plot_joint(timefreqs = timefreqs_theta, topomap_args = dict(contours=0, vmin = -3, vmax = 3))


for mask in masks_cope:
    itmin = clutimes[np.where(mask==True)[1]].min()
    itmax = clutimes[np.where(mask==True)[1]].max()
    fig = mne.viz.plot_tfr_topomap(jplot, tmin = itmin, tmax = itmax, fmin = 8, fmax = 12,contours=0, unit='t',
                              vmin = -3, vmax = 3, res = 300,
                             colorbar = True, cmap = 'RdBu_r')






#%%
