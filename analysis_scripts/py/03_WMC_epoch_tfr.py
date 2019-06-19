#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:05:48 2019

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

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1,2])
subind = 1 #get first subject

sub = dict(loc='workstation', id=subs[subind-1]) #get subject name for param extraction
param = get_subject_info_wmConfidence(sub)


#read raw data
raw = mne.io.read_raw_fif(fname = param['rawcleaned'], preload=True)


#epoching
#here it's important to specify a dictionary that assigns each trigger to its own integer value
#mne.events_from_annotations will assign each trigger to an ordered integer, so e.g. trig11 will be 2, but epoching 11 will include another trigger
#this solves it
event_id = {'1' : 1, '2':2,'11':11,'12':12,'13':13,'14':14,'21':21,'22':22,'23':23,'24':24,
            '31':31,'32':32,'33':33,'34':34,'41':41,'42':42,'43':43,'44':44,'51':51,'52':52,'53':53,'54':54,
            '61':61,'62':62,'63':63,'64':64,'71':71,'72':72,'73':73,'74':74,'76':76,'77':77,'78':78,'254':254,'255':255}

events_cue = {'neutral/probeleft'  : 11,
              'neutral/proberight' : 12,
              'cued/probeleft'     : 13,
              'cued/proberight'    : 14}
events,_ = mne.events_from_annotations(raw, event_id = event_id)
tmin, tmax = -0.5, 1.5
baseline = (None,0)


cuelocked_epochs = mne.Epochs(raw, events, events_cue, tmin, tmax, baseline, reject_by_annotation=True, preload=True)



bdata = pd.DataFrame.from_csv(path = param['behaviour'])
cuelocked_epochs.metadata = bdata

cuelocked_epochs = cuelocked_epochs['DTcheck ==0 and clickresp == 1'] #throw out trials based on behavioural data
#go through and check for bad epochs -- remove trls with excessive blinks and/or excessive noise on visual inspection
cuelocked_epochs.plot(scalings='auto', n_epochs=4, n_channels=len(cuelocked_epochs.info['ch_names'])) 
cuelocked_epochs.drop_bad()

cuelocked_epochs.set_eeg_reference(ref_channels=['RM'])

freqs = np.arange(1, 40, 1)  # frequencies from 2-35Hz
n_cycles = freqs *.3  # 300ms timewindow for estimation

# Run TF decomposition overall epochs
tfr = mne.time_frequency.tfr_morlet(cuelocked_epochs, freqs=freqs, n_cycles=n_cycles,
                     use_fft=True, return_itc=False, average=False)



tfr_cueleft = tfr['cuetrig == 13'].average()
tfr_cueright = tfr['cuetrig == 14'].average()

visright_picks = mne.pick_channels(tfr.ch_names, ['PO8', 'O2', 'PO4'])
visleft_picks  = mne.pick_channels(tfr.ch_names, ['PO7', 'O1', 'PO3'])

tfr_cueleft.plot_joint(baseline=(-0.5,-0.2),mode='mean')

tfr_cueleft.plot(baseline = (-0.5,-0.2),mode='mean', combine='mean',picks=visright_picks)

#get contra-ipsi and contra+ipsi

ldata = tfr_cueleft.data
rdata = tfr_cueright.data
lvsr  = np.subtract(ldata,rdata) 
lplsr = np.add(ldata,rdata)
lvsr_scaled = np.multiply(np.divide(lvsr,lplsr),100)

tf_lvsr_scaled = mne.time_frequency.AverageTFR(info=tfr_cueleft.info, data = lvsr_scaled, times=tfr_cueleft.times, freqs = tfr_cueleft.freqs, nave = tfr_cueleft.nave)


#plots mean of the left and right sensors (which obv makes no sense)
tf_lvsr_scaled.plot(baseline = (-0.5,-0.2),mode='mean', combine='mean', picks=np.ravel([visright_picks,visleft_picks]))

#plot average of the right visual electrodes (because cued left vs cued right)
tf_lvsr_scaled.plot_joint(baseline=(-0.5,-0.2),mode='mean',vmin=-25,vmax=25, cmap='RdBu', picks= visleft_picks)

#plot the topography of the alpha band (8-14Hz) in the timewindow of the alpha suppression seen
tf_lvsr_scaled.plot_topomap(tmin=0.6,tmax=1.0,fmin=8,fmax=14,baseline=(-0.5,-0.2), mode='mean', sensors=True,vmin=-25,vmax=25, cmap='RdBu', cbar_fmt=None)




















# cluster stats

tf_lvsr_scaled_vis = tf_lvsr_scaled.data[np.ravel([visleft_picks,visright_picks]),:,:]
tf_lvsr_scaled_vis_ave = np.nanmean(tf_lvsr_scaled_vis,0)

plt.figure()
plt.contourf(tf_lvsr_scaled_vis_ave, levels=64,cmap = 'RdBu_r')


# # # # # can't do cluster stats until multiple subjects

tobs1 ,c1, p1, H01 = mne.stats.permutation_cluster_1samp_test(tf_lvsr_scaled_vis_ave,out_type='mask')
tobs2 ,c2, p2, H02 = mne.stats.permutation_cluster_1samp_test(tf_lvsr_scaled_vis_ave, tail = -1, n_permutations=100, step_down_p=0.05,seed=1,buffer_size=None)


mne.stats


mask_1 = np.asarray(c1)[p1<0.01]

extent = [tf_lvsr_scaled.times[0], tf_lvsr_scaled.times[-1], tf_lvsr_scaled.freqs[0], tf_lvsr_scaled.freqs[-1]]

axes = plt.axes()
im = axes.imshow(tf_lvsr_scaled_vis_ave, cmap = 'RdBu_r', vmin = -3, vmax = 3, extent = extent, origin = 'lower', aspect='auto')
plt.colorbar(im)

big_mask = np.kron(np.squeeze(mask_1),np.ones((10,10)))





















tf_lvsr_scaled.plot(mask=mask_1, picks=visright_picks,baseline=(-0.5,-0.2), mode='mean')