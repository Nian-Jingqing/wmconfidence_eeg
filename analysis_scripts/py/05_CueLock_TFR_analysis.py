#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 14:23:45 2019

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

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1,2,4,5,6,7])

alldata_left = []
alldata_right = []
alldata_lvsr = []

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
    #tf_cleft.average().plot_joint(baseline=(-0.3,-0.1), title='cued left trials', timefreqs = {(.75,10):(.5,4)})
    #tf_cright.average().plot_joint(baseline=(-0.3,-0.1), title='cued right trials', timefreqs = {(.75,10):(.5,4)})
    
    #tf_cleft.average().plot(baseline=(-0.3,-0.1), picks=po8, cmap='RdBu_r', title='PO8 cleft ave sub'+str(i))
    #tf_cright.average().plot(baseline=(-0.3,-0.1),picks=po8, cmap='RdBu_r', title='PO8 cright ave sub'+str(i))
    
#    tf_cleft.average().plot_topomap(tmin=.5, tmax=1.0, fmin=8, fmax=12,
#                    baseline = (-0.3, -0.1), mode='mean', ch_type='eeg', cmap='RdBu_r', sensors=True,
#                    title = 'cleft topo 8-14Hz .5-1s, sub'+str(i))
#    
#    tf_cright.average().plot_topomap(tmin=.5, tmax=1.0, fmin=8, fmax=12,
#                    baseline = (-0.3, -0.1), mode='mean', ch_type='eeg', cmap='RdBu_r', sensors=True,
#                    title = 'cright topo 8-14Hz .5-1s, sub'+str(i))
#    
    tmp = np.multiply(tf_cright.data, -1) #flip signs of all data, so addition becomes subtraction
    tf_tmp = mne.time_frequency.EpochsTFR(info  = tf_cright.info,
                                          data  = tmp,
                                          times = tf_cright.times,
                                          freqs = tf_cright.freqs,
                                          events=tf_cright.events,
                                          event_id=tf_cright.event_id,
                                          metadata = tf_cright.metadata)
    
    lvsr  = mne.grand_average([tf_cleft.average(), tf_tmp.average()])
    lplsr = mne.grand_average([tf_cleft.average(), tf_cright.average()])
    
    lvsr_scaled = np.multiply(np.divide(lvsr.data, lplsr.data),100)
    lvsr.data = lvsr_scaled
    
    lvsr.plot_joint(baseline = (-0.5, -0.2), timefreqs = {(.75,10):(.5,4)}, title = 'lvsr sub'+str(i))
    
    alldata_lvsr.append(lvsr)
    alldata_left.append(tf_cleft.average())
    alldata_right.append(tf_cright.average())
    


pwr_lvsr_gave= mne.grand_average(alldata_lvsr)

baseline = (-0.5,-0.2)
pwr_lvsr_gave.plot_joint(baseline = baseline, timefreqs = {(.75,10):(.5,4)}, title = 'grand ave lvsr cue locked')
pwr_lvsr_gave.plot(baseline = baseline, picks = po8, cmap = 'RdBu_r')
pwr_lvsr_gave.plot(baseline = baseline, picks = mne.pick_channels(pwr_lvsr_gave.ch_names, ['PO7']), cmap = 'RdBu_r')

mne.viz.plot_sensors(pwr_lvsr_gave.info, show_names=True)







    
    