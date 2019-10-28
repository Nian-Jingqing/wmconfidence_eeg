#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:29:05 2019

@author: sammirc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:23:54 2019

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


subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
subs = np.array([17, 18])
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    if i <= 2: #these subjects only have one session
        
        #read in the already ica cleaned data, already been filtered too
        raw = mne.io.read_raw_fif(fname = param['rawcleaned'], preload = True)
        raw.set_montage('easycap-M1')
        
        #epoching
        #here it's important to specify a dictionary that assigns each trigger to its own integer value
        #mne.events_from_annotations will assign each trigger to an ordered integer,
        #so e.g. trig11 will be 2, but epoching 11 will include another trigger
        #this solves it
        event_id = {'1' : 1, '2':2,
                    '11':11,'12':12,'13':13,'14':14,
                    '21':21,'22':22,'23':23,'24':24,
                    '31':31,'32':32,'33':33,'34':34,
                    '41':41,'42':42,'43':43,'44':44,
                    '51':51,'52':52,'53':53,'54':54,
                    '61':61,'62':62,'63':63,'64':64,
                    '71':71,'72':72,'73':73,'74':74,
                    '76':76,'77':77,'78':78,'79':79,
                    '254':254,'255':255}
        
        events_probe = {'neutral_left'  : 21,
                        'neutral_right' : 22,
                        'cued_left'     : 23,
                        'cued_right'    : 24} 
        events,_ = mne.events_from_annotations(raw, event_id = event_id)
        tmin, tmax = -2, 1.0
        baseline = None
        
        probelocked   = mne.Epochs(raw, events, events_probe, tmin, tmax, baseline, reject_by_annotation=False, preload=True)        
        bdata = pd.read_csv(param['behaviour_blinkchecked'], index_col=None)
        bdata['nexttrlconfdiff'] = bdata.confdiff.shift(1) #write down on each trial what the subsequent trials error awareness was (used in glm)

        probelocked.metadata = bdata

        #save the epoched data, combined with metadata, to file
        probelocked.save(fname=param['probelocked'], overwrite=True)
        del(probelocked)
        del(raw)

        
    if i == 3 or i == 10: #subject 3 has one session of 8 blocks because of time constraints
        
        #we're going to read in the raw data, filter it, epoch it around the array/cue triggers and check to see if there are blinks nearby
        raw = mne.io.read_raw_fif(fname = param['rawcleaned_sess1'], preload=True)
        raw.set_montage('easycap-M1')
        #raw.filter(1,40)
        
        #epoching
        #here it's important to specify a dictionary that assigns each trigger to its own integer value
        #mne.events_from_annotations will assign each trigger to an ordered integer, so e.g. trig11 will be 2, but epoching 11 will include another trigger
        #this solves it
        event_id = {'1' : 1, '2':2,
                    '11':11,'12':12,'13':13,'14':14,
                    '21':21,'22':22,'23':23,'24':24,
                    '31':31,'32':32,'33':33,'34':34,
                    '41':41,'42':42,'43':43,'44':44,
                    '51':51,'52':52,'53':53,'54':54,
                    '61':61,'62':62,'63':63,'64':64,
                    '71':71,'72':72,'73':73,'74':74,
                    '76':76,'77':77,'78':78,'79':79,
                    '254':254,'255':255}
        
        events_probe = {'neutral_left'  : 21,
                        'neutral_right' : 22,
                        'cued_left'     : 23,
                        'cued_right'    : 24} 
        events,_ = mne.events_from_annotations(raw, event_id = event_id)
        tmin, tmax = -2, 1.0
        baseline = None
        
        probelocked   = mne.Epochs(raw, events, events_probe, tmin, tmax, baseline, reject_by_annotation=False, preload=True)      
        bdata = pd.read_csv(param['behaviour_blinkchecked'], index_col=None)
        bdata['nexttrlconfdiff'] = bdata.confdiff.shift(1) #write down on each trial what the subsequent trials error awareness was (used in glm)

        probelocked.metadata = bdata

        probelocked.save(fname = param['probelocked'], overwrite=True)        
        del(probelocked)
        del(raw)
    if i > 3 and i != 10: #subjects 4 onwards, with two sessions per participant
        parts = ['a', 'b']

        #we're going to read in the raw data, filter it, epoch it around the array/cue triggers and check to see if there are blinks nearby
        raw1 = mne.io.read_raw_fif(fname = param['rawcleaned_sess1'], preload=True)
        raw2 = mne.io.read_raw_fif(fname = param['rawcleaned_sess2'], preload=True)

        raw1.set_montage('easycap-M1')
        raw2.set_montage('easycap-M1')
        
        #epoching
        #here it's important to specify a dictionary that assigns each trigger to its own integer value
        #mne.events_from_annotations will assign each trigger to an ordered integer, so e.g. trig11 will be 2, but epoching 11 will include another trigger
        #this solves it
        event_id = {'1' : 1, '2':2,
                    '11':11,'12':12,'13':13,'14':14,
                    '21':21,'22':22,'23':23,'24':24,
                    '31':31,'32':32,'33':33,'34':34,
                    '41':41,'42':42,'43':43,'44':44,
                    '51':51,'52':52,'53':53,'54':54,
                    '61':61,'62':62,'63':63,'64':64,
                    '71':71,'72':72,'73':73,'74':74,
                    '76':76,'77':77,'78':78,'79':79,
                    '254':254,'255':255}
        
        events_probe = {'neutral_left'  : 21,
                        'neutral_right' : 22,
                        'cued_left'     : 23,
                        'cued_right'    : 24} 
        events1, _ = mne.events_from_annotations(raw1, event_id = event_id)
        events2, _ = mne.events_from_annotations(raw2, event_id = event_id)
        tmin, tmax = -2, 1.0
        baseline = None
        
        probelocked1   = mne.Epochs(raw1, events1, events_probe, tmin, tmax, baseline, reject_by_annotation=False, preload=True)        
        probelocked2   = mne.Epochs(raw2, events2, events_probe, tmin, tmax, baseline, reject_by_annotation=False, preload=True)        

        bdata1 = pd.read_csv(param['behaviour_blinkchecked1'], index_col=None)
        bdata1['nexttrlconfdiff'] = bdata1.confdiff.shift(1)#write down on each trial what the subsequent trials error awareness was (used in glm)
        bdata2 = pd.read_csv(param['behaviour_blinkchecked2'], index_col=None)
        bdata2['nexttrlconfdiff'] = bdata2.confdiff.shift(1)#write down on each trial what the subsequent trials error awareness was (used in glm)

        probelocked1.metadata = bdata1
        probelocked2.metadata = bdata2
        
        probelocked = mne.concatenate_epochs([probelocked1, probelocked2]) #combine the epoched data with aligned metadata

        probelocked.save(fname = param['probelocked'], overwrite=True)  
        del(probelocked)
        del(probelocked1)
        del(probelocked2)
        del(raw1)
        del(raw2)