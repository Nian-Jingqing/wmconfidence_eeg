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


subs = np.array([1,2, 3, 4, 5, 6, 7])
#subs = np.array([5,6,7])
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
        event_id = {'1' : 1, '2':2,'11':11,'12':12,'13':13,'14':14,'21':21,'22':22,'23':23,'24':24,
                    '31':31,'32':32,'33':33,'34':34,'41':41,'42':42,'43':43,'44':44,'51':51,'52':52,'53':53,'54':54,
                    '61':61,'62':62,'63':63,'64':64,'71':71,'72':72,'73':73,'74':74,'76':76,'77':77,'78':78,'79':79,
                    '254':254,'255':255}
        
        events_fb = {'neutral_left'  : 76,
                      'neutral_right' : 77,
                      'cued_left'     : 78,
                      'cued_right'    : 79}
        events,_ = mne.events_from_annotations(raw, event_id = event_id)
        tmin, tmax = -0.5, 1.5
        baseline = (None,0)
        
        fblocked   = mne.Epochs(raw, events, events_fb, tmin, tmax, baseline, reject_by_annotation=False, preload=True)        
        bdata = pd.DataFrame.from_csv(path = param['behaviour_blinkchecked'], index_col=None)
        fblocked.metadata = bdata

        #save the epoched data, combined with metadata, to file
        fblocked.save(fname=param['fblocked'], overwrite=True)
        del(fblocked)
        del(raw)

        
    if i == 3: #subject 3 has one session of 8 blocks because of time constraints
        
        #we're going to read in the raw data, filter it, epoch it around the array/cue triggers and check to see if there are blinks nearby
        raw = mne.io.read_raw_fif(fname = param['rawcleaned_sess1'], preload=True)
        raw.set_montage('easycap-M1')
        #raw.filter(1,40)
        
        #epoching
        #here it's important to specify a dictionary that assigns each trigger to its own integer value
        #mne.events_from_annotations will assign each trigger to an ordered integer, so e.g. trig11 will be 2, but epoching 11 will include another trigger
        #this solves it
        event_id = {'1' : 1, '2':2,'11':11,'12':12,'13':13,'14':14,'21':21,'22':22,'23':23,'24':24,
                    '31':31,'32':32,'33':33,'34':34,'41':41,'42':42,'43':43,'44':44,'51':51,'52':52,'53':53,'54':54,
                    '61':61,'62':62,'63':63,'64':64,'71':71,'72':72,'73':73,'74':74,'76':76,'77':77,'78':78,'79':79,
                    '254':254,'255':255}
        
        events_fb = {'neutral_left'  : 76,
                      'neutral_right' : 77,
                      'cued_left'     : 78,
                      'cued_right'    : 79}
        events,_ = mne.events_from_annotations(raw, event_id = event_id)
        tmin, tmax = -0.5, 1.5
        baseline = (None,0)
        
        fblocked   = mne.Epochs(raw, events, events_fb, tmin, tmax, baseline, reject_by_annotation=False, preload=True)        
        bdata = pd.DataFrame.from_csv(path = param['behaviour_blinkchecked'], index_col=None)
        fblocked.metadata = bdata

        fblocked.save(fname = param['fblocked'], overwrite=True)        
        del(fblocked)
        del(raw)
    if i > 3: #subjects 4 onwards, with two sessions per participant
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
        event_id = {'1' : 1, '2':2,'11':11,'12':12,'13':13,'14':14,'21':21,'22':22,'23':23,'24':24,
                    '31':31,'32':32,'33':33,'34':34,'41':41,'42':42,'43':43,'44':44,'51':51,'52':52,'53':53,'54':54,
                    '61':61,'62':62,'63':63,'64':64,'71':71,'72':72,'73':73,'74':74,'76':76,'77':77,'78':78,'79':79,
                    '254':254,'255':255}
        
        events_fb = {'neutral_left'  : 76,
                      'neutral_right' : 77,
                      'cued_left'     : 78,
                      'cued_right'    : 79}
        events1, _ = mne.events_from_annotations(raw1, event_id = event_id)
        events2, _ = mne.events_from_annotations(raw2, event_id = event_id)
        tmin, tmax = -0.5, 1.5
        baseline = (None,0)
        
        fblocked1   = mne.Epochs(raw1, events1, events_fb, tmin, tmax, baseline, reject_by_annotation=False, preload=True)        
        fblocked2   = mne.Epochs(raw2, events2, events_fb, tmin, tmax, baseline, reject_by_annotation=False, preload=True)        

        bdata1 = pd.DataFrame.from_csv(path = param['behaviour_blinkchecked1'], index_col=None)
        bdata2 = pd.DataFrame.from_csv(path = param['behaviour_blinkchecked2'], index_col=None)
        fblocked1.metadata = bdata1
        fblocked2.metadata = bdata2
        
        fblocked = mne.concatenate_epochs([fblocked1, fblocked2]) #combine the epoched data with aligned metadata

        fblocked.save(fname = param['fblocked'], overwrite=True)  
        del(fblocked)
        del(fblocked1)
        del(fblocked2)
        del(raw1)
        del(raw2)