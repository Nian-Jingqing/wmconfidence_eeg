#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:55:47 2019

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
plt.ion()

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri/analysis_scripts')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10])
subs = np.array([5,6,7])
subs = [8, 9, 10]
for i in subs:
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    if i <= 2: #these subjects only have one session
        
        #we're going to read in the raw data, filter it, epoch it around the array/cue triggers and check to see if there are blinks nearby
        raw = mne.io.read_raw_eeglab(input_fname = param['rawset'], montage = 'easycap-M1', eog = ['VEOG', 'HEOG'], preload=True)
        raw.rename_channels({'PO6':'PO8'})
        raw.set_montage('easycap-M1')
        raw.filter(1,40)
        
        #epoching
        #here it's important to specify a dictionary that assigns each trigger to its own integer value
        #mne.events_from_annotations will assign each trigger to an ordered integer, so e.g. trig11 will be 2, but epoching 11 will include another trigger
        #this solves it
        event_id = {'1' : 1, '2':2,'11':11,'12':12,'13':13,'14':14,'21':21,'22':22,'23':23,'24':24,
                    '31':31,'32':32,'33':33,'34':34,'41':41,'42':42,'43':43,'44':44,'51':51,'52':52,'53':53,'54':54,
                    '61':61,'62':62,'63':63,'64':64,'71':71,'72':72,'73':73,'74':74,'76':76,'77':77,'78':78,'254':254,'255':255}
        
        events_array = {'neutral':1, 'cued':2}
        events_cue = {'neutral/probeleft'  : 11,
                      'neutral/proberight' : 12,
                      'cued/probeleft'     : 13,
                      'cued/proberight'    : 14}
        events,_ = mne.events_from_annotations(raw, event_id = event_id)
        tmin, tmax = -0.25, 0.25
        baseline = (None,0)
        
        arraylocked = mne.Epochs(raw, events, events_array, tmin, tmax, baseline, reject_by_annotation=False, preload=True)
        cuelocked   = mne.Epochs(raw, events, events_cue, tmin, tmax, baseline, reject_by_annotation=False, preload=True)        
        bdata = pd.DataFrame.from_csv(path = param['behaviour'])
        arraylocked.metadata = bdata
        cuelocked.metadata   = bdata

        arraylocked.plot(n_epochs=4, picks = ['eog'])
        arraythrowouts = []
        if len(arraylocked.metadata.trialnum) != 320: #should be 320 trials in these datasets
            for x,y in np.ndenumerate(bdata.trialnum): #loop over every trial id in the behavioural data
                if y not in arraylocked.metadata.trialnum.tolist(): #check if the trial ids in behavioural data are in the eeg metadata
                    arraythrowouts.append(y) #if not, this trial was discarded, so note down that this trial was discarded
                    
        cuelocked.plot(n_epochs=4, picks = ['eog'])
        cuethrowouts = []
        if len(cuelocked.metadata.trialnum) != 320:
            for x,y in np.ndenumerate(bdata.trialnum):
                if y not in cuelocked.metadata.trialnum.tolist():
                    cuethrowouts.append(y)
        
        discards = np.concatenate([arraythrowouts, cuethrowouts])
        discards = np.unique(discards) #take only uniques, in case a trial had a blink at both array and cue so we don't double it here
        discard  = np.zeros(max(bdata.trialnum)).astype(int)
        discard[discards] = 1 #mark trials that should be thrown out with a 1
        
        bdata['arraycueblink'] = discard
        
        #save the behavioural data back to csv, with column for if there was a blink by the array or cue presentation
        bdata.to_csv(param['behaviour_blinkchecked'], index=False) 
        
    if i == 3: #subject 3 has one session of 8 blocks because of time constraints
        
        #we're going to read in the raw data, filter it, epoch it around the array/cue triggers and check to see if there are blinks nearby
        raw = mne.io.read_raw_eeglab(input_fname = param['rawset_sess1'], montage = 'easycap-M1', eog = ['VEOG', 'HEOG'], preload=True)
        raw.set_montage('easycap-M1')
        raw.filter(1,40)
        
        #epoching
        #here it's important to specify a dictionary that assigns each trigger to its own integer value
        #mne.events_from_annotations will assign each trigger to an ordered integer, so e.g. trig11 will be 2, but epoching 11 will include another trigger
        #this solves it
        event_id = {'1' : 1, '2':2,'11':11,'12':12,'13':13,'14':14,'21':21,'22':22,'23':23,'24':24,
                    '31':31,'32':32,'33':33,'34':34,'41':41,'42':42,'43':43,'44':44,'51':51,'52':52,'53':53,'54':54,
                    '61':61,'62':62,'63':63,'64':64,'71':71,'72':72,'73':73,'74':74,'76':76,'77':77,'78':78,'254':254,'255':255}
        
        events_array = {'neutral':1, 'cued':2}
        events_cue = {'neutral/probeleft'  : 11,
                      'neutral/proberight' : 12,
                      'cued/probeleft'     : 13,
                      'cued/proberight'    : 14}
        events,_ = mne.events_from_annotations(raw, event_id = event_id)
        tmin, tmax = -0.25, 0.25
        baseline = (None,0)
        
        arraylocked = mne.Epochs(raw, events, events_array, tmin, tmax, baseline, reject_by_annotation=False, preload=True)
        cuelocked   = mne.Epochs(raw, events, events_cue, tmin, tmax, baseline, reject_by_annotation=False, preload=True)        
        bdata = pd.DataFrame.from_csv(path = param['behaviour_sess1'])
        arraylocked.metadata = bdata
        cuelocked.metadata   = bdata

        arraylocked.plot(n_epochs=4, picks = ['eog'])
        arraythrowouts = []
        if len(arraylocked.metadata.trialnum) != len(bdata): #should be 320 trials in these datasets
            for x,y in np.ndenumerate(bdata.trialnum): #loop over every trial id in the behavioural data
                if y not in arraylocked.metadata.trialnum.tolist(): #check if the trial ids in behavioural data are in the eeg metadata
                    arraythrowouts.append(y) #if not, this trial was discarded, so note down that this trial was discarded
                    
        cuelocked.plot(n_epochs=4, picks = ['eog'])
        cuethrowouts = []
        if len(cuelocked.metadata.trialnum) != len(bdata):
            for x,y in np.ndenumerate(bdata.trialnum):
                if y not in cuelocked.metadata.trialnum.tolist():
                    cuethrowouts.append(y)
        
        discards = np.concatenate([arraythrowouts, cuethrowouts])
        discards = np.unique(discards) #take only uniques, in case a trial had a blink at both array and cue so we don't double it here
        discard  = np.zeros(max(bdata.trialnum)).astype(int)
        discard[discards] = 1 #mark trials that should be thrown out with a 1
        
        bdata['arraycueblink'] = discard
        
        #save the behavioural data back to csv, with column for if there was a blink by the array or cue presentation
        bdata.to_csv(param['behaviour_blinkchecked'], index=False) 
        
    if i > 3: #subjects 4 onwards, with two sessions per participant
        for part in ['a', 'b']:
            if part == 'a':
                session = '1'
            if part == 'b':
                session = '2'
                
            #we're going to read in the raw data, filter it, epoch it around the array/cue triggers and check to see if there are blinks nearby
        raw = mne.io.read_raw_eeglab(input_fname = param['rawset_sess'+session], montage = 'easycap-M1', eog = ['VEOG', 'HEOG'], preload=True)
        raw.set_montage('easycap-M1')
        raw.filter(1,40, picks='eog')
        
        #epoching
        #here it's important to specify a dictionary that assigns each trigger to its own integer value
        #mne.events_from_annotations will assign each trigger to an ordered integer, so e.g. trig11 will be 2, but epoching 11 will include another trigger
        #this solves it
        event_id = {'1' : 1, '2':2,'11':11,'12':12,'13':13,'14':14,'21':21,'22':22,'23':23,'24':24,
                    '31':31,'32':32,'33':33,'34':34,'41':41,'42':42,'43':43,'44':44,'51':51,'52':52,'53':53,'54':54,
                    '61':61,'62':62,'63':63,'64':64,'71':71,'72':72,'73':73,'74':74,'76':76,'77':77,'78':78,'254':254,'255':255}
        
        events_array = {'neutral':1, 'cued':2}
        events_cue = {'neutral/probeleft'  : 11,
                      'neutral/proberight' : 12,
                      'cued/probeleft'     : 13,
                      'cued/proberight'    : 14}
        events,_ = mne.events_from_annotations(raw, event_id = event_id)
        tmin, tmax = -0.25, 0.25
        baseline = (None,0)
        
        arraylocked = mne.Epochs(raw, events, events_array, tmin, tmax, baseline, reject_by_annotation=False, preload=True)
        cuelocked   = mne.Epochs(raw, events, events_cue, tmin, tmax, baseline, reject_by_annotation=False, preload=True)        
        bdata = pd.DataFrame.from_csv(path = param['behaviour_sess'+session])
        arraylocked.metadata = bdata
        cuelocked.metadata   = bdata

        arraylocked.plot(n_epochs=4, picks = ['eog'])
        arraythrowouts = []
        if len(arraylocked.metadata.trialnum) != len(bdata): #should be 320 trials in these datasets
            for x,y in np.ndenumerate(bdata.trialnum): #loop over every trial id in the behavioural data
                if y not in arraylocked.metadata.trialnum.tolist(): #check if the trial ids in behavioural data are in the eeg metadata
                    arraythrowouts.append(y) #if not, this trial was discarded, so note down that this trial was discarded
                    
        cuelocked.plot(n_epochs=4, picks = ['eog'])
        cuethrowouts = []
        if len(cuelocked.metadata.trialnum) != len(bdata):
            for x,y in np.ndenumerate(bdata.trialnum):
                if y not in cuelocked.metadata.trialnum.tolist():
                    cuethrowouts.append(y)
        
        discards = np.concatenate([arraythrowouts, cuethrowouts]).astype(int)
        discards = np.unique(discards) #take only uniques, in case a trial had a blink at both array and cue so we don't double it here
        discard  = np.zeros(max(bdata.trialnum)).astype(int)
        discard[discards] = 1 #mark trials that should be thrown out with a 1
        
        bdata['arraycueblink'] = discard
        
        #save the behavioural data back to csv, with column for if there was a blink by the array or cue presentation
        bdata.to_csv(param['behaviour_blinkchecked'+session], index=False) 
        
        
        
        
        
        
        
        
        
        
        