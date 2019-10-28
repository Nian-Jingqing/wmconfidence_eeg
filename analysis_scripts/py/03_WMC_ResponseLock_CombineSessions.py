#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:15:56 2019

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
        event_id = {'1' : 1, '2':2,                 #array
                    '11':11,'12':12,'13':13,'14':14,#cue
                    '21':21,'22':22,'23':23,'24':24,#probe
                    '31':31,'32':32,'33':33,'34':34,#space
                    '41':41,'42':42,'43':43,'44':44,#click
                    '51':51,'52':52,'53':53,'54':54,#confprobe
                    '61':61,'62':62,'63':63,'64':64,#space
                    '71':71,'72':72,'73':73,'74':74,#click
                    '76':76,'77':77,'78':78,'79':79, #feedback
                    '254':254,'255':255}
        
        events_response = {'neutral_left'  : 31,
                           'neutral_right' : 32,
                           'cued_left'     : 33,
                           'cued_right'    : 34} 
        events,_ = mne.events_from_annotations(raw, event_id = event_id)
        tmin, tmax = -1, .5
        baseline = None
        
        resplocked   = mne.Epochs(raw, events, events_response, tmin, tmax, baseline, reject_by_annotation=False, preload=True)        
        bdata = pd.read_csv(param['behaviour_blinkchecked'], index_col=None)
        bdata['prevtrlconfdiff'] = bdata.confdiff.shift(1)  #write down on each trial what the previous trials error awareness was (used in glm)
        bdata['nxtrlconfdiff']   = bdata.confdiff.shift(-1) #write down on each trial what the next trials error awareness was (used in glm)

        resplocked.metadata = bdata

        #save the epoched data, combined with metadata, to file
        resplocked.save(fname=param['resplocked'], overwrite=True)
        del(resplocked)
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
        event_id = {'1' : 1, '2':2,                 #array
                    '11':11,'12':12,'13':13,'14':14,#cue
                    '21':21,'22':22,'23':23,'24':24,#probe
                    '31':31,'32':32,'33':33,'34':34,#space
                    '41':41,'42':42,'43':43,'44':44,#click
                    '51':51,'52':52,'53':53,'54':54,#confprobe
                    '61':61,'62':62,'63':63,'64':64,#space
                    '71':71,'72':72,'73':73,'74':74,#click
                    '76':76,'77':77,'78':78,'79':79, #feedback
                    '254':254,'255':255}
        
        events_response = {'neutral_left'  : 31,
                           'neutral_right' : 32,
                           'cued_left'     : 33,
                           'cued_right'    : 34} 
        events,_ = mne.events_from_annotations(raw, event_id = event_id)
        tmin, tmax = -1, .5
        baseline = None
        
        resplocked   = mne.Epochs(raw, events, events_response, tmin, tmax, baseline, reject_by_annotation=False, preload=True)      
        bdata = pd.read_csv(param['behaviour_blinkchecked'], index_col=None)
        bdata['prevtrlconfdiff'] = bdata.confdiff.shift(1)  #write down on each trial what the previous trials error awareness was (used in glm)
        bdata['nxtrlconfdiff']   = bdata.confdiff.shift(-1) #write down on each trial what the next trials error awareness was (used in glm)

        
        #this section is used because there's a trigger missing in subject 10 for some reason, not sure why
        if resplocked.events.shape[0] != bdata.shape[0]: #check if there are different numbers of trials in the behavioural and event locked data
            #if there are, we need to iteratively fix this (by removing trial at a time)
            print('\nmissing triggers in subject %02d'%i)
            evtrigs    = resplocked.events[:,2]
            #evtrigs = np.delete(evtrigs, 100) #this will just remove a random trial just to see if it works with more than 1 missing (it does)
            bdatatrigs = bdata.moveontrig.to_numpy()
            count = 0
            
            numtorem = np.max([evtrigs.size,bdatatrigs.size]) - np.min([evtrigs.size, bdatatrigs.size])
            print('%d trial(s) need removing\n'%numtorem)
            removed = 0
            while removed != numtorem:
                for x,y in enumerate(evtrigs):
                    found = 0
                    if y != bdatatrigs[x]:
                        found = 1
                        to_rem = x
                        removed += 1
                        bdatatrigs = np.delete(bdatatrigs, to_rem)
                        bdata = bdata.drop(to_rem)
                        if found == 1:
                            break
        
        resplocked.metadata = bdata

        resplocked.save(fname = param['resplocked'], overwrite=True)        
        del(resplocked)
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
        event_id = {'1' : 1, '2':2,                 #array
                    '11':11,'12':12,'13':13,'14':14,#cue
                    '21':21,'22':22,'23':23,'24':24,#probe
                    '31':31,'32':32,'33':33,'34':34,#space
                    '41':41,'42':42,'43':43,'44':44,#click
                    '51':51,'52':52,'53':53,'54':54,#confprobe
                    '61':61,'62':62,'63':63,'64':64,#space
                    '71':71,'72':72,'73':73,'74':74,#click
                    '76':76,'77':77,'78':78,'79':79, #feedback
                    '254':254,'255':255}
        
        events_response = {'neutral_left'  : 31,
                           'neutral_right' : 32,
                           'cued_left'     : 33,
                           'cued_right'    : 34} 
        events1, _ = mne.events_from_annotations(raw1, event_id = event_id)
        events2, _ = mne.events_from_annotations(raw2, event_id = event_id)
        tmin, tmax = -1, .5
        baseline = None
        
        resplocked1   = mne.Epochs(raw1, events1, events_response, tmin, tmax, baseline, reject_by_annotation=False, preload=True)        
        resplocked2   = mne.Epochs(raw2, events2, events_response, tmin, tmax, baseline, reject_by_annotation=False, preload=True)        

        bdata1 = pd.read_csv(param['behaviour_blinkchecked1'], index_col=None)
        bdata1['prevtrlconfdiff'] = bdata1.confdiff.shift(1)  #write down on each trial what the previous trials error awareness was (used in glm)
        bdata1['nxtrlconfdiff']   = bdata1.confdiff.shift(-1) #write down on each trial what the next trials error awareness was (used in glm)
        bdata2 = pd.read_csv(param['behaviour_blinkchecked2'], index_col=None)
        bdata2['prevtrlconfdiff'] = bdata2.confdiff.shift(1)  #write down on each trial what the previous trials error awareness was (used in glm)
        bdata2['nxtrlconfdiff']   = bdata2.confdiff.shift(-1) #write down on each trial what the next trials error awareness was (used in glm)

        resplocked1.metadata = bdata1
        resplocked2.metadata = bdata2
        
        resplocked = mne.concatenate_epochs([resplocked1, resplocked2]) #combine the epoched data with aligned metadata

        resplocked.save(fname = param['resplocked'], overwrite=True)  
        del(resplocked)
        del(resplocked1)
        del(resplocked2)
        del(raw1)
        del(raw2)