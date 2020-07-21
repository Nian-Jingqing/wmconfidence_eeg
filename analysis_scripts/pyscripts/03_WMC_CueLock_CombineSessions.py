#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:05:13 2019

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

subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])

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

        events_cue = {'neutral_left'  : 11,
                      'neutral_right' : 12,
                      'cued_left'     : 13,
                      'cued_right'    : 14}
        events,_ = mne.events_from_annotations(raw, event_id = event_id)
        tmin, tmax = -2, 2.25
        baseline = None

        cuelocked   = mne.Epochs(raw, events, events_cue, tmin, tmax, baseline, reject_by_annotation=False, preload=True)
        bdata = pd.read_csv(param['behaviour_blinkchecked'], index_col=None)
        cuelocked.metadata = bdata

        #save the cuelocked data, combined with metadata, to file
        cuelocked.save(fname=param['cuelocked'], overwrite=True)
        del(cuelocked)
        del(raw)


    if i == 3 or i == 10 or i == 19: #subject 3 has one session of 8 blocks because of time constraints

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

        events_array = {'neutral':1, 'cued':2}
        events_cue = {'neutral_left'  : 11,
                      'neutral_right' : 12,
                      'cued_left'     : 13,
                      'cued_right'    : 14}
        events,_ = mne.events_from_annotations(raw, event_id = event_id)
        tmin, tmax = -2, 2.25
        baseline = None

        cuelocked   = mne.Epochs(raw, events, events_cue, tmin, tmax, baseline, reject_by_annotation=False, preload=True)
        if i == 19:
            bdata = pd.read_csv(param['behaviour_blinkchecked1'], index_col = None)
        else:
            bdata = pd.read_csv(param['behaviour_blinkchecked'], index_col=None)

        cuelocked.metadata = bdata #assign behavioural data into the datafile from here on

        cuelocked.save(fname = param['cuelocked'], overwrite=True)
        del(cuelocked)
        del(raw)
    if i > 3 and i != 10 and i!=19: #subjects 4 onwards, with two sessions per participant
        parts = ['a', 'b']

        #we're going to read in the raw data, filter it, epoch it around the array/cue triggers and check to see if there are blinks nearby
        raw1 = mne.io.read_raw_fif(fname = param['rawcleaned_sess1'], preload=True)
        raw2 = mne.io.read_raw_fif(fname = param['rawcleaned_sess2'], preload=True)

#        raw1.set_montage('easycap-M1');
#        raw2.set_montage('easycap-M1')

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

        events_array = {'neutral':1, 'cued':2}
        events_cue = {'neutral_left'  : 11,
                      'neutral_right' : 12,
                      'cued_left'     : 13,
                      'cued_right'    : 14}
        events1, _ = mne.events_from_annotations(raw1, event_id = event_id)
        events2, _ = mne.events_from_annotations(raw2, event_id = event_id)
        tmin, tmax = -2, 2.25
        baseline = None

        cuelocked1   = mne.Epochs(raw1, events1, events_cue, tmin, tmax, baseline, reject_by_annotation=False, preload=True)
        cuelocked2   = mne.Epochs(raw2, events2, events_cue, tmin, tmax, baseline, reject_by_annotation=False, preload=True)

        bdata1 = pd.read_csv(param['behaviour_blinkchecked1'], index_col=None)
        bdata1['prevtrlconfdiff'] = bdata1.confdiff.shift(1)  #write down on each trial what the previous trial error awareness was
        bdata1['nexttrlconfdiff'] = bdata1.confdiff.shift(-1) #write down on each trial what the next trials error awareness is (used in glm)
        bdata1['prevtrlcw']       = bdata1.confwidth.shift(1) #what was the confidence width on the previous trial?
        bdata1['nexttrlcw']       = bdata1.confwidth.shift(-1) #and what is the confidence width on the next trial?

        bdata2 = pd.read_csv(param['behaviour_blinkchecked2'], index_col=None)
        bdata2['prevtrlconfdiff'] = bdata2.confdiff.shift(1)  #write down on each trial what the previous trial error awareness was
        bdata2['nexttrlconfdiff'] = bdata2.confdiff.shift(-1) #write down on each trial what the next trials error awareness is (used in glm)
        bdata2['prevtrlcw']       = bdata2.confwidth.shift(1) #what was the confidence width on the previous trial?
        bdata2['nexttrlcw']       = bdata2.confwidth.shift(-1) #and what is the confidence width on the next trial?
        cuelocked1.metadata = bdata1
        cuelocked2.metadata = bdata2

        #bad channels get marked and cant concatenate epochs unless these are the same
        if cuelocked1.info['bads'] != []:
            cuelocked1.interpolate_bads(reset_bads = True)
        if cuelocked2.info['bads'] != []:
            cuelocked2.interpolate_bads(reset_bads = True)




        cuelocked = mne.concatenate_epochs([cuelocked1, cuelocked2]) #combine the epoched data with aligned metadata
        
        #trial rejection here
        #step1 - automated gesd removal
        #step 2, just check these epochs to make sure some haven't slipped through the net by accident. looking for more catastrophic failures, or noise in baseline
        _, keeps = plot_AR(cuelocked.pick_types(eeg=True), method = 'gesd', zthreshold = 1.5, p_out=.1, alpha = .05, outlier_side = 1)
        keeps = keeps.flatten()
        plt.close()

        discards = np.ones(len(cuelocked), dtype = 'bool')
        discards[keeps] = False
        cuelocked = cuelocked.drop(discards) #first we'll drop trials with excessive noise in the EEG

        cuelocked = cuelocked['DTcheck == 0 and clickresp == 1'] #the last trial of the session doesn't have a following trial!

        #now go through manually
#        cuelocked.plot(n_channels=62, scalings = dict(eeg=200e-6), n_epochs=3)
        
        
        
        cuelocked.save(fname = param['cuelocked'], overwrite=True)
        del(cuelocked)
        del(cuelocked1)
        del(cuelocked2)
        del(raw1)
        del(raw2)
