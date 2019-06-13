#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:03:39 2019

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
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence_eegfmri/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence_eegfmri' #workstation wd
os.chdir(wd)


subs = np.array(['pilot1_inside', 'pilot2_inside', 'pilot1_outside'])


subind = 3 #get first subject

sub = dict(loc='workstation',
           id=subs[subind-1]) #get subject name for param extraction


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


#read in behavioural data to allow removal of trials
bdata = pd.DataFrame.from_csv(path = param['behaviour'])
cuelocked_epochs.metadata = bdata

cuelocked_epochs = cuelocked_epochs['DTcheck ==0 and clickresp == 1'] #throw out trials based on behavioural data
cuelocked_epochs.filter(1,30)
#go through and check for bad epochs -- remove trls with excessive blinks and/or excessive noise on visual inspection
cuelocked_epochs.plot(scalings='auto', n_epochs=4, n_channels=len(cuelocked_epochs.info['ch_names'])) 
cuelocked_epochs.drop_bad()


cuelocked_epochs_noref = deepcopy(cuelocked_epochs)
cuelocked_epochs_mast  = deepcopy(cuelocked_epochs).set_eeg_reference(ref_channels = ['M1', 'M2'])
cuelocked_epochs_car   = deepcopy(cuelocked_epochs).set_eeg_reference(ref_channels = 'average')

#save these data objects -- can't seem to save them (assert not self.times.flags['WRITEABLE']) some error in mne


#for data with no re-referencing
tl_cueleft_noref = cuelocked_epochs_noref['cued/probeleft'].average()
tl_cueleft_noref.apply_baseline(baseline = (-0.5, 0))
tl_cueright_noref = cuelocked_epochs_noref['cued/proberight'].average()
tl_cueright_noref.apply_baseline(baseline=(-0.5, 0))

mne.viz.plot_evoked_topomap(tl_cueleft_noref, sensors = 'k+', times = np.arange(0,1,0.2), average=.4,
                            title = 'evoked scalp activity, cued left, no reference')


#for data with mastoid re-referencing
tl_cueleft_mast  = cuelocked_epochs_mast['cued/probeleft'].average()
tl_cueleft_mast.apply_baseline(baseline=(-0.5,0))
tl_cueright_mast = cuelocked_epochs_mast['cued/proberight'].average()
tl_cueright_mast.apply_baseline(baseline=(-0.5,0))

mne.viz.plot_evoked_topomap(tl_cueleft_mast, sensors = 'k+', times = np.arange(0,1,0.2), average=.4,
                            title = 'evoked scalp activity, cued left, mastoid reference')

#now for data with common average referencing
tl_cueleft_car = cuelocked_epochs_car['cued/probeleft'].average()
tl_cueleft_car.apply_baseline(baseline=(-0.5,0))
tl_cueright_car = cuelocked_epochs_car['cued/proberight'].average()
tl_cueright_car.apply_baseline(baseline=(-0.5,0))


mne.viz.plot_evoked_topomap(tl_cueleft_car, sensors = 'k+', times = np.arange(0,1,0.2), average=.4,
                            title = 'evoked scalp activity, cued left, common reference')



#get indices for right visual channels
visright_picks = mne.pick_channels(tl_cueleft_noref.ch_names, ['PO8', 'O2', 'PO4'])
visleft_picks  = mne.pick_channels(tl_cueleft_noref.ch_names, ['PO7', 'O1', 'PO3'])


#plot difference in evoked activity with no reference applied
tl_lvsr_noref = mne.combine_evoked([tl_cueleft_noref, -tl_cueright_noref], weights = 'equal')
tl_lvsr_noref.plot_topomap(times=0.6, average=.400, vmin=-1, vmax=1)

mne.viz.plot_compare_evokeds(dict(cueleft=tl_cueleft_noref, cueright=tl_cueright_noref),
                            show_legend='upper left', show_sensors='upper right',
                            picks = np.ravel([visleft_picks,visright_picks]))

tl_lvsr_noref.plot_joint(times=np.arange(0,1,.2), picks=np.ravel([visleft_picks,visright_picks]))



#for common average reference
tl_lvsr_car = mne.combine_evoked([tl_cueleft_car, -tl_cueright_car], weights = 'equal')
tl_lvsr_car.plot_topomap(times=0.6, average=.400)#, vmin=-1, vmax=1)
mne.viz.plot_compare_evokeds(dict(cueleft=tl_cueleft_car, cueright=tl_cueright_car),
                            show_legend='upper left', show_sensors='upper right',
                            picks = np.ravel([visleft_picks,visright_picks]))

tl_lvsr_car.plot_joint(times=np.arange(0,1,.2), picks=np.ravel([visleft_picks,visright_picks]))


#for mastoid reference
tl_lvsr_mast = mne.combine_evoked([tl_cueleft_mast, -tl_cueright_mast], weights = 'equal')
tl_lvsr_mast.plot_topomap(times=0.6, average=.400)#,)
mne.viz.plot_compare_evokeds(dict(cueleft=tl_cueleft_mast, cueright=tl_cueright_mast),
                            show_legend='upper left', show_sensors='upper right',
                            picks = np.ravel([visleft_picks,visright_picks]))
