#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 12:03:07 2019

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


subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([        4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])
subs = np.array([                   21, 22,     24, 25, 26])
for i in subs:
    sub   = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    
    if i in [1,2]: #these are two pilot subjects that have 10 blocks in one session, probably not going to be used
        raw = mne.io.read_raw_eeglab(
                input_fname=param['rawset'],
                montage = 'easycap-M1',
                eog=['VEOG', 'HEOG'], preload = True)
        raw.rename_channels({'PO6':'PO8'})
        raw.set_montage('easycap-M1')
        
        raw.filter(1, 40) #filter raw data
        ica = mne.preprocessing.ICA(n_components = .99, method = 'infomax').fit(raw)
        eog_epochs = mne.preprocessing.create_eog_epochs(raw)
        eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
        ica.plot_scores(eog_scores)
        
        ica.plot_components(inst=raw)
        ica.exclude.extend(eog_inds)
        ica.apply(inst=raw)
        
        raw.save(fname = param['rawcleaned'], fmt='double')
    elif i in [3, 10, 19]:
        raw = mne.io.read_raw_eeglab(
                input_fname=param['rawset_sess1'],
                montage = 'easycap-M1',
                eog=['VEOG', 'HEOG'], preload = True)
        #raw.set_montage('easycap-M1')
        
        #left mastoid was online reference
        #right mastoid recorded for offline re-referencing
        #add in channel of zeros for left mastoid, and rename left and right
        raw = mne.add_reference_channels(raw, ref_channels = 'LM')
        raw.rename_channels({'RM':'M2', 'LM':'M1'})
        
        chnames = np.asarray(raw.ch_names)
        chnamemapping = {}
        for x in range(len(chnames)):
            chnamemapping[chnames[x]] = chnames[x].replace('Z', 'z').replace('FP', 'Fp')
                
        raw.rename_channels(chnamemapping)
        raw.set_montage('easycap-M1', raise_if_subset = False)
        
        raw.filter(1, 40) #filter raw data
        ica = mne.preprocessing.ICA(n_components = 60, method = 'infomax').fit(raw)
        eog_epochs = mne.preprocessing.create_eog_epochs(raw)
        eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
        ica.plot_scores(eog_scores)
        
        ica.plot_components(inst=raw)
        ica.exclude.extend(eog_inds)
        ica.apply(inst=raw)
        
        raw.save(fname = param['rawcleaned_sess1'], fmt='double')
        
    else:
        for part in [1,2]:#[1, 2]:
            sub   = dict(loc = 'workstation', id = i)
            param = get_subject_info_wmConfidence(sub)
            
            raw = mne.io.read_raw_eeglab(
                input_fname=param['rawset_sess'+str(part)],
                #montage = 'easycap-M1',
                eog=['VEOG', 'HEOG'], preload = True)
            #raw.set_channel_types(mapping = dict(RM = 'misc'))
            
            #left mastoid was online reference
            #right mastoid recorded for offline re-referencing
            #add in channel of zeros for left mastoid, and rename left and right
            raw = mne.add_reference_channels(raw, ref_channels = 'LM')
            raw.rename_channels({'RM':'M2', 'LM':'M1'})
            raw.set_eeg_reference(ref_channels = ['M2', 'M1'])
            
            chnames = np.asarray(raw.ch_names)
            chnamemapping = {}
            for x in range(len(chnames)):
                chnamemapping[chnames[x]] = chnames[x].replace('Z', 'z').replace('FP', 'Fp')
                
            raw.rename_channels(chnamemapping)
            if i == 26:
                raw.rename_channels(mapping = {'VEOG':'HEO', 'HEOG':'VEO'})
                raw.rename_channels(mapping = dict(HEO='HEOG', VEO = 'VEOG')) #in this subject, they were put in the wrong order!
            raw.set_montage('easycap-M1', raise_if_subset = False)
            raw.set_channel_types(mapping = dict(M1 = 'misc', M2 = 'misc'))
            #mont=mne.channels.read_montage('easycap-M1')
            #raw.set_eeg_reference(ref_channels=['RM'])
            
            
        
            
            
            raw.filter(1, 40, picks = ['eeg', 'eog', 'misc']) #filter raw data
#            raw.plot(n_channels=64, duration=10,scalings=dict(eog=40e-6)) #run this if you didn't note down what channels were bad during recording
#            raw.set_eeg_reference(ref_channels = ['RM'])
            
            for chan in range(len(param['badchans'+str(part)])):
                if 'FP' in param['badchans'+str(part)][chan]:
                    param['badchans'+str(part)][chan] = param['badchans'+str(part)][chan].replace('FP', 'Fp')
                    
            
            raw.info['bads'] = param['badchans'+str(part)] #get the noisy channels for the dataset that we want to interpolate
            raw.interpolate_bads() #interpolate these channels
            #raw.plot(n_channels=64, duration=10)#,scalings=dict(eog=40e-6)) #use this to exclude data between blocks as v noisy and messes ICA
            
            if i == 11 and part == 2:
                raw.info['bads'].extend(['P1']) #this channel messes up ica big time
#                ica = mne.preprocessing.ICA(n_components = 0.999, method = 'fastica').fit(raw)
            ica = mne.preprocessing.ICA(n_components=.99, method = 'fastica').fit(raw) #this will by default not include periods marked as bad

            eog_epochs = mne.preprocessing.create_eog_epochs(raw) #;eog_epochs.average().plot_joint()
            eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
            ica.plot_scores(eog_scores, eog_inds)
                        
            ica.plot_components(inst=raw)
            print('subject %d, part %d components to remove:'%(i, part), eog_inds)
            
            comps = ica.get_sources(inst=raw)
            c = comps.get_data()
            rsac = np.empty(shape = (c.shape[0]))
            rblk = np.empty(shape = (c.shape[0]))
            heog = raw.get_data(picks = 'HEOG').squeeze()
            veog = raw.get_data(picks = 'VEOG').squeeze()
            for comp in range(len(c)):
                rsac[comp] = sp.stats.pearsonr(c[comp,:], heog)[0]
                rblk[comp] = sp.stats.pearsonr(c[comp,:], veog)[0]
            
            fig = plt.figure()
            ax = fig.add_subplot(2,1,1)
            ax.bar(np.arange(len(c)), rsac)
            ax.set_title('corr with heog')
            
            ax2 = fig.add_subplot(2,1,2)
            ax2.bar(np.arange(len(c)), rblk)
            ax2.set_title('corr with veog')
           
            plt.pause(3)
            
            comps2rem = input('components to remove: ') #need to separate component numbers by comma and space
            comps2rem = list(map(int, comps2rem.split(', ')))
            
            
            
            ica.exclude.extend(comps2rem)
#            if i == 26 and part == 1:
#                ica.exclude.extend([0, 5, 8, 9])
#            elif i == 26 and part == 2:
#                ica.exclude.extend([0,5, 12, 15, 17, 19])
            ica.apply(inst=raw)
            
#            if i == 26 and part == 1: #if we remove that component that is noise in frontal channels, do we get proper artefacts in the rest of the data too that we can use
#                ica = mne.preprocessing.ICA(n_components = .99, method = 'fastica').fit(raw)
#                
#                eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name = 'HEOG')
#                eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)
#                ica.plot_scores(eog_scores, eog_inds)
#                ica.plot_components(inst=raw)
#                print('subject %d, part %d components to remove:'%(i, part), eog_inds)
            
            raw.save(fname = param['rawcleaned_sess'+str(part)], fmt='double', overwrite = 'True')
            plt.close('all')

#%%  