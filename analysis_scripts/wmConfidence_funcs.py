#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:25:35 2019

@author: sammi
"""

#functions that are going to be useful in analysis of eeg data in python
import os.path as op

def get_subject_info_wmConfidence(subject):
    
    param = {}
    
    if subject['loc']   == 'workstation':
        param['path']   = '/home/sammirc/Desktop/DPhil/wmConfidence/data'
    elif subject['loc'] == 'laptop': 
        param['path']   = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence/data'
    
    if subject['id'] == 1:
        param['subid']          = 's01'
        param['behaviour']      = op.join(param['path'], 'datafiles/s01/wmConfidence_S01_allData_preprocessed.csv')
        param['rawdata']        = op.join(param['path'], 'eeg/s01/wmConfidence_s01_12062019.cdt')
        param['rawset']         = op.join(param['path'], 'eeg/s01/wmConfidence_s01_12062019.set')
        param['rawcleaned']     = op.join(param['path'], 'eeg/s01/wmConfidence_s01_icacleaned_raw.fif')
        param['cuelock_noref']  = op.join(param['path'], 'eeg/s01/wmConfidence_s01_cuelock_noref-epo.fif')
        param['cuelock_car']    = op.join(param['path'], 'eeg/s01/wmConfidence_s01_cuelock_car-epo.fif')
        param['cuelock_mast']   = op.join(param['path'], 'eeg/s01/wmConfidence_s01_cuelock_mast-epo.fif')
        param['raweyes']        = op.join(param['path'], 'eyes/s01/WMCS01.asc')
        param['cleanedeyes']    = op.join(param['path'], 'eyes/s01/wmConfidence_s01_preprocessed.pickle')
        
    if subject['id'] == 2:
        param['subid']          = 's02'
        param['behaviour']      = op.join(param['path'], 'datafiles/s02/wmConfidence_S02_allData_preprocessed.csv')
        param['rawdata']        = op.join(param['path'], 'eeg/s02/wmConfidence_s02_12062019.cdt')
        param['rawset']         = op.join(param['path'], 'eeg/s02/wmConfidence_s02_12062019.set')
        param['rawcleaned']     = op.join(param['path'], 'eeg/s02/wmConfidence_s02_icacleaned_raw.fif')
        param['cuelock_noref']  = op.join(param['path'], 'eeg/s02/wmConfidence_s02_cuelock_noref-epo.fif')
        param['cuelock_car']    = op.join(param['path'], 'eeg/s02/wmConfidence_s02_cuelock_car-epo.fif')
        param['cuelock_mast']   = op.join(param['path'], 'eeg/s02/wmConfidence_s02_cuelock_mast-epo.fif')
        param['raweyes']        = op.join(param['path'], 'eyes/s02/WMCS02.asc')
        param['cleanedeyes']    = op.join(param['path'], 'eyes/s02/wmConfidence_s02_preprocessed.pickle')
    
    
    return param

