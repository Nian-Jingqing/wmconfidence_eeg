#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:25:35 2019

@author: sammi
"""

#functions that are going to be useful in analysis of eeg data in python
import os.path as op
import numpy as np
import scipy as sp
from copy import deepcopy
from scipy import ndimage
import mne
from mne import *

def get_subject_info_wmConfidence(subject):
    
    param = {}
    
    if subject['loc']   == 'workstation':
        param['path']   = '/home/sammirc/Desktop/DPhil/wmConfidence/data'
    elif subject['loc'] == 'laptop': 
        param['path']   = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence/data'
    
    if subject['id'] == 1:
        param['subid']                  = 's01'
        param['behaviour']              = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S01_allData_preprocessed.csv')
        param['rawdata']                = op.join(param['path'], 'eeg/s01/wmConfidence_s01_12062019.cdt')
        param['rawset']                 = op.join(param['path'], 'eeg/s01/wmConfidence_s01_12062019.set')
        param['rawcleaned']             = op.join(param['path'], 'eeg/s01/wmConfidence_s01_icacleaned_raw.fif')
        param['raweyes']                = op.join(param['path'], 'eyes/s01/WMCS01.asc')
        param['cleanedeyes']            = op.join(param['path'], 'eyes/s01/wmConfidence_s01_preprocessed.pickle')
        param['behaviour_blinkchecked'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S01_blinkchecked_preprocessed.csv')
        
    if subject['id'] == 2:
        param['subid']                  = 's02'
        param['behaviour']              = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S02_allData_preprocessed.csv')
        param['rawdata']                = op.join(param['path'], 'eeg/s02/wmConfidence_s02_12062019.cdt')
        param['rawset']                 = op.join(param['path'], 'eeg/s02/wmConfidence_s02_12062019.set')
        param['rawcleaned']             = op.join(param['path'], 'eeg/s02/wmConfidence_s02_icacleaned_raw.fif')
        param['raweyes']                = op.join(param['path'], 'eyes/s02/WMCS02.asc')
        param['cleanedeyes']            = op.join(param['path'], 'eyes/s02/wmConfidence_s02_preprocessed.pickle')
        param['behaviour_blinkchecked'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S02_blinkchecked_preprocessed.csv')
        
    if subject['id'] == 3:
        param['subid']                  = 's03'
        param['behaviour_sess1']        = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S03_allData_preprocessed.csv')
        param['rawdata_sess1']          = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_24062019.cdt')
        param['rawset_sess1']           = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_24062019.set')
        param['rawcleaned_sess1']       = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_icacleaned_raw.fif')
        param['raweyes_sess1']          = op.join(param['path'], 'eyes/s03/WMCS03a.asc')
        param['cleanedeyes_sess1']      = op.join(param['path'], 'eyes/s03/wmConfidence_s03a_preprocessed.pickle')
        param['behaviour_blinkchecked'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S03_blinkchecked_preprocessed.csv')
        
    
    if subject['id'] == 4:
        param['subid']                   = 's04'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S04a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_24062019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_24062019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes/s04/WMCS04a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes/s04/wmConfidence_s04a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S04a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S04b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_24062019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_24062019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes/s04/WMCS04b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes/s04/wmConfidence_s04b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S04b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = []
        param['badchans2'] = []
        
    if subject['id'] == 5:
        param['subid']                   = 's05'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S05a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_25062019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_25062019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes/s05/WMCS05a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes/s05/wmConfidence_s05a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S05a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S05b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_25062019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_25062019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes/s05/WMCS05b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes/s05/wmConfidence_s05b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S05b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = []
        param['badchans2'] = []
        
    if subject['id'] == 6:
        param['subid']                   = 's06'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S06a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_26062019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_26062019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes/s06/WMCS06a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes/s06/wmConfidence_s06a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S06a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S06b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_26062019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_26062019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes/s06/WMCS06b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes/s06/wmConfidence_s06b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S06b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = []
        param['badchans2'] = []
        
    if subject['id'] == 7:
        param['subid']                   = 's07'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S07a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_26062019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_26062019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes/s07/WMCS07a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes/s07/wmConfidence_s07a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S07a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S07b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_26062019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_26062019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes/s07/WMCS07b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes/s07/wmConfidence_s07b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S07b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = []
        param['badchans2'] = []
        
    if subject['id'] == 8:
        param['subid']                   = 's08'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S08a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s08a_17072019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s08a_17072019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s08a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS08a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s08a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S08a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S08b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s08b_17072019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s08b_17072019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s08b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS08b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s08b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S08b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = []
        param['badchans2'] = []
        
    if subject['id'] == 9:
        param['subid']                   = 's09'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S09a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s09a_18072019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s09a_18072019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s09a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS09a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s09a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S09a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S09b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s09b_18072019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s09b_18072019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s09b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS09b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s09b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S09b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = ['T8']
        param['badchans2'] = ['T8']
        
    if subject['id'] == 10:
        param['subid']                   = 's10'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S10_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s10a_18072019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s10a_18072019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s10a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS10a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s10a_preprocessed.pickle')
        param['behaviour_blinkchecked'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S10_blinkchecked_preprocessed.csv')
      
    if subject['id'] == 11:
        param['subid']                   = 's11'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S11a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s11a_02092019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s11a_02092019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s11a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS11a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s11a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S11a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S11b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s11b_02092019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s11b_02092019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s11b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS11b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s11b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S11b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = ['T8']
        param['badchans2'] = ['T8']
        
    if subject['id'] == 12:
        param['subid']                   = 's12'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S12a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s12a_03092019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s12a_03092019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s12a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS12a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s12a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S12a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S12b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s12b_03092019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s12b_03092019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s12b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS12b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s12b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S12b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = ['T7']
        param['badchans2'] = ['T7']
        
        
    if subject['id'] == 13:
        param['subid']                   = 's13'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S13a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s13a_04092019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s13a_04092019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s13a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS13a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s13a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S13a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S13b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s13b_04092019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s13b_04092019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s13b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS13b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s13b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S13b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = ['T8']
        param['badchans2'] = ['T8']
        
    if subject['id'] == 14:
        param['subid']                   = 's14'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S14a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s14a_04092019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s14a_04092019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s14a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS14a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s14a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S14a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S14b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s14b_04092019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s14b_04092019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s14b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS14b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s14b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S14b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = ['T7', 'TP8']
        param['badchans2'] = ['T7', 'TP8']
        
        
    if subject['id'] == 15:
        param['subid']                   = 's15'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S15a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s15a_09092019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s15a_09092019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s15a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS15a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s15a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S15a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S15b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s15b_09092019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s15b_09092019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s15b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS15b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s15b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S15b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = []
        param['badchans2'] = []
        
    if subject['id'] == 16:
        param['subid']                   = 's16'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S16a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s16a_16092019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s16a_16092019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s16a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS16a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s16a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S16a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S16b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s16b_16092019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s16b_16092019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s16b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS16b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s16b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S16b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = []
        param['badchans2'] = []
        
    if subject['id'] == 17:
        param['subid']                   = 's17'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S17a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s17a_07102019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s17a_07102019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s17a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS17a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s17a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S17a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S17b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s17b_07102019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s17b_07102019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s17b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS17b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s17b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S17b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = ['TP7', 'TP8']
        param['badchans2'] = ['TP7', 'TP8']
        
    if subject['id'] == 18:
        param['subid']                   = 's18'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S18a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s18a_10102019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s18a_10102019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s18a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS18a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s18a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S18a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S18b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s18b_10102019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s18b_10102019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s18b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS18b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s18b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S18b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = ['TP7']
        param['badchans2'] = ['TP7']
        
    if subject['id'] == 19:
        param['subid']                   = 's19'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S19_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s19a_21102019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s19a_21102019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s19a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS19a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s19a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S19_blinkchecked_preprocessed.csv')

#        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S19b_allData_preprocessed.csv')
#        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s18b_10102019.cdt')
#        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s18b_10102019.set')
#        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s18b_icacleaned_raw.fif')
#        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS19b.asc')
#        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s19b_preprocessed.pickle')
#        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S19b_blinkchecked_preprocessed.csv')
    
    if subject['id'] == 20:
        param['subid']                   = 's20'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S20a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s20a_23102019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s20a_23102019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s20a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS20a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s20a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S20a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S20b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s20b_23102019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s20b_23102019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s20b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS20b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s20b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S20b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = []
        param['badchans2'] = []
        
    if subject['id'] == 21:
        param['subid']                   = 's21'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S21a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s21a_24102019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s21a_24102019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s21a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS21a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s21a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S21a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S21b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s21b_24102019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s21b_24102019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s21b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS21b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s21b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S21b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = ['T8', 'T7', 'F6', 'FP2']
        param['badchans2'] = ['T8', 'T7', 'F6', 'FP2']
        
    if subject['id'] == 22:
        param['subid']                   = 's22'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S22a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s22a_30102019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s22a_30102019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s22a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS22a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s22a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S22a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S22b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s22b_30102019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s22b_30102019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s22b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS22b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s22b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S22b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = []
        param['badchans2'] = []
        
    if subject['id'] == 23:
        param['subid']                   = 's23'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S23a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s23a_31102019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s23a_31102019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s23a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS23a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s23a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S23a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S23b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s23b_31102019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s23b_31102019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s23b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS23b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s23b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S23b_blinkchecked_preprocessed.csv')
    
        param['badchans1'] = []
        param['badchans2'] = []
    
        
    if subject['id'] == 24:
        param['subid']                   = 's24'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S24a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s24a_04112019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s24a_04112019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s24a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS24a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s24a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S24a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S24b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s24b_04112019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s24b_04112019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s24b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS24b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s24b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S24b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = ['FT8', 'T7', 'T8']
        param['badchans2'] = ['FT8', 'T7', 'T8']
        
    if subject['id'] == 25:
        param['subid']                   = 's25'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S25a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s25a_05112019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s25a_05112019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s25a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS25a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s25a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S25a_blinkchecked_preprocessed.csv')

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S25b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s25b_05112019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s25b_05112019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s25b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS25b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s25b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S25b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = ['T8', 'TP7']
        param['badchans2'] = ['T8', 'TP7']
        
    if subject['id'] == 26:
        param['subid']                   = 's26'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S26a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s26a_25112019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s26a_25112019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s26a_icacleaned_raw.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS26a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s26a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S26a_blinkchecked_preprocessed.csv')
        

        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S26b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s26b_25112019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s26b_25112019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s26b_icacleaned_raw.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes', param['subid'], 'WMCS26b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes', param['subid'], 'wmConfidence_s26b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S26b_blinkchecked_preprocessed.csv')
        
        param['badchans1'] = ['T8', 'F3', 'F5', 'T7', 'FT8']
        param['badchans2'] = ['T8', 'F3', 'F5', 'T7', 'FT8']
        
    #these are coded in a way that we don't have to vary across ppts (it's consistent) so we don't need to repeat per subject
    param['arraylocked']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_arraylocked-epo.fif')
    param['arraylocked_tfr']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_arraylocked-tfr.h5')
    param['arraylocked_tfr_meta']   = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_arraylocked_metadata.csv')
    param['cuelocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-epo.fif')
    param['cuelocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-tfr.h5')
    param['cuelocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_metadata.csv')
    param['probelocked']            = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_probelocked-epo.fif')
    param['probelocked_tfr']        = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_probelocked-tfr.h5')
    param['probelocked_tfr_meta']   = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_probelocked_metadata.csv')
    param['resplocked']             = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_resplocked-epo.fif')
    param['resplocked_tfr']         = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_resplocked-tfr.h5')
    param['resplocked_tfr_meta']    = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_resplocked_metadata.csv')
    param['fblocked']               = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-epo.fif')
    param['fblocked_tfr']           = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-tfr.h5')
    param['fblocked_tfr_meta']      = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked_metadata.csv')

    
    return param

def gesd(x, alpha = .05, p_out = .1, outlier_side = 0):
    import numpy as np
    import scipy.stats
    import copy
    
    '''
    Detect outliers using Generalizes ESD test
    based on the code from Romesh Abeysuriya implementation for OSL
      
    Inputs:
    - x : Data set containing outliers - should be a np.array 
    - alpha : Significance level to detect at (default = .05)
    - p_out : percent of max number of outliers to detect (default = 10% of data set)
    - outlier_side : Specify sidedness of the test
        - outlier_side = -1 -> outliers are all smaller
        - outlier_side = 0 -> outliers could be small/negative or large/positive (default)
        - outlier_side = 1 -> outliers are all larger
        
    Outputs
    - idx : Logicial array with True wherever a sample is an outlier
    - x2 : input array with outliers removed
    
    For details about the method, see
    B. Rosner (1983). Percentage Points for a Generalized ESD Many-outlier Procedure, Technometrics 25(2), pp. 165-172.
    http://www.jstor.org/stable/1268549?seq=1
    '''

    if outlier_side == 0:
        alpha = alpha/2
    
    
    if type(x) != np.ndarray:
        x = np.asarray(x)

    n_out = int(np.ceil(len(x)*p_out))

    if any(~np.isfinite(x)):
        #Need to find outliers only in non-finite x
        y = np.where(np.isfinite(x))[0] # these are the indexes of x that are finite
        idx1, x2 = gesd(x[np.isfinite(x)], alpha, n_out, outlier_side)
        # idx1 has the indexes of y which were marked as outliers
        # the value of y contains the corresponding indexes of x that are outliers
        idx = [False] * len(x)
        idx[y[idx1]] = True

    n      = len(x)
    temp   = x
    R      = np.zeros((1, n_out))[0]
    rm_idx = copy.deepcopy(R)
    lam    = copy.deepcopy(R)

    for j in range(0,int(n_out)):
        i = j+1
        if outlier_side == -1:
            rm_idx[j] = np.nanargmin(temp)
            sample    = np.nanmin(temp)
            R[j]      = np.nanmean(temp) - sample
        elif outlier_side == 0:
            rm_idx[j] = int(np.nanargmax(abs(temp-np.nanmean(temp))))
            R[j]      = np.nanmax(abs(temp-np.nanmean(temp)))
        elif outlier_side == 1: 
            rm_idx[j] = np.nanargmax(temp)
            sample    = np.nanmax(temp)
            R[j]      = sample - np.nanmean(temp)
        
        R[j] = R[j] / np.nanstd(temp)
        temp[int(rm_idx[j])] = np.nan
        
        p = 1-alpha/(n-i+1)
        t = scipy.stats.t.ppf(p,n-i-1)
        lam[j] = ((n-i) * t) / (np.sqrt((n-i-1+t**2)*(n-i+1)))
    
    #And return a logical array of outliers
    idx = np.zeros((1,n))[0]
    idx[np.asarray(rm_idx[range(0,np.max(np.where(R>lam))+1)],int)] = np.nan
    idx = ~np.isfinite(idx)
    
    x2 = x[~idx]

        
    return idx, x2 


def plot_AR(epochs, method = 'gesd', zthreshold = 1.5, p_out = .1, alpha = .05, outlier_side = 1):
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import scipy.stats
    from matplotlib import pyplot as plt

    #Get data, variance, number of trials, and number of channels
    dat     = epochs.get_data()
    var     = np.var(dat, 2)
    ntrials = np.shape(dat)[0]
    nchan   = len(epochs.ch_names)

    #set up the axis for the plots
    x_epos  = range(1,ntrials+1)
    y_epos  = np.mean(var,1)
    y_chans = range(1,nchan+1)
    x_chans = np.mean(var,0)

    #scale the variances
    y_epos  = [x * 10**6 for x in y_epos]
    x_chans = [x * 10**6 for x in x_chans]

    #Get the zScore
    zVar = scipy.stats.zscore(y_epos)

    #save everything in the dataFrame
    df_epos           = pd.DataFrame({'var': y_epos, 'epochs': x_epos, 'zVar': zVar})
    df_chans          = pd.DataFrame({'var': x_chans, 'chans': y_chans})
    
    # Apply the artefact rejection method
    if method == 'gesd':
        try:
            idx,x2            = gesd(y_epos, p_out=p_out, alpha=alpha, outlier_side=outlier_side) #use the gesd to find outliers (idx is the index of the outlier trials)
        except:
            print('***** gesd failed here, no trials removed *****')
            idx = []
        keepTrials        = np.ones((1,ntrials))[0]
        keepTrials[idx]   = 0
        title = 'Generalized ESD test (alpha=' + str(alpha) + ', p_out=' + str(p_out) + ', outlier_side=' + str(outlier_side) + ')'
    elif method == 'zScore':
        keepTrials        = np.where(df_epos['zVar'] > zthreshold, 0, 1)
        title = 'ZVarience threshold of ' + str(zthreshold)
    elif method == 'none':
        title = 'no additional artefact rejection '
        keepTrials        = np.ones((1,ntrials))[0]
    
    df_epos['keepTrial'] = keepTrials
    df_keeps = df_epos[df_epos['keepTrial'] == 1]
    print(str(ntrials - len(df_keeps)) + ' trials discarded')
    
    # get the clean data
    keep_idx    = np.asarray(np.where(keepTrials),int)
    clean_dat    = np.squeeze(dat[keep_idx])
    
    #recalculate the var for chan
    clean_var    = np.var(clean_dat, 2)
    x_chans_c    = np.mean(clean_var,0)
    x_chans_c    = [x * 10**6 for x in x_chans_c]

    df_chans_c   = pd.DataFrame({'var': x_chans_c, 'chans': y_chans})
    
    
    # Plot everything
    fig, axis = plt.subplots(2, 2, figsize=(12, 12))
    axis[0,0].set_ylim([0, max(y_epos) + min(y_epos)*2])
    axis[0,1].set_xlim([0, max(x_chans)+ min(x_chans)*2])
    axis[1,0].set_ylim([0, max(df_keeps['var'])+ min(df_keeps['var'])*2])
    axis[1,1].set_xlim([0, max(x_chans_c)+ min(x_chans_c)*2])

    axis[0,0].set_title(title)
    sns.scatterplot(x = 'epochs', y = 'var', hue = 'keepTrial', hue_order = [1,0], ax = axis[0,0], data = df_epos)
    sns.scatterplot(x = 'var', y = 'chans', ax = axis[0,1], data = df_chans)
    sns.scatterplot(x = 'epochs', y = 'var', ax = axis[1,0], data =df_keeps)
    sns.scatterplot(x = 'var', y = 'chans', ax = axis[1,1], data = df_chans_c)
    
    
    
    return axis, keep_idx 


def toverparam(alldata):
    import scipy as sp
    import numpy as np
    from scipy import stats
    '''
    function to conduct a t-test over all samples of multiple subject data, prevents repeating the same thing over and over again
    
    input data is a list of individual subject objects
    
    NB this doesn't really care whether it's betas or tstats, it just does shit
    
    output = the t stats over the parameter given (i..e goes from having a dimension of length nsubs to not having it as it runs across subs)
    '''
    
    ndata = np.array(len(alldata))
    indiv_data_shape = np.array(alldata[0].data.shape)
    newshape = np.append(ndata, indiv_data_shape)
    tmp = np.empty(shape = newshape)

    for i in range(ndata):
        tmp[i] = alldata[i].data
    toverparam = sp.stats.ttest_1samp(tmp, popmean = 0, axis = 0)
    
    return toverparam[0]


def smooth(signal, twin , method = 'boxcar'):
    '''
    
    function to smooth a signal. defaults to a 50ms boxcar smoothing (so quite small), just smooths out some of the tremor in the trace signals to clean it a bit
    can change the following parameters:
        
    twin    -- number of samples (if 1KHz sampling rate, then ms) for the window
    method  -- type of smoothing (defaults to a boxcar smoothing) - defaults to a boxcar    
    '''
    import scipy as sp
    from scipy import signal
    import numpy as np
    
    
    if method == 'boxcar':
        #set up the boxcar
        filt = sp.signal.windows.boxcar(twin)
    
    #smooth the signal
    if method == 'boxcar':
        smoothed_signal = np.convolve(filt/filt.sum(), signal, mode = 'same')
    
    return smoothed_signal

def runclustertest_epochs(data, contrast_name, channels, tmin = None, tmax = None, gauss_smoothing = None, out_type = 'indices', n_permutations = 'Default', n_jobs = 1):
    '''
    func to run cluster permutation tests on voltage data (epochs)
    data = data object. dictionary where each key is a contrast name, and inside it is a list (of length nsubs) of Evoked objects
    contrast_name = name of the contrast you want to run the test on
    channels = list. list of channels you want to average over. if one channel only, obviously no averaging across channels. still needs to be list
    tmin, tmax = if you want to restrict permutation tests to a time window, do it here
    gauss_smoothing = width (sigma) of a gaussian smoothing that is performed on the single subject data prior to running the test. if None (default) - no smoothing.
                      NOTE: the time width of this smoothing depends on your sampling frequency so make sure to use this properly
    out_type = specify output type. default to indices, can set to mask if you really want
    '''
    import scipy as sp
    from scipy import ndimage
    from copy import deepcopy
    
    dat       = deepcopy(data[contrast_name])
    nsubs     = len(dat)
    times    = deepcopy(dat[0]).crop(tmin=tmin, tmax=tmax).times
    cludat    = np.empty(shape = (nsubs, 1, times.size)) #specify 1 because we're going to average across channels anyway
    
    for i in range(nsubs):
        tmp = deepcopy(dat[i])
        tmp.crop(tmin=tmin, tmax=tmax).pick_channels(channels) #select time window and channels we want
        if gauss_smoothing != None:
            cludat[i,:,:] = sp.ndimage.gaussian_filter1d(np.nanmean(tmp.data, axis=0), sigma = gauss_smoothing)
        else:
            cludat[i,:,:] = np.nanmean(tmp.data, axis=0) #average across channels
    if n_permutations != 'Default':
        t, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(cludat, out_type=out_type, n_permutations = n_permutations, n_jobs = n_jobs)
    else:
        t, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(cludat, out_type=out_type, n_jobs = n_jobs)
    return t, clusters, cluster_pv, H0


def runclustertest_tfr(data, contrast_name, channels, contra_channels, ipsi_channels, tmin = None, tmax = None,  out_type = 'mask', n_permutations = 'Default'):
    '''
    func to run cluster permutation tests on voltage data (epochs)
    data = data object. dictionary where each key is a contrast name, and inside it is a list (of length nsubs) of Evoked objects
    contrast_name = name of the contrast you want to run the test on
    channels = list. list of channels you want to average over. if one channel only, obviously no averaging across channels. still needs to be list
    tmin, tmax = if you want to restrict permutation tests to a time window, do it here
    gauss_smoothing = width (sigma) of a gaussian smoothing that is performed on the single subject data prior to running the test. if None (default) - no smoothing.
                      NOTE: the time width of this smoothing depends on your sampling frequency so make sure to use this properly
    out_type = specify output type. default to indices, can set to mask if you really want
    
    if you dont want to do any lateralisation, set contra_channels and ipsi_channels to None
    if you want to do lateralisation only, and not just a certain few electrodes, set 'channels' to None and feed in contra/ipsi channels
    '''
    import scipy as sp
    from copy import deepcopy
    
    dat       = deepcopy(data[contrast_name])
    nsubs     = len(dat)
    times    = deepcopy(dat[0]).crop(tmin=tmin, tmax=tmax).times
    freqs    = deepcopy(dat[0]).freqs
    cludat    = np.empty(shape = (nsubs, freqs.size, times.size)) #specify 1 because we're going to average across channels anyway
    
    for i in range(nsubs):
        tmp = deepcopy(dat[i])
        tmp.crop(tmin=tmin, tmax=tmax)
        
        if contra_channels != None:
            cvsi = get_cvsi_tfrs(tmp, contrachans = contra_channels, ipsichans = ipsi_channels)
#            tmp_contra  = deepcopy(tmp).pick_channels(contra_channels).data
#            tmp_ipsi = deepcopy(tmp).pick_channels(ipsi_channels).data
            cludat[i,:,:] = cvsi
        elif contra_channels == None and channels != None:
            chandat = deepcopy(tmp).pick_channels(channels).data
            if len(channels) > 1:
                chandat = np.nanmean(chandat, 0)
            cludat[i,:,:] = chandat
#        cludat[i,:,:] = tmp_cvsi

    if n_permutations != 'Default':
        t, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(cludat, out_type=out_type, n_permutations = n_permutations)
    else:
        t, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(cludat, out_type=out_type)
    return t, clusters, cluster_pv, H0

def get_cvsi_tfrs(data, contrachans, ipsichans):
    import numpy as np
    from copy import deepcopy
    
    contra = deepcopy(data).pick_channels(contrachans).data
    ipsi = deepcopy(data).pick_channels(ipsichans).data
    
    if len(contrachans) > 1:
        contra = np.nanmean(contra, 0)
    if len(ipsichans) > 1:
        ipsi = np.nanmean(ipsi, 0)
    
    cvsi = np.subtract(contra, ipsi)
    return cvsi
        
    




def nanzscore(vector, zero_out_nans = True):
            '''
            zscore a vector ignoring nans
            optionally can set nans to 0 afterwards. useful for regressors
            '''
            vector = np.divide(np.subtract(vector, np.nanmean(vector)), np.nanstd(vector))
            if zero_out_nans:
                vector = np.where(np.isnan(vector), 0, vector)
            
            return vector

def flip_tfrdata(data, layout = 'easycapM1', compute_lateralisation = True, lateralisation = 'leftvsright'):
    '''
    
    layout -- easycapM1 is the standard for EEG analyses in our lab, so will assume a certain layout structure for the data
    '''
    if layout == 'easycapM1':
        #assume the channel names for the data in this structure
        chnames =     np.array([       'FP1', 'FPZ', 'FP2', 
                                'AF7', 'AF3', 'AFZ', 'AF4', 'AF8', 
                  'F7',  'F5',   'F3',  'F1',  'FZ',  'F2',  'F4',  'F6',  'F8',
                 'FT7', 'FC5',  'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
                  'T7',  'C5',   'C3',  'C1',  'CZ',  'C2',  'C4',  'C6',  'T8',
                 'TP7', 'CP5',  'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                  'P7',  'P5',   'P3',  'P1',  'PZ',  'P2',  'P4',  'P6',  'P8',
                 'PO7',         'PO3',        'POZ',        'PO4', 'PO8',
                                        'O1',  'OZ',  'O2'])
    
        #these are the IDs for the channel names
        chids =         np.array([             1,  2,  3,
                                           4,  5,  6,  7,  8,
                                   9, 10, 11, 12, 13, 14, 15, 16, 17,
                                  18, 19, 20, 21, 22, 23, 24, 25, 26,
                                  27, 28, 29, 30, 31, 32, 33, 34, 35,
                                  36, 37, 38, 39, 40, 41, 42, 43, 44,
                                  45, 46, 47, 48, 49, 50, 51, 52, 53,
                                          54, 55, 56, 57, 58,
                                              59, 60, 61
                                                  ])
        
        chids = np.subtract(chids,1) #because of 0 indexing ...
        
        #this is what the flipped channel ids would look like
        flipids =       np.array([             3,  2,  1,
                                           8,  7,  6,  5,  4,
                                  17, 16, 15, 14, 13, 12, 11, 10,  9,
                                  26, 25, 24, 23, 22, 21, 20, 19, 18,
                                  35, 34, 33, 32, 31, 30, 29, 28, 27,
                                  44, 43, 42, 41, 40, 39, 38, 37, 36,
                                  53, 52, 51, 50, 49, 48, 47, 46, 45,
                                          58, 57, 56, 55, 54,
                                              61, 60, 59
                                                  ])
        #these are the channel ids for left hand side channels in the un-flipped data
    #    lhs_chanids  = np.array([1,4,5,9,10,11,12,18,19,20,21,27,28,29,30,36,37,38,39,45,46,47,48,54,55,59])
    
    #    midline_chaninds = np.array([2,6,13,22,31,40,49,56,60]) #the midline channels, which just have themselves subtracted from them (flipped(midline) = midline)
        flipids = np.subtract(flipids,1)
        flippednames = chnames[flipids]
        
        renaming_mapping = dict()
        for i in range(len(chnames)):
            renaming_mapping[chnames[i]] = flippednames[i]
        
        flipids = flipids
        flipped_data = deepcopy(data)
        flipped_data.data = flipped_data.data[flipids,:,:] #flip the data leave the channels where they are
    
    
    if compute_lateralisation: #do the subtraction here
        tmp_data = deepcopy(data)
        
#        if lateralisation == 'leftvsright':
        tmp_data.data = np.subtract(data.data, flipped_data.data)
#        elif lateralisation == 'rightvsleft':
#            tmp_data.data = np.subtract(flipped_data.data, data.data)
        
        flipped_data.data = tmp_data.data #set this output
    
    return flipped_data




