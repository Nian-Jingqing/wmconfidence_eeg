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
        
    if subject['id'] == 3:
        param['subid']          = 's03'
        param['behaviour_sess1']      = op.join(param['path'], 'datafiles/s03/wmConfidence_S03a_allData_preprocessed.csv')
        param['rawdata_sess1']        = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_24062019.cdt')
        param['rawset_sess1']         = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_24062019.set')
        param['rawcleaned_sess1']     = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_icacleaned_raw.fif')
        param['cuelock_noref_sess1']  = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_cuelock_noref-epo.fif')
        param['cuelock_car_sess1']    = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_cuelock_car-epo.fif')
        param['cuelock_mast_sess1']   = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_cuelock_mast-epo.fif')
        param['raweyes_sess1']        = op.join(param['path'], 'eyes/s03/WMCS03a.asc')
        param['cleanedeyes_sess1']    = op.join(param['path'], 'eyes/s03/wmConfidence_s03a_preprocessed.pickle')
    
    if subject['id'] == 4:
        param['subid']                = 's04'
        param['behaviour_sess1']      = op.join(param['path'], 'datafiles/s04/wmConfidence_S04a_allData_preprocessed.csv')
        param['rawdata_sess1']        = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_24062019.cdt')
        param['rawset_sess1']         = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_24062019.set')
        param['rawcleaned_sess1']     = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_icacleaned_raw.fif')
        param['cuelock_noref_sess1']  = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_cuelock_noref-epo.fif')
        param['cuelock_car_sess1']    = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_cuelock_car-epo.fif')
        param['cuelock_mast_sess1']   = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_cuelock_mast-epo.fif')
        param['raweyes_sess1']        = op.join(param['path'], 'eyes/s04/WMCS04a.asc')
        param['cleanedeyes_sess1']    = op.join(param['path'], 'eyes/s04/wmConfidence_s04a_preprocessed.pickle')
        
        param['behaviour_sess2']      = op.join(param['path'], 'datafiles/s04/wmConfidence_S04b_allData_preprocessed.csv')
        param['rawdata_sess2']        = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_24062019.cdt')
        param['rawset_sess2']         = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_24062019.set')
        param['rawcleaned_sess2']     = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_icacleaned_raw.fif')
        param['cuelock_noref_sess2']  = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_cuelock_noref-epo.fif')
        param['cuelock_car_sess2']    = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_cuelock_car-epo.fif')
        param['cuelock_mast_sess2']   = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_cuelock_mast-epo.fif')
        param['raweyes_sess2']        = op.join(param['path'], 'eyes/s04/WMCS04b.asc')
        param['cleanedeyes_sess2']    = op.join(param['path'], 'eyes/s04/wmConfidence_s04b_preprocessed.pickle')
        
    if subject['id'] == 5:
        param['subid']                = 's05'
        param['behaviour_sess1']      = op.join(param['path'], 'datafiles/s05/wmConfidence_S05a_allData_preprocessed.csv')
        param['rawdata_sess1']        = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_25062019.cdt')
        param['rawset_sess1']         = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_25062019.set')
        param['rawcleaned_sess1']     = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_icacleaned_raw.fif')
        param['cuelock_noref_sess1']  = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_cuelock_noref-epo.fif')
        param['cuelock_car_sess1']    = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_cuelock_car-epo.fif')
        param['cuelock_mast_sess1']   = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_cuelock_mast-epo.fif')
        param['raweyes_sess1']        = op.join(param['path'], 'eyes/s05/WMCS05a.asc')
        param['cleanedeyes_sess1']    = op.join(param['path'], 'eyes/s05/wmConfidence_s05a_preprocessed.pickle')
        
        param['behaviour_sess2']      = op.join(param['path'], 'datafiles/s05/wmConfidence_S05b_allData_preprocessed.csv')
        param['rawdata_sess2']        = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_25062019.cdt')
        param['rawset_sess2']         = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_25062019.set')
        param['rawcleaned_sess2']     = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_icacleaned_raw.fif')
        param['cuelock_noref_sess2']  = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_cuelock_noref-epo.fif')
        param['cuelock_car_sess2']    = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_cuelock_car-epo.fif')
        param['cuelock_mast_sess2']   = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_cuelock_mast-epo.fif')
        param['raweyes_sess2']        = op.join(param['path'], 'eyes/s05/WMCS05b.asc')
        param['cleanedeyes_sess2']    = op.join(param['path'], 'eyes/s05/wmConfidence_s05b_preprocessed.pickle')
        
    if subject['id'] == 6:
        param['subid']                = 's06'
        param['behaviour_sess1']      = op.join(param['path'], 'datafiles/s06/wmConfidence_S06a_allData_preprocessed.csv')
        param['rawdata_sess1']        = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_26062019.cdt')
        param['rawset_sess1']         = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_26062019.set')
        param['rawcleaned_sess1']     = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_icacleaned_raw.fif')
        param['cuelock_noref_sess1']  = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_cuelock_noref-epo.fif')
        param['cuelock_car_sess1']    = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_cuelock_car-epo.fif')
        param['cuelock_mast_sess1']   = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_cuelock_mast-epo.fif')
        param['raweyes_sess1']        = op.join(param['path'], 'eyes/s06/WMCS06a.asc')
        param['cleanedeyes_sess1']    = op.join(param['path'], 'eyes/s06/wmConfidence_s06a_preprocessed.pickle')
        
        param['behaviour_sess2']      = op.join(param['path'], 'datafiles/s06/wmConfidence_S06b_allData_preprocessed.csv')
        param['rawdata_sess2']        = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_26062019.cdt')
        param['rawset_sess2']         = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_26062019.set')
        param['rawcleaned_sess2']     = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_icacleaned_raw.fif')
        param['cuelock_noref_sess2']  = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_cuelock_noref-epo.fif')
        param['cuelock_car_sess2']    = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_cuelock_car-epo.fif')
        param['cuelock_mast_sess2']   = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_cuelock_mast-epo.fif')
        param['raweyes_sess2']        = op.join(param['path'], 'eyes/s06/WMCS06b.asc')
        param['cleanedeyes_sess2']    = op.join(param['path'], 'eyes/s06/wmConfidence_s06b_preprocessed.pickle')
        
        
    if subject['id'] == 7:
        param['subid']                = 's07'
        param['behaviour_sess1']      = op.join(param['path'], 'datafiles/s07/wmConfidence_S06a_allData_preprocessed.csv')
        param['rawdata_sess1']        = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_26062019.cdt')
        param['rawset_sess1']         = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_26062019.set')
        param['rawcleaned_sess1']     = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_icacleaned_raw.fif')
        param['cuelock_noref_sess1']  = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_cuelock_noref-epo.fif')
        param['cuelock_car_sess1']    = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_cuelock_car-epo.fif')
        param['cuelock_mast_sess1']   = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_cuelock_mast-epo.fif')
        param['raweyes_sess1']        = op.join(param['path'], 'eyes/s07/WMCS07a.asc')
        param['cleanedeyes_sess1']    = op.join(param['path'], 'eyes/s07/wmConfidence_s07a_preprocessed.pickle')
        
        param['behaviour_sess2']      = op.join(param['path'], 'datafiles/s07/wmConfidence_S07b_allData_preprocessed.csv')
        param['rawdata_sess2']        = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_26062019.cdt')
        param['rawset_sess2']         = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_26062019.set')
        param['rawcleaned_sess2']     = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_icacleaned_raw.fif')
        param['cuelock_noref_sess2']  = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_cuelock_noref-epo.fif')
        param['cuelock_car_sess2']    = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_cuelock_car-epo.fif')
        param['cuelock_mast_sess2']   = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_cuelock_mast-epo.fif')
        param['raweyes_sess2']        = op.join(param['path'], 'eyes/s07/WMCS07b.asc')
        param['cleanedeyes_sess2']    = op.join(param['path'], 'eyes/s07/wmConfidence_s07b_preprocessed.pickle')
        
        
        
        
    return param

