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
        param['subid']                  = 's01'
        param['behaviour']              = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S01_allData_preprocessed.csv')
        param['rawdata']                = op.join(param['path'], 'eeg/s01/wmConfidence_s01_12062019.cdt')
        param['rawset']                 = op.join(param['path'], 'eeg/s01/wmConfidence_s01_12062019.set')
        param['rawcleaned']             = op.join(param['path'], 'eeg/s01/wmConfidence_s01_icacleaned_raw.fif')
        param['cuelock_noref']          = op.join(param['path'], 'eeg/s01/wmConfidence_s01_cuelock_noref-epo.fif')
        param['cuelock_car']            = op.join(param['path'], 'eeg/s01/wmConfidence_s01_cuelock_car-epo.fif')
        param['cuelock_mast']           = op.join(param['path'], 'eeg/s01/wmConfidence_s01_cuelock_mast-epo.fif')
        param['raweyes']                = op.join(param['path'], 'eyes/s01/WMCS01.asc')
        param['cleanedeyes']            = op.join(param['path'], 'eyes/s01/wmConfidence_s01_preprocessed.pickle')
        param['behaviour_blinkchecked'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S01_blinkchecked_preprocessed.csv')
        param['cuelocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-epo.fif')
        param['cuelocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-tfr.h5')
        param['cuelocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_metadata.csv')
        param['cuelocked_tfr_lvsr']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_lvsr-tfr.h5')

        param['fblocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-epo.fif')
        param['fblocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-tfr.h5')
        param['fblocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked_metadata.csv')
        
    if subject['id'] == 2:
        param['subid']                  = 's02'
        param['behaviour']              = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S02_allData_preprocessed.csv')
        param['rawdata']                = op.join(param['path'], 'eeg/s02/wmConfidence_s02_12062019.cdt')
        param['rawset']                 = op.join(param['path'], 'eeg/s02/wmConfidence_s02_12062019.set')
        param['rawcleaned']             = op.join(param['path'], 'eeg/s02/wmConfidence_s02_icacleaned_raw.fif')
        param['cuelock_noref']          = op.join(param['path'], 'eeg/s02/wmConfidence_s02_cuelock_noref-epo.fif')
        param['cuelock_car']            = op.join(param['path'], 'eeg/s02/wmConfidence_s02_cuelock_car-epo.fif')
        param['cuelock_mast']           = op.join(param['path'], 'eeg/s02/wmConfidence_s02_cuelock_mast-epo.fif')
        param['raweyes']                = op.join(param['path'], 'eyes/s02/WMCS02.asc')
        param['cleanedeyes']            = op.join(param['path'], 'eyes/s02/wmConfidence_s02_preprocessed.pickle')
        param['behaviour_blinkchecked'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S02_blinkchecked_preprocessed.csv')
        param['cuelocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-epo.fif')
        param['cuelocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-tfr.h5')
        param['cuelocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_metadata.csv')
        param['cuelocked_tfr_lvsr']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_lvsr-tfr.h5')

        param['fblocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-epo.fif')
        param['fblocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-tfr.h5')
        param['fblocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked_metadata.csv')
        
    if subject['id'] == 3:
        param['subid']                  = 's03'
        param['behaviour_sess1']        = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S03_allData_preprocessed.csv')
        param['rawdata_sess1']          = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_24062019.cdt')
        param['rawset_sess1']           = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_24062019.set')
        param['rawcleaned_sess1']       = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_icacleaned_raw.fif')
        param['cuelock_noref_sess1']    = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_cuelock_noref-epo.fif')
        param['cuelock_car_sess1']      = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_cuelock_car-epo.fif')
        param['cuelock_mast_sess1']     = op.join(param['path'], 'eeg/s03/wmConfidence_s03a_cuelock_mast-epo.fif')
        param['raweyes_sess1']          = op.join(param['path'], 'eyes/s03/WMCS03a.asc')
        param['cleanedeyes_sess1']      = op.join(param['path'], 'eyes/s03/wmConfidence_s03a_preprocessed.pickle')
        param['behaviour_blinkchecked'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S03_blinkchecked_preprocessed.csv')
        param['cuelocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-epo.fif')
        param['cuelocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-tfr.h5')
        param['cuelocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_metadata.csv')
        param['cuelocked_tfr_lvsr']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_lvsr-tfr.h5')

        param['fblocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-epo.fif')
        param['fblocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-tfr.h5')
        param['fblocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked_metadata.csv')
    
    if subject['id'] == 4:
        param['subid']                   = 's04'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S04a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_24062019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_24062019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_icacleaned_raw.fif')
        param['cuelock_noref_sess1']     = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_cuelock_noref-epo.fif')
        param['cuelock_car_sess1']       = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_cuelock_car-epo.fif')
        param['cuelock_mast_sess1']      = op.join(param['path'], 'eeg/s04/wmConfidence_s04a_cuelock_mast-epo.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes/s04/WMCS04a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes/s04/wmConfidence_s04a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S04a_blinkchecked_preprocessed.csv')

        
        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S04b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_24062019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_24062019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_icacleaned_raw.fif')
        param['cuelock_noref_sess2']     = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_cuelock_noref-epo.fif')
        param['cuelock_car_sess2']       = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_cuelock_car-epo.fif')
        param['cuelock_mast_sess2']      = op.join(param['path'], 'eeg/s04/wmConfidence_s04b_cuelock_mast-epo.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes/s04/WMCS04b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes/s04/wmConfidence_s04b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S04b_blinkchecked_preprocessed.csv')
        param['cuelocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-epo.fif')
        param['cuelocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-tfr.h5')
        param['cuelocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_metadata.csv')
        param['cuelocked_tfr_lvsr']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_lvsr-tfr.h5')

        param['fblocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-epo.fif')
        param['fblocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-tfr.h5')
        param['fblocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked_metadata.csv')
        
    if subject['id'] == 5:
        param['subid']                   = 's05'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S05a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_25062019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_25062019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_icacleaned_raw.fif')
        param['cuelock_noref_sess1']     = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_cuelock_noref-epo.fif')
        param['cuelock_car_sess1']       = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_cuelock_car-epo.fif')
        param['cuelock_mast_sess1']      = op.join(param['path'], 'eeg/s05/wmConfidence_s05a_cuelock_mast-epo.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes/s05/WMCS05a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes/s05/wmConfidence_s05a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S05a_blinkchecked_preprocessed.csv')

        
        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S05b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_25062019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_25062019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_icacleaned_raw.fif')
        param['cuelock_noref_sess2']     = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_cuelock_noref-epo.fif')
        param['cuelock_car_sess2']       = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_cuelock_car-epo.fif')
        param['cuelock_mast_sess2']      = op.join(param['path'], 'eeg/s05/wmConfidence_s05b_cuelock_mast-epo.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes/s05/WMCS05b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes/s05/wmConfidence_s05b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S05b_blinkchecked_preprocessed.csv')
        param['cuelocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-epo.fif')
        param['cuelocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-tfr.h5')
        param['cuelocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_metadata.csv')
        param['cuelocked_tfr_lvsr']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_lvsr-tfr.h5')

        param['fblocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-epo.fif')
        param['fblocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-tfr.h5')
        param['fblocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked_metadata.csv')
        
    if subject['id'] == 6:
        param['subid']                   = 's06'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S06a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_26062019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_26062019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_icacleaned_raw.fif')
        param['cuelock_noref_sess1']     = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_cuelock_noref-epo.fif')
        param['cuelock_car_sess1']       = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_cuelock_car-epo.fif')
        param['cuelock_mast_sess1']      = op.join(param['path'], 'eeg/s06/wmConfidence_s06a_cuelock_mast-epo.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes/s06/WMCS06a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes/s06/wmConfidence_s06a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S06a_blinkchecked_preprocessed.csv')

        
        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S06b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_26062019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_26062019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_icacleaned_raw.fif')
        param['cuelock_noref_sess2']     = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_cuelock_noref-epo.fif')
        param['cuelock_car_sess2']       = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_cuelock_car-epo.fif')
        param['cuelock_mast_sess2']      = op.join(param['path'], 'eeg/s06/wmConfidence_s06b_cuelock_mast-epo.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes/s06/WMCS06b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes/s06/wmConfidence_s06b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S06b_blinkchecked_preprocessed.csv')
        param['cuelocked']               = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_s06_cuelocked-epo.fif')
        param['cuelocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-tfr.h5')
        param['cuelocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_metadata.csv')
        param['cuelocked_tfr_lvsr']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_lvsr-tfr.h5')

        param['fblocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-epo.fif')
        param['fblocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-tfr.h5')
        param['fblocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked_metadata.csv')
        
    if subject['id'] == 7:
        param['subid']                   = 's07'
        param['behaviour_sess1']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S06a_allData_preprocessed.csv')
        param['rawdata_sess1']           = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_26062019.cdt')
        param['rawset_sess1']            = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_26062019.set')
        param['rawcleaned_sess1']        = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_icacleaned_raw.fif')
        param['cuelock_noref_sess1']     = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_cuelock_noref-epo.fif')
        param['cuelock_car_sess1']       = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_cuelock_car-epo.fif')
        param['cuelock_mast_sess1']      = op.join(param['path'], 'eeg/s07/wmConfidence_s07a_cuelock_mast-epo.fif')
        param['raweyes_sess1']           = op.join(param['path'], 'eyes/s07/WMCS07a.asc')
        param['cleanedeyes_sess1']       = op.join(param['path'], 'eyes/s07/wmConfidence_s07a_preprocessed.pickle')
        param['behaviour_blinkchecked1'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S07a_blinkchecked_preprocessed.csv')

        
        param['behaviour_sess2']         = op.join(param['path'], 'datafiles/preprocessed_data/wmConfidence_S07b_allData_preprocessed.csv')
        param['rawdata_sess2']           = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_26062019.cdt')
        param['rawset_sess2']            = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_26062019.set')
        param['rawcleaned_sess2']        = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_icacleaned_raw.fif')
        param['cuelock_noref_sess2']     = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_cuelock_noref-epo.fif')
        param['cuelock_car_sess2']       = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_cuelock_car-epo.fif')
        param['cuelock_mast_sess2']      = op.join(param['path'], 'eeg/s07/wmConfidence_s07b_cuelock_mast-epo.fif')
        param['raweyes_sess2']           = op.join(param['path'], 'eyes/s07/WMCS07b.asc')
        param['cleanedeyes_sess2']       = op.join(param['path'], 'eyes/s07/wmConfidence_s07b_preprocessed.pickle')
        param['behaviour_blinkchecked2'] = op.join(param['path'], 'datafiles/blinkchecked/wmConfidence_S07b_blinkchecked_preprocessed.csv')
        param['cuelocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-epo.fif')
        param['cuelocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked-tfr.h5')
        param['cuelocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_metadata.csv')
        param['cuelocked_tfr_lvsr']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_cuelocked_lvsr-tfr.h5')

        param['fblocked']              = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-epo.fif')
        param['fblocked_tfr']          = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked-tfr.h5')
        param['fblocked_tfr_meta']     = op.join(param['path'], 'eeg', param['subid'], 'wmConfidence_'+param['subid']+'_fblocked_metadata.csv')
        
        
        
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
        idx,x2            = gesd(y_epos, p_out=p_out, alpha=alpha, outlier_side=outlier_side) #use the gesd to find outliers (idx is the index of the outlier trials)
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
