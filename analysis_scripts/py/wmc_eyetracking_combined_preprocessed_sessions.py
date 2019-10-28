#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:50:08 2019

@author: sammi
"""
import numpy as np
import scipy as sp
import pandas as pd
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
import pickle

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence/analysis_scripts')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/BCEyes')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/BCEyes')
import BCEyes as bce



wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence';
os.chdir(wd)


subs = np.array([4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18])

for i in subs:
    sub = dict(loc = 'workstation', id = i)
    param = get_subject_info_wmConfidence(sub)
    print('-- working on subject %d --\n'%(i))
    
    print(' -- reading in both session data --\n')
    with open(param['cleanedeyes_sess1'], 'rb') as handle1:
        data1 = pickle.load(handle1)
    with open(param['cleanedeyes_sess2'], 'rb') as handle2:
        data2 = pickle.load(handle2)
    
    data = np.concatenate([data1, data2]) #concatenates both subjects
    
    #write to pickle here
    print('-- writing concatenated data to file --\n')
    with open(op.join(param['path'], 'eyes', param['subid'], 'wmc_'+param['subid']+'_preprocessed_combined.pickle'), 'wb') as handle:
        pickle.dump(data, handle)
