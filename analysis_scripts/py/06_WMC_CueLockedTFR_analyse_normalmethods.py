import numpy as np
import scipy as sp
import pandas as pd
import mne
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
from copy import deepcopy
from scipy import stats

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, toverparam

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)

subs = np.array([        4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 24, 25, 26])


timefreqs_alpha ={(.4, 10):(.4, 4),
                  (.6, 10):(.4, 4),
                  (.8, 10):(.4, 4),
                  (1., 10):(.4, 4),
                  (1.2, 10):(.4, 4)}

tfrs = []
laplacian=True
#%%
for stiff in [3,4]: # check out which number of legendre terms is better
    cleft  = []
    cright = []
    nleft  = []
    nright = []
    
    clvsr = []
    
    
    for i in subs:
        print('-- reading in subject ' + str(i) +' --\n\n')
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub)
    
        #get tfr
        if not laplacian:
            tfr = mne.time_frequency.read_tfrs(fname=param['cuelocked_tfr']); tfr=tfr[0]
        else:
            tfr  = mne.time_frequency.read_tfrs(fname = param['cuelocked_tfr'].replace('cuelocked-tfr', 'cuelocked_laplacian_stiffness%d-tfr'%(stiff)));  tfr = tfr[0]
#            tfr2 = mne.time_frequency.read_tfrs(fname = param['cuelocked_tfr'].replace('cuelocked-tfr', 'cuelocked_laplacian_stiffness%d-tfr'%(4))); tfr2 = tfr2[0]
        
        tfr.metadata  = pd.read_csv(param['cuelocked_tfr_meta'], index_col = None) #read in and attach metadata
#        tfr2.metadata = pd.read_csv(param['cuelocked_tfr_meta'], index_col = None) 
        
        cl = tfr['cue == 1 & pside == 0'].average()
        cr = tfr['cue == 1 & pside == 1'].average()
        lvsr = cl - cr
    #    lvsr.plot_joint(timefreqs = timefreqs_alpha, topomap_args=dict(contours=0))
        clvsr.append(lvsr)
        cleft.append(cl); cright.append(cr)
        
        del(tfr); del(cl); del(cr); del(lvsr)

    gave_clvsr = mne.grand_average(clvsr);
    gave_clvsr.plot_joint(timefreqs = timefreqs_alpha, topomap_args = dict(contours=0), title = 'clvsr, stiffness = %d'%stiff)
    
    gave_cleft = mne.grand_average(cleft)
    gave_cleft.apply_baseline((-2, -1.5))
    gave_cleft.plot_joint(timefreqs = timefreqs_alpha, topomap_args = dict(contours = 0), title = 'cleft, stiffness = %d'%stiff)
    
    gave_cright = mne.grand_average(cright)
    gave_cright.apply_baseline((-2,-1.5))
    gave_cright.plot_joint(timefreqs = timefreqs_alpha, topomap_args = dict(contours = 0), title = 'right, stiffness = %d'%stiff)

