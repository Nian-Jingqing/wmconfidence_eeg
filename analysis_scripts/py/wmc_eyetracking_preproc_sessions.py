#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:22:32 2019

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
#%%
for i in subs:
    for part in [1, 2]: #2 blocks per participant
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub)
        print('-- working on subject %d --\n'%(i))



        print('-- working on part %d --\n\n'%(part))

        data = bce.parse_eye_data(eye_fname = param['raweyes_sess'+str(part)], block_rec = True, trial_rec = False, nblocks = 8)

        nblocks = len(data)
        if nblocks !=8: #exit this script if the data isnt right, as it will need looking into
            raise bce.DataError('found %d blocks when expecting 8 in this block -- something went wrong with this recording'%nblocks)

        dat = deepcopy(data)

        data = bce.nanblinkperiods_fromeyelink(data = data,
                                               nblocks = nblocks,
                                               traces_to_scan = ['lx', 'ly', 'lp', 'rx', 'ry', 'rp'],
                                               remove_win = 60) #50ms sample to remove around the start and end of the blink period
        #it seems like this routine is actually not too bad? if you use the blinks from the appropriate eyes to nan the blink period
        #it works ok using the eyelink blink identified areas

        #this should just find the nanperiods that we made earlier
        data = bce.find_missing_periods(data = data,
                                        nblocks = nblocks,
                                        traces_to_scan = ['lp', 'rp', 'lx', 'rx', 'ly', 'ry'])

        #if after running this line you run:
        for iblock in range(len(data)):
            if i == 11 and part == 1 and iblock == 7: # this subject is missing data for one entire block in this session
                print('skipping block %d for subject 11 session 1'%(iblock+1))
                continue
            if i == 11 and part == 2 and iblock in [2]: #eyes missing in this block
                print('skipping block %d for subject 11 session 2'%(iblock+1))
                continue
            if i == 12 and part == 1 and iblock in [6]:
                print('skipping block %d for subject 12 session 1'%(iblock+1))
                continue
            if i == 12 and part == 2 and iblock in [5,6]:
                print('skipping block %d for subject 12 session 1'%(iblock+1))
                continue
            else:
                print(data[iblock]['Eblk_lx'][0])
                print(data[iblock]['Eblk_rx'][0], '\n')

        #this prints out the first blink in the data (period of missing data). if durations (last column) are negative
        #then there are nans at the beginning of the trace that need handling

        #finds if the block was started with a blink before recording began
        #extrapolates first 10 samples to the beginning of the trace
        if i == 11 and part == 1:
            data = bce.check_nansatstart(data = data,
                                         blocks_to_ignore = np.subtract([8], 1) ) #block 8 (last block) missing eyes)
        elif i == 11 and part == 2:
            data = bce.check_nansatstart(data = data,
                                         blocks_to_ignore = np.subtract([3], 1) ) #block 3 missing eyes
        elif i == 12 and part == 1:
            data = bce.check_nansatstart(data = data,
                                         blocks_to_ignore = np.subtract([7], 1) )
        elif i == 12 and part == 2:
            data = bce.check_nansatstart(data = data,
                                         blocks_to_ignore = np.subtract([6, 7], 1))
        else:
            data = bce.check_nansatstart(data = data)
        #this should just find the nanperiods that we made earlier
        #rerunning this is essential to check if the above function has worked
        data = bce.find_missing_periods(data = data,
                                        nblocks = nblocks,
                                        traces_to_scan = ['lp', 'rp', 'lx', 'rx', 'ly', 'ry'])

        #if after running this line you run:
        for iblock in range(len(data)):
            if i == 11 and part == 1 and iblock == 7: # this subject is missing data for one entire block in this session
                print('skipping block %d for subject 11 session 1'%(iblock+1))
                continue
            if i == 11 and part == 2 and iblock in [2]: #eyes missing in this block
                print('skipping block %d for subject 11 session 2'%(iblock+1))
                continue
            if i == 12 and part == 1 and iblock in [6]:
                print('skipping block %d for subject 12 session 1'%(iblock+1))
                continue
            if i == 12 and part == 2 and iblock in [5,6]:
                print('skipping block %d for subject 12 session 1'%(iblock+1))
                continue
            else:
                print(data[iblock]['Eblk_lx'][0])
                print(data[iblock]['Eblk_rx'][0], '\n')
            #this should now output no negative duration blinks if it's worked properly
        #this prints out the first blink in th left eye trace. if the last column is negative, you have a problem at the beginning of the trace

        blockid = 1
        clean_traces = ['lp', 'rp', 'lx', 'rx', 'ly', 'ry']
        for block in data:
            print('cleaning data for block %d/%d'%(blockid, nblocks))
            for trace in clean_traces:
                block = bce.interpolateBlinks_Blocked(block, trace, method = 'linear')
            blockid += 1

        plotsubs_blocks = False
        if plotsubs_blocks:
            for iblock in range(len(data)):
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                fig.suptitle('block %d'%(iblock+1))
                ax1.plot(data[iblock]['rx'])
                remsampsr = data[iblock]['rblink_removedsamps']
                nansampsr = np.squeeze(np.where(np.isnan(data[iblock]['rx'])))
                ysampsr = np.multiply(np.ones(remsampsr.size), 800)
                ax1.scatter(remsampsr, ysampsr, color = 'r', marker = '.')
                if nansampsr.size > 0:
                    ynansampsr = np.multiply(np.ones(nansampsr.size), 780)
                    ax1.scatter(nansampsr, ynansampsr, color = 'green', marker = '.')

                ax2.plot(data[iblock]['lx'])
                remsampsl = data[iblock]['lblink_removedsamps']
                nansampsl = np.squeeze(np.where(np.isnan(data[iblock]['lx'])))
                ysampsl = np.multiply(np.ones(remsampsl.size), 800)
                ax2.scatter(remsampsl, ysampsl, color = 'r', marker = '.')
                if nansampsl.size > 0:
                    ynansampsl = np.multiply(np.ones(nansampsl.size), 780)
                    ax2.scatter(nansampsl, ynansampsl, color = 'green', marker = '.')
        #to plot the output of this
#        iblock=4
#        fig = plt.figure()
#        fig.suptitle('subject %s part %d, block 4'%(param['subid'], part))
#        ax = fig.add_subplot(211)
#        ax.set_title('left x')
#        ax.plot(dat[iblock]['lx'], lw = .75)
#        ax.plot(data[iblock]['lx'], lw = .75)
#        ax = fig.add_subplot(212)
#        ax.set_title('right x')
#        ax.plot(dat[iblock]['rx'], lw = .75)
#        ax.plot(data[iblock]['rx'], lw = .75)

        with open(param['cleanedeyes_sess'+str(part)], 'wb') as handle:
            pickle.dump(data, handle)
