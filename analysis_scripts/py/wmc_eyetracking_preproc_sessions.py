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

sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence/analysis_scripts')
#sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence_eegfmri/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence

sys.path.insert(0, '/Users/sammi/Desktop/Experiments/BCEyes')
#sys.path.insert(0, '/home/sammirc/Desktop/DPhil/BCEyes')
import BCEyes as bce



wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
os.chdir(wd)


subs = np.array([4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16])
#%%
for i in subs:
    for part in [1, 2]: #2 blocks per participant
        sub = dict(loc = 'laptop', id = i)
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
            print(data[iblock]['Eblk_lx'][0])
            print(data[iblock]['Eblk_rx'][0], '\n')
        #this prints out the first blink in the data (period of missing data). if durations (last column) are negative
        #then there are nans at the beginning of the trace that need handling
        
#        if i == 4 and part == 1:
#            #there is one block for this subject where the very first sample is a nan in both the left and right eye after interpolating blinks
#            #the eyes must have just fallen out
#            #this causes two problems:
#            # 1 -- you get a blink offset but not onset (from the start) so all subsequent blink durations are negative
#            #      and interpolation fails on the right eye
#            # 2 -- prevents you from lpfiltering because it nans the entire trace when you filter
#            
#            #FIX: set specific values to the average of the first 10 samples in both left and right eyes
#            #     NB this needs to be done prior to find_missing_periods or the blink (nan period) durations are all messed up
#            #     not all problems need fixing pre-interpolation as some are related to non-blink missing data that can be interpolated elsewhere
#            data[4]['lx'][0] = np.round(np.nanmean(data[4]['lx'][0:10]), 1) #round to 1 dp so same as other values
#            data[4]['rx'][0] = np.round(np.nanmean(data[4]['rx'][0:10]), 1)
#        if i  == 5 and part == 1:
#            #in this session for subject 5, there are some blocks where there are nans at the start
#            #this is evident after running bce.find_missing_periods if the duration of blinks (e.g. in data[4]['Eblk_lx']) are negative (last column)
#            #for example, block 4 has two missing samples in the left eye at the beginning of the trace
#            #we will set these values to the average of the first ten samples
#            data[4]['lx'][0:2] = np.round(np.nanmean(data[4]['lx'][0:10]), 1)      
#            data[4]['rx'][0:3] = np.round(np.nanmean(data[4]['rx'][0:10]), 1) #in the right eye its the first three samples that are missing
#            
#            #block 5 has the same problem
#            data[5]['lx'][0:2] = np.round(np.nanmean(data[5]['lx'][0:10]), 1) #in the left eye in this block its the first two samples
#            data[5]['rx'][0:5] = np.round(np.nanmean(data[5]['rx'][0:10]), 1) #in the right eye its the first four samples
#            
#            #block 6 is a longer starting blink (it just means the blink started before the recording started really...)
#            data[6]['lx'][0:238] = np.round(np.nanmean(data[6]['lx'][0:250]), 1) #so we'll give it a slightly longer period to get data
#            data[6]['rx'][0:374] = np.round(np.nanmean(data[6]['rx'][0:400]), 1) #even longer period at the start here so get average for a bit longer
#            
#            #block 7 is a shorter messed up period
#            data[7]['lx'][0:4]   = np.round(np.nanmean(data[7]['lx'][0:10]), 1) #so give it the average of the first 10 samples here
#            data[7]['rx'][0:4]   = np.round(np.nanmean(data[7]['rx'][0:10]), 1) #so give it the average of the first 10 samples here
#        if i == 5 and part == 2:
#            #some blocks also have the same problem here: #blocks = [1, 4, 7]
#            data[1]['lx'][0:128] = np.round(np.nanmean(data[1]['lx'][0:150]), 1)
#            data[1]['rx'][0:110] = np.round(np.nanmean(data[1]['rx'][0:120]), 1)
#            
#            #4th block is missing fewer data points at the beginning of the right x trace
#            data[4]['lx'][:94] = np.round(np.nanmean(data[4]['lx'][:100]), 1)
#            data[4]['rx'][0]   = np.round(np.nanmean(data[4]['rx'][0:10]), 1)
#            
#            data[7]['lx'][:245] = np.round(np.nanmean(data[7]['lx'][:260]), 1)
#            data[7]['rx'][:213] = np.round(np.nanmean(data[7]['rx'][:220]), 1)
#        if i == 7 and part == 1:
#            data[2]['rx'][:505] = np.round(np.nanmean(data[2]['rx'][:510]), 1)
#        if i == 8 and part == 1:
#            #blocks = [1, 2, 6]
#            data[1]['lx'][:104] = np.round(np.nanmean(data[1]['lx'][:110]), 1)
#            data[2]['lx'][:2]   = np.round(np.nanmean(data[2]['lx'][:10]),  1)
#            data[2]['rx'][:4]   = np.round(np.nanmean(data[2]['rx'][:10]),  1)
#            data[6]['lx'][:2]   = np.round(np.nanmean(data[6]['lx'][:10]),  1)
#            data[6]['rx'][:3]   = np.round(np.nanmean(data[6]['rx'][:10]),  1)
#        if i == 8 and part == 2: #blocks = [2,5]
#            data[2]['lx'][:140] = np.round(np.nanmean(data[2]['lx'][:150]), 1)
#            data[2]['rx'][:10]  = np.round(np.nanmean(data[2]['rx'][:15]),  1)
#            data[5]['lx'][0]    = np.round(np.nanmean(data[5]['lx'][:10]),  1)
#            data[5]['rx'][:2]   = np.round(np.nanmean(data[5]['rx'][:10]),  1)
#        if i == 9 and part == 2: #blocks = [6,7]
        

        #finds if the block was started with a blink before recording began
        #extrapolates first 10 samples to the beginning of the trace
        data = bce.check_nansatstart(data = data)  
        #this should just find the nanperiods that we made earlier
        #rerunning this is essential to check if the above function has worked
        data = bce.find_missing_periods(data = data,
                                        nblocks = nblocks,
                                        traces_to_scan = ['lp', 'rp', 'lx', 'rx', 'ly', 'ry'])
        
        #if after running this line you run:
        for iblock in range(len(data)):
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
            
        if i == 5 and part == 1:
            for iblock in range(len(data)):
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                fig.suptitle('block %d'%iblock)
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
