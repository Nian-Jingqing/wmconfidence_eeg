#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:54:05 2019
@author: sammirc
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

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri/analysis_scripts')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence_eegfmri/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/BCEyes')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/BCEyes')
import BCEyes as bce


wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence_eegfmri' #workstation wd
os.chdir(wd)


subs = np.array(['pilot1_inside', 'pilot2_inside', 'pilot1_outside'])


subind = 3 #get first subject

sub = dict(loc='workstation',
           id=subs[subind-1]) #get subject name for param extraction
param = get_subject_info_wmConfidence(sub)

if op.exists(param['cleanedeyes']):
    with open(param['cleanedeyes'], 'rb') as handle:
        data = pickle.load(handle)
else:    
    data = bce.parse_eye_data(eye_fname = param['raweyes'], block_rec=True,trial_rec = False, nblocks=8) #load the data
    
    nblocks = len(data) #get number of blocks in data
    
    #this will just nan any points that aren't possible with the screen dimensions you have (something went wrong, or tracker fritzed)
    data = bce.replace_points_outside_screen(data = data, nblocks = nblocks,
                                             traces_to_scan = ['lx', 'ly', 'rx', 'ry'], screen_dim = [1920, 1080],
                                             adjust_pupils=False)
    
    #detect blinks in the data using pupil trace, not gaze coords. uses physiological limits for pupil movement to detect blinks
    data = bce.cleanblinks_usingpupil(data             = data,
                                      nblocks          = nblocks,
                                      signals_to_clean = ['x', 'y'],
                                      eyes             = ['left', 'right'])
    
    #find all nan periods in the data (eg missing due to blinks) and add to data structure
    data = bce.find_missing_periods(data = data,
                                    nblocks = nblocks,
                                    traces_to_scan = ['lp', 'rp', 'lx', 'rx', 'ly', 'ry'])
    
    #now interpolate these missing periods to clean the data
    ds = deepcopy(data) #will use this to actually see how much better the data is after cleaning
    #data = deepcopy(ds) #if you make a mistake at some point, run this line
    
    blockid = 1
    clean_traces = ['lp', 'rp', 'lx', 'rx', 'ly', 'ry']
    for block in data:
        print('cleaning data for block %02d/%02d'%(blockid,nblocks))
        for trace in clean_traces:
            block = bce.interpolateBlinks_Blocked(block,trace)
        blockid +=1
        
    import pickle
    with open(param['cleanedeyes'], 'wb') as handle:
        pickle.dump(data, handle)
    

#check quality of cleaning here

blockid = 0

plt.figure()
plt.plot(data[blockid]['lp'], lw = 1, color = '#238b45', label = 'left pupil cleaned')
plt.plot(ds[blockid]['lp'], lw =  1, color = '#d7301f', label = 'left pupil raw')
plt.legend()
plt.title('example of blink artefact removal in pupil trace')


#before we can epoch, lets read in the behavioural data and add it to the data structure
bdata = pd.DataFrame.from_csv(param['behaviour'])



# epoch the cleaned data
all_trigs = sorted(np.unique(data[0]['Msg'][:,2]))

cue_trigs      = dict(neutral_left='trig11', neutral_right='trig12', cued_left='trig13', cued_right='trig14', neutral=['trig11','trig12'], cued=['trig13','trig14'])


cued_cueperiod = bce.epoch(data, trigger_values = cue_trigs['cued'], traces=['lx','rx'],twin=[-0.5,1.5],srate=1000)
cued_cueperiod = bce.apply_baseline(cued_cueperiod, traces = ['lx', 'rx'], baseline_window=[-0.2,0.0], mode='mean', baseline_shift_gaze=[960,540])
cued_cueperiod = bce.average_eyes(cued_cueperiod, traces=['x'])

neutral_cueperiod = bce.epoch(data,trigger_values=cue_trigs['neutral'],traces=['lx','rx'],twin=[-0.5,1.5],srate=1000)
neutral_cueperiod = bce.apply_baseline(neutral_cueperiod,traces=['lx','rx'],baseline_window=[-0.2,0.0], mode='mean', baseline_shift_gaze=[960,540])
neutral_cueperiod = bce.average_eyes(neutral_cueperiod, traces = ['x'])

twin_srate          = np.multiply([cued_cueperiod['info']['tmin'],cued_cueperiod['info']['tmax']],cued_cueperiod['info']['srate']).astype(int)
timerange = np.divide(np.arange(twin_srate[0],twin_srate[1],1,dtype=float),cued_cueperiod['info']['srate'])

#plot just raw x coordinates, average across trials, evoked by valid retrocues
plt.figure()
plt.plot(timerange, np.subtract(np.nanmean(cued_cueperiod['ave_x'][cued_cueperiod['info']['trigger']==cue_trigs['cued_left'],],0),960),label='cued left', lw=1,color='#fc8d59')
plt.plot(timerange, np.subtract(np.nanmean(cued_cueperiod['ave_x'][cued_cueperiod['info']['trigger']==cue_trigs['cued_right'],],0),960),label='cued right', lw=1,color='#91bfdb')
plt.plot(timerange, np.subtract(np.nanmean(neutral_cueperiod['ave_x'],0),960), label = 'neutral cue', lw = 1, color = '#636363')
plt.axvline(x=0.026,ls='dashed', color='k',lw=1)
plt.axhline(y=0,ls='dashed', color='k', lw=1)
plt.legend()
plt.title('average x coordinate relative to cue')


plt_left = cued_cueperiod['ave_x'][cued_cueperiod['info']['trigger']==cue_trigs['cued_left']]
plt.figure()
for i in range(plt_left.shape[0]):
    plt.plot(timerange, np.subtract(plt_left[i,:],np.nanmean(plt_left[i,476:526])),lw=1, color = '#bdbdbd')
plt.plot(timerange, np.subtract(np.nanmean(plt_left,0),np.nanmean(np.nanmean(plt_left,0)[476:526])),lw=1, color = '#fc8d59')
plt.axvline(x=.026,ls='dashed',color='k',lw=2)
plt.title('all trials, cued left, rel. to cue onset')

plt.figure()
plt.plot(timerange, np.nanmean(plt_left,0), label='cued left', lw=1, color = '#fc8d59')
plt.axvline(x=0.026,lw=1,ls='dashed',color='k')
plt.legend()


# -----------------------------------------------------------------------------------------------------------------------
#epoch and plot relative to feedback that people received
bdata = pd.DataFrame.from_csv(param['behaviour'])
traces2epoch = ['lp','rp']
timewin = [-0.5, 1.5]

bdata_nleft  = bdata.query('fbtrig==76')
bdata_cleft  = bdata.query('fbtrig==78')
bdata_nright = bdata.query('fbtrig==77')
bdata_cright = bdata.query('fbtrig==79')


#watch out when epoching around the feedback, because the final trials feedback will be messed up if you dont wait out at the end of the task!
feedback_trigs = dict(neutral_left='trig76', neutral_right='trig77', cued_left='trig78', cued_right='trig79', neutral=['trig76','trig77'], cued=['trig78','trig79'])

cued_fb_epoch       = bce.epoch(data, trigger_values = feedback_trigs['cued'],    traces = ['lp', 'rp'], twin = timewin, srate=1000)
neutral_fb_epoch    = bce.epoch(data, trigger_values = feedback_trigs['neutral'], traces = ['lp', 'rp'], twin = timewin, srate=1000)

#get all feedback events
fb_epochs = bce.epoch(data, trigger_values = ['trig76', 'trig77', 'trig78', 'trig79'], traces = ['lp', 'rp'], twin=timewin,srate=1000)
fb_epochs = bce.apply_baseline( fb_epochs, traces=['lp', 'rp'], baseline_window = [-0.1,0.0], mode='mean' ) #baseline
fb_epochs = bce.average_eyes(fb_epochs, traces = ['p'])

#baseline pupil data
cued_fb_epoch    = bce.apply_baseline(cued_fb_epoch,    traces = ['lp', 'rp'], baseline_window = [-0.1,0.0], mode='mean')
neutral_fb_epoch = bce.apply_baseline(neutral_fb_epoch, traces = ['lp', 'rp'], baseline_window = [-0.1,0.0], mode='mean')

#average pupil data across the two eyes
cued_fb_epoch    = bce.average_eyes(cued_fb_epoch   , traces = ['p'])
neutral_fb_epoch = bce.average_eyes(neutral_fb_epoch, traces = ['p'])


#plotting
twin_srate = np.multiply([timewin[0], timewin[1]],fb_epochs['info']['srate']).astype(int)
timerange  = np.divide(np.arange(twin_srate[0],twin_srate[1],1,dtype=float),fb_epochs['info']['srate'])

#just plot the average for now, no separation by any behavioural information
plt.figure()
plt.plot(timerange, np.nanmean(cued_fb_epoch['ave_p'],0),    label = 'cued'   , lw = 1, color = '#3182bd') #blue for cued
plt.plot(timerange, np.nanmean(neutral_fb_epoch['ave_p'],0), label = 'neutral', lw = 1, color = '#bdbdbd') #grey for neutral
plt.axvline(x = 0.0, lw = 1, ls = 'dashed', color = 'k')
plt.axvline(x = 0.5, lw = 1, ls = 'dashed', color = 'k', label = 'feedback offset')
plt.axvline(x = 1.5, lw = 1, ls = 'dashed', color = 'r', label = 'minimum onset of next trial')
plt.legend()
plt.title('pupil dilation rel. to feedback onset')


#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/glm')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm
cued_bdata = bdata.query('cond==\'cued\'') #get the cued behavioural data

#glmdata = glm.data.TrialGLMData(data= cued_fb_epoch['ave_p'], time_dim=1, sample_rate=1000)
glmdata = glm.data.TrialGLMData(data = fb_epochs['ave_p'], category_list = bdata.cue.to_numpy(), time_dim=1, sample_rate=1000)

regressors = list()
regressors.append( glm.regressors.ParametricRegressor(name= 'confdiff', values = bdata.confdiff.to_numpy(), preproc='z', num_observations = glmdata.num_observations))
regressors.append( glm.regressors.CategoricalRegressor(category_list = bdata.cue.to_numpy(), codes = 0))
regressors.append( glm.regressors.CategoricalRegressor(category_list = bdata.cue.to_numpy(), codes = 1))
#regressors.append( glm.regressors.CategoricalRegressor(category_list = bdata.cue.to_numpy(), codes = (0,1)) )
#regressors.append( glm.regressors.ParametricRegressor(name = 'condition comparison', values = bdata.cue.to_numpy(), num_observations = glmdata.num_observations) )
cuecond = bdata.cue.to_numpy()
cuecond = np.where(cuecond==0, -1,cuecond)
regressors.append( glm.regressors.ParametricRegressor(name = 'condition x confdiff', values = np.multiply(bdata.confdiff.to_numpy(), cuecond), num_observations=glmdata.num_observations))



contrasts = list()
contrasts.append( glm.design.Contrast([1, 0, 0, 0], 'confidence deviation'))
contrasts.append( glm.design.Contrast([0, 1, 0, 0], 'neutral'))
contrasts.append( glm.design.Contrast([0, 0, 1, 0], 'cued'))
contrasts.append( glm.design.Contrast([0, 0, 0, 1], 'condition x confdiff'))

ftests = list()
ftests.append( glm.design.FTest( [0,1,1,0], 'mean condition') )



glmdes = glm.design.GLMDesign.initialise(regressors, contrasts, ftests)
print(glmdes.ftest_names)

glmdes.plot_summary()

model = glm.fit.OLSModel( glmdes, glmdata )

#i'm pretty sure that here the middle plot is plotting the beta coefficients across the length of the time series to show where the effect starts to emerge (which is awesome)
plt.figure()
plt.subplot(311)
plt.plot(timerange, model.copes.T,lw=1) 
plt.legend(glmdes.contrast_names, loc = 'upper left', ncol=2)
plt.title('COPEs')
plt.subplot(312)
plt.plot(timerange, model.get_tstats().T, lw=1)
plt.axhline(y=0, ls='dashed', color='k', lw=1)
plt.axhline(y=2.58, ls='dashed', color = '#bdbdbd') #2.58 tstat line for significance
plt.axhline(y=-2.58, ls='dashed', color = '#bdbdbd') #2.58 tstat line for significance
plt.legend(glmdes.contrast_names, loc = 'upper left', ncol=2)
plt.title('t-stats')
plt.subplot(313)
plt.plot(timerange, np.nanmean(cued_fb_epoch['ave_p'],0),    label = 'cued'   , lw = 1, color = '#3182bd') #blue for cued
plt.plot(timerange, np.nanmean(neutral_fb_epoch['ave_p'],0), label = 'neutral', lw = 1, color = '#bdbdbd') #grey for neutral
plt.axvline(x = 0.0, lw = 1, ls = 'dashed', color = 'k')
plt.axvline(x = 0.5, lw = 1, ls = 'dashed', color = 'k', label = 'feedback offset')
plt.axvline(x = 1.5, lw = 1, ls = 'dashed', color = 'r', label = 'minimum onset of next trial')
plt.legend()
plt.title('pupil dilation rel. to feedback onset')
plt.show()


plt.figure()
plt.plot(timerange,model.fstats.T)
plt.legend(glmdes.ftest_names)
plt.title('FTests')
