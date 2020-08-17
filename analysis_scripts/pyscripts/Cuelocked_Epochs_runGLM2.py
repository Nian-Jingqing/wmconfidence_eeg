import numpy as np
import scipy as sp
import pandas as pd
import mne
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt


# wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
# wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
wd = 'C:/Users/sammi/Desktop/Experiments/DPhil/wmConfidence' # windows laptop wd
os.chdir(wd)

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri/analysis_scripts')
# sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
sys.path.insert(0, 'C:\\Users\\sammi\\Desktop\\Experiments\\DPhil\\wmConfidence\\analysis_scripts')#because working from laptop to make this script
# os.chdir(op.join(os.getcwd, 'analysis_scripts'))
from wmconfidence_funcs import get_subject_info_wmConfidence
from wmconfidence_funcs import gesd, plot_AR, nanzscore

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/glm')
# sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
# sys.path.insert(0, 'C:/Users/sammi/Desktop/Experiments/glm')
import glmtools as glm


subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])

glmnum = 2
#%%
for i in subs:
    print('\n\nworking on subject ' + str(i) +'\n\n')
    
    param = {}
    param['path'] = 'C:/Users/sammi/Desktop/Experiments/DPhil/wmConfidence/data'
    
    sub = dict(loc = 'windows', id = i)
    # param = get_subject_info_wmConfidence(sub)
    for laplacian in [True, False]:
        #get the epoched data
        
        fname = op.join(param['path'], 'eeg', 'wmConfidence_s%02d_cuelocked-epo.fif'%(i))
        
        epochs = mne.read_epochs(fname = fname, preload = True) #this is loaded in with the metadata
        
        #based on photodiode testing, there is a 25ms delay from trigger onset to maximal photodiode onset, so lets adjust times here
        epochs.shift_time(tshift = -0.025, relative = True)
        
        epochs.apply_baseline((-.25, 0)) #baseline 250ms prior to feedback
        epochs.resample(500) #resample to 500Hz
        if laplacian:
            epochs = mne.preprocessing.compute_current_source_density(epochs, stiffness=4)
            lapstr = 'laplacian_'
        else:
            lapstr = ''
        ntrials = len(epochs)
        print('\nSubject %02d has %03d trials\n'%(i, ntrials))
        
        #will do an automated process of looking for trials with heightened variance (noise) and output which trials to keep
        glmdata         = glm.data.TrialGLMData(data = epochs.get_data(), time_dim = 2, sample_rate = 500)
        nobs = glmdata.num_observations
        trials = np.ones(nobs)
        
        cues = epochs.metadata.cue.to_numpy()
        error = epochs.metadata.absrdif.to_numpy()
        confwidth = epochs.metadata.confwidth.to_numpy() #confidence width in degrees, higher = less confident
        conf = np.radians(np.multiply(confwidth, -1)) #reverse the sign so higher (less negative) = more confident, then convert to radians so same scale as error
        confdiff = np.radians(epochs.metadata.confdiff.to_numpy()) #error awareness (metacognition) on that trial (or prediction error)
        neutral = np.where(cues == 0, 1, 0)
        pside = epochs.metadata.pside.to_numpy()
        pside = np.where(pside == 0, 1, -1)
        
        left  =  np.where(pside == 1, 1, 0)
        right =  np.where(pside == -1, 1, 0)
        
        cleft = np.where(np.logical_and( left == 1, cues == 1), 1, 0)
        nleft = np.where(np.logical_and( left == 1, cues == 0), 1, 0)
        
        cright = np.where(np.logical_and( right == 1, cues == 1), 1, 0)
        nright = np.where(np.logical_and( right == 1, cues == 0), 1, 0)
        
        err_cleft  = nanzscore( np.where(cleft  == 1, error, np.nan))
        err_nleft  = nanzscore( np.where(nleft  == 1, error, np.nan))
        err_cright = nanzscore( np.where(cright == 1, error, np.nan)) 
        err_nright = nanzscore( np.where(nright == 1, error, np.nan))
        
        
        #add regressors to the model
        
        regressors = list()
        contrasts  = list()

        regressors.append(glm.regressors.CategoricalRegressor(category_list = nleft , codes = 1, name = 'neutral left'))
        regressors.append(glm.regressors.CategoricalRegressor(category_list = cleft , codes = 1, name = 'cued left'))
        regressors.append(glm.regressors.CategoricalRegressor(category_list = nright, codes = 1, name = 'neutral right'))
        regressors.append(glm.regressors.CategoricalRegressor(category_list = cright, codes = 1, name = 'cued right'))
        regressors.append(glm.regressors.ParametricRegressor(name = 'error cleft',  values = err_cleft,  preproc = None, num_observations = nobs))
        regressors.append(glm.regressors.ParametricRegressor(name = 'error nleft',  values = err_nleft,  preproc = None, num_observations = nobs))
        regressors.append(glm.regressors.ParametricRegressor(name = 'error cright', values = err_cright, preproc = None, num_observations = nobs))
        regressors.append(glm.regressors.ParametricRegressor(name = 'error nright', values = err_nright, preproc = None, num_observations = nobs))
        
        contrasts.append(glm.design.Contrast([ 1,  1,  1,  1,  0,  0,  0,  0], 'grandmean'))
        contrasts.append(glm.design.Contrast([ 1,  0,  0,  0,  0,  0,  0,  0], 'neutral left'))
        contrasts.append(glm.design.Contrast([ 0,  1,  0,  0,  0,  0,  0,  0], 'cued left'))
        contrasts.append(glm.design.Contrast([ 0,  0,  1,  0,  0,  0,  0,  0], 'neutral right'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  1,  0,  0,  0,  0], 'cued right'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  1,  0,  0,  0], 'error cleft'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  1,  0,  0], 'error nleft'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  0,  1,  0], 'error cright'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  0,  0,  1], 'error nright'))
        contrasts.append(glm.design.Contrast([ 1,  0,  1,  0,  0,  0,  0,  0], 'neutral'))
        contrasts.append(glm.design.Contrast([ 0,  1,  0,  1,  0,  0,  0,  0], 'cued'))
        contrasts.append(glm.design.Contrast([-1,  1, -1,  1,  0,  0,  0,  0], 'cued vs neutral'))
        contrasts.append(glm.design.Contrast([-1,  1,  0,  0,  0,  0,  0,  0], 'clvsn'))
        contrasts.append(glm.design.Contrast([ 0,  0, -1,  1,  0,  0,  0,  0], 'crvsn'))
        contrasts.append(glm.design.Contrast([ 0,  1,  0, -1,  0,  0,  0,  0], 'clvsr'))
        contrasts.append(glm.design.Contrast([ 1,  0, -1,  0,  0,  0,  0,  0], 'nlvsr')) #this should only really be useful if looking at behavioural regressor, shouldn't find anything in main effects
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  1,  0, -1,  0], 'error clvsr'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  1, -1,  0,  0], 'error clvsn'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  0,  1, -1], 'error crvsn'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  1,  0, -1], 'error nlvsr'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  1,  0,  1,  0], 'error cued'))
        contrasts.append(glm.design.Contrast([ 0,  0,  0,  0,  0,  1,  0,  1], 'error neutral'))
        
        glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
        # glmdes.plot_summary() #if you want to plot the model summary
        
        total_nave = len(epochs)
        neut_nave  = nleft.sum() + nright.sum()
        cued_nave  = cleft.sum() + cright.sum()
        left_nave  = nleft.sum() + cleft.sum()
        right_nave = nright.sum() + cright.sum()
        tmin = epochs.tmin
        info = epochs.info
        
        print('\n running glm \n')
        model = glm.fit.OLSModel( glmdes, glmdata)
        
        
        del(glmdata) #clear from RAM as not used from now on really
        #contrasts = np.stack([np.arange(len(contrasts)), glmdes.contrast_names], axis=1)
        for iname in range(len(glmdes.contrast_names)):
            name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name
            if iname in [0, 11]:
                nave = total_nave
            elif iname in [1, 6]:
                nave = nleft.sum()
            elif iname in [2, 5]:
                nave = cleft.sum()
            elif iname in [3, 8]:
                nave = nright.sum()
            elif iname in [4, 7]:
                nave = cright.sum()
            elif iname in [9, 15, 19, 21]:
                nave = neut_nave
            elif iname in [10, 14, 16, 20]:
                nave = cued_nave
            elif iname in [12, 17]:
                nave = left_nave
            elif iname in [13, 18]:
                nave = right_nave
                        
            for stat in ['betas', 'tstats']:
                if stat == 'betas':
                    dat = np.squeeze(model.copes[iname, :, :])
                elif stat == 'tstats':
                    dat = np.squeeze(model.get_tstats()[iname, :, :])
                
                tl = mne.EvokedArray(info = info, nave = nave, tmin = tmin, data = dat)
                tl.save(fname = op.join(param['path'], 'glms', 'cuelocked', 'epochs_glm'+str(glmnum), 'wmc_'+'s%02d'%(i)+'_cuelocked_tl_'+ lapstr + name + '_%s-ave.fif'%(stat)))
                # deepcopy(tl).plot_joint(topomap_args = dict(outlines='head', contours=0), times = np.arange(0.1, 0.8, 0.1))
            
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)
        
        
        
        
        
        
        
        
        