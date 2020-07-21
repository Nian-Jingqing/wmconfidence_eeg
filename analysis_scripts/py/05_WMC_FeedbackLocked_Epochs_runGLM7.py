import numpy as np
import scipy as sp
import pandas as pd
import mne
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri/analysis_scripts')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR, nanzscore

#sys.path.insert(0, '/Users/sammi/Desktop/Experiments/DPhil/glm')
sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'; #laptop wd
wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
subs = np.array([         4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18,     20, 21, 22,     24, 25, 26])



#%% only needs running if cuelocked TFR glms not already present
for i in subs:
    for glmnum in [7, 8]:
        for laplacian in [True, False]:
            print('\n\nworking on subject ' + str(i) +'\n\n')
            sub = dict(loc = 'workstation', id = i)
            param = get_subject_info_wmConfidence(sub)
        
            #get the epoched data
            epochs = mne.read_epochs(fname = param['fblocked'], preload = True) #this is loaded in with the metadata
            #based on photodiode testing, there is a 25ms delay from trigger onset to maximal photodiode onset, so lets adjust times here
            epochs.shift_time(tshift = -0.025, relative = True)
            epochs = epochs['trialnum != 256'] #this last trial of a session doesnt have a subsequent trial to look at updates
    
            epochs.apply_baseline((-.25, 0)) #baseline 250ms prior to feedback
            epochs.resample(500) #resample to 500Hz
            
            if laplacian:
                laptext='_laplacian'
    #            epochs.set_eeg_reference()# default sets average reference, just wanna see what this looks like really
                epochs = mne.preprocessing.compute_current_source_density(epochs, stiffness = 4)
            else:
                laptext = ''

            ntrials = len(epochs)
        
        
            glmdata     = glm.data.TrialGLMData(data = epochs.get_data(), time_dim = 2, sample_rate = 500)
            nobs        = glmdata.num_observations
            trials      = np.ones(nobs)
        
            cues        = epochs.metadata.cue.to_numpy()
            error       = epochs.metadata.absrdif.to_numpy()
            confwidth   = epochs.metadata.confwidth.to_numpy() #confidence width in degrees, so higher = less confident
            conf        = np.multiply(confwidth, -1) #reverse the sign so higher (less negative) = more confident
            confdiff    = epochs.metadata.confdiff.to_numpy() #error awareness on that trial
            neutral     =  np.where(cues == 0, 1, 0)
            
            nxttrlcw    = epochs.metadata.nexttrlcw.to_numpy(); nxttrlconf = np.multiply(nxttrlcw, -1)
        #    trlupdate = np.subtract(confwidth, nxttrlcw) #positive value means that the confidence with got narrower -- negative value means it got wider
            trlupdate   =  np.subtract(nxttrlconf, conf) #positive values mean they became more confident, negative values they became less confident
            
            targinconf = np.less_equal(confdiff,0)
            targoutsideconf = np.greater(confdiff,0)
            incorrvscorr = np.where(targinconf == 0, 1, -1)
            
            confwidth = nanzscore(confwidth);
            conf      = nanzscore(conf);
            confdiff  = nanzscore(confdiff);
            trlupdate = nanzscore(trlupdate);
            
            regressors = list()
            contrasts  = list()
            if glmnum == 7:
                regressors.append(glm.regressors.ParametricRegressor(name = 'trials', values = trials,    preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name = 'update', values = trlupdate, preproc = None, num_observations = nobs))
                contrasts.append(glm.design.Contrast([1,0], 'grand mean'))
                contrasts.append(glm.design.Contrast([0,1], 'trial update'))
            elif glmnum == 8:
                regressors.append(glm.regressors.ParametricRegressor(name = 'trials',    values = trials,    preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name ='update',     values = trlupdate, preproc = None, num_observations = nobs))
                regressors.append(glm.regressors.ParametricRegressor(name ='confidence', values = conf,      preproc = None, num_observations = nobs))
                contrasts.append(glm.design.Contrast([1,0,0], 'grand mean'))
                contrasts.append(glm.design.Contrast([0,1,0], 'trial update'))
                contrasts.append(glm.design.Contrast([0,0,1],'confidence'))
    
            glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
        #    glmdes.plot_summary()
        
        
            total_nave = len(epochs)
            tmin = epochs.tmin
            info = epochs.info
        
            del(epochs)
        
            print('\nrunning glm\n')
            model = glm.fit.OLSModel( glmdes, glmdata)
        
            del(glmdata) #clear from RAM as not used from now on really
            #contrasts = np.stack([np.arange(len(contrasts)), glmdes.contrast_names], axis=1)
            #ftestz = np.stack([np.arange(len(ftests)), glmdes.ftest_names], axis = 1)
            for iname in range(len(glmdes.contrast_names)):
                name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name
                
                nave = total_nave
        
        
                tl_betas = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
                                                          data = np.squeeze(model.copes[iname,:,:]))
                deepcopy(tl_betas).plot_joint(topomap_args = dict(outlines = 'head', contours = 0),
                         times=np.arange(0,.5,.05), title=name)
                tl_betas.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm'+str(glmnum), 'wmc_' + param['subid'] + '_feedbacklocked_tl_'+ name + laptext + '_betas-ave.fif'))
                del(tl_betas)
        
                tl_tstats = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
                                                          data = np.squeeze(model.get_tstats()[iname,:,:]))
                tl_tstats.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm'+str(glmnum), 'wmc_' + param['subid'] + '_feedbacklocked_tl_'+ name + laptext + '_tstats-ave.fif'))
                del(tl_tstats)
        
        #        tl_varcopes = mne.EvokedArray(info = info, nave = nave, tmin = tmin,
        #                                                  data = np.squeeze(model.varcopes[iname,:,:]))
        #        tl_varcopes.save(fname = op.join(param['path'], 'glms', 'feedback', 'epochs_glm'+str(glmnum), 'wmc_' + param['subid'] + '_feedbacklocked_tl_'+ name + '_varcopes-ave.fif'))
        #        del(tl_varcopes)
        
            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            del(glmdes)
            del(model)
    
    
    
    
    
    
    
