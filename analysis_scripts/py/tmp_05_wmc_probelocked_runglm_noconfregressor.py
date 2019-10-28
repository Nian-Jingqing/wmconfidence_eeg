#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:55:09 2019

@author: sammirc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:57:57 2019

@author: sammirc
"""
import numpy as np
import scipy as sp
import pandas as pd
import mne
import os
import os.path as op
import sys
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from copy import deepcopy

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR


sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)
#subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
subs = np.array([        4, 5, 6, 7, 8, 9,     11, 12, 13, 14, 15, 16, 17, 18])
subs = np.array([15, 16, 17, 18])
#subs = np.array([17, 18])

#subs = np.array([10]) #encountered memory error in subject 7 so rerun from here
#%% only needs running if Probelocked TFR glms not already present
#subs = np.array([7])
glmstorun = 2
for i in subs:
    for iglm in range(glmstorun):
        print('\n\nrunning glm %d/%d'%(iglm+1, glmstorun))
        print('-- working on subject ' + str(i) +' --\n\n')
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub)

        #get tfr
        tfr             = mne.time_frequency.read_tfrs(fname=param['probelocked_tfr'])[0]
        tfr.metadata    = pd.read_csv(param['probelocked_tfr_meta'], index_col=None) #read in and attach metadata

        #some regressors need to have baselined data as the input because we aren't looking at a specific contrast
        #this is true for grand mean responses, neutral trials only, cued trials only (i.e. average neutral response)
        #also true for main effects of error and reaction time (and confidence if included) that are not carried by a side interaction
        #so i think it might be best if i run the same glm twice, but with one the data is baselined and in the others it isn't.

        # for just subject 7 and 9 i want to force rerunning the second glm (with baselines), so ...
        #if i == 9:
        #    iglm = iglm +1

        if iglm == 0:
            addtopath = ''
            baseline_input = False
        elif iglm == 1:
            addtopath = '_baselined'
            baseline_input = True

        if baseline_input:
           print(' -- baselining the TFR data -- ')
           tfr = tfr.apply_baseline((-2.0, -1.7))

        glmdata         = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 100)
        nobs = glmdata.num_observations

        #get some behavioural things we're going to look at
        trials = np.ones(glmdata.num_observations) #regressor for just grand mean response

        cues   = tfr.metadata.cue.to_numpy()
        pside = tfr.metadata.pside.to_numpy()
        pside = np.where(pside == 0, 1, -1)

        regressors = list()
        regressors.append( glm.regressors.ParametricRegressor(name = 'grand mean', values = trials, preproc = None, num_observations = nobs))
        #regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 0, name = 'neutral'))
        #regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 1, name = 'cued'))

        probeleft = np.where(pside == 1, 1, 0)
        proberight = np.where(pside == -1, 1, 0)

        pleft_neut = np.where(np.logical_and(pside == 1, cues == 0), 1, 0)
        pleft_cued = np.where(np.logical_and(pside == 1, cues == 1), 1, 0)

        pright_neut = np.where(np.logical_and(pside == -1, cues == 0), 1, 0)
        pright_cued = np.where(np.logical_and(pside == -1, cues == 1), 1, 0)

        regressors.append(glm.regressors.CategoricalRegressor(category_list = pleft_neut, codes = 1, name = 'probe left neutral'))
        regressors.append(glm.regressors.CategoricalRegressor(category_list = pleft_cued, codes = 1, name = 'probe left cued'))

        regressors.append(glm.regressors.CategoricalRegressor(category_list = pright_neut, codes = 1, name = 'probe right neutral'))
        regressors.append(glm.regressors.CategoricalRegressor(category_list = pright_cued, codes = 1, name = 'probe right cued'))

        DT = tfr.metadata.DT.to_numpy()
        error = tfr.metadata.absrdif.to_numpy()
        error = np.where(error == 0, 0.1, error)
        confwidth = tfr.metadata.confwidth.to_numpy()
        cw = np.where(confwidth == 0, 0.1, confwidth)


        #set up behavioural correlation regressors now
        dt_pleft_neut = np.log(np.where(np.logical_and(pside == 1, cues == 0), DT, np.nan))
        dt_pleft_cued = np.log(np.where(np.logical_and(pside == 1, cues == 1), DT, np.nan))

        dt_pright_neut = np.log(np.where(np.logical_and(pside == -1, cues == 0), DT, np.nan))
        dt_pright_cued = np.log(np.where(np.logical_and(pside == -1, cues == 1), DT, np.nan))

        #now we need to zscore these before setting nans to 0 for the glm

        dt_pleft_neut = np.divide(np.subtract(dt_pleft_neut, np.nanmean(dt_pleft_neut)), np.nanstd(dt_pleft_neut))
        dt_pleft_cued = np.divide(np.subtract(dt_pleft_cued, np.nanmean(dt_pleft_cued)), np.nanstd(dt_pleft_cued))

        dt_pright_neut = np.divide(np.subtract(dt_pright_neut, np.nanmean(dt_pright_neut)), np.nanstd(dt_pright_neut))
        dt_pright_cued = np.divide(np.subtract(dt_pright_cued, np.nanmean(dt_pright_cued)), np.nanstd(dt_pright_cued))


        dt_pleft_neut  = np.where(np.isnan(dt_pleft_neut), 0, dt_pleft_neut)
        dt_pleft_cued  = np.where(np.isnan(dt_pleft_cued), 0, dt_pleft_cued)
        dt_pright_neut = np.where(np.isnan(dt_pright_neut), 0, dt_pright_neut)
        dt_pright_cued = np.where(np.isnan(dt_pright_cued), 0, dt_pright_cued)

        regressors.append(glm.regressors.ParametricRegressor(name = 'dt pleft neutral', values = dt_pleft_neut, preproc = None, num_observations = nobs))
        regressors.append(glm.regressors.ParametricRegressor(name = 'dt pleft cued', values = dt_pleft_cued, preproc = None, num_observations = nobs))
        regressors.append(glm.regressors.ParametricRegressor(name = 'dt right neutral', values = dt_pright_neut, preproc = None, num_observations = nobs))
        regressors.append(glm.regressors.ParametricRegressor(name = 'dt pright cued', values = dt_pright_cued, preproc = None, num_observations = nobs))


        #same for error
        err_pleft_neut = np.log(np.where(np.logical_and(pside == 1, cues == 0), error, np.nan))
        err_pleft_cued = np.log(np.where(np.logical_and(pside == 1, cues == 1), error, np.nan))

        err_pright_neut = np.log(np.where(np.logical_and(pside == -1, cues == 0), error, np.nan))
        err_pright_cued = np.log(np.where(np.logical_and(pside == -1, cues == 1), error, np.nan))

        #now we need to zscore these before setting nans to 0 for the glm

        err_pleft_neut = np.divide(np.subtract(err_pleft_neut, np.nanmean(err_pleft_neut)), np.nanstd(err_pleft_neut))
        err_pleft_cued = np.divide(np.subtract(err_pleft_cued, np.nanmean(err_pleft_cued)), np.nanstd(err_pleft_cued))

        err_pright_neut = np.divide(np.subtract(err_pright_neut, np.nanmean(err_pright_neut)), np.nanstd(err_pright_neut))
        err_pright_cued = np.divide(np.subtract(err_pright_cued, np.nanmean(err_pright_cued)), np.nanstd(err_pright_cued))


        err_pleft_neut  = np.where(np.isnan(err_pleft_neut), 0, err_pleft_neut)
        err_pleft_cued  = np.where(np.isnan(err_pleft_cued), 0, err_pleft_cued)
        err_pright_neut = np.where(np.isnan(err_pright_neut), 0, err_pright_neut)
        err_pright_cued = np.where(np.isnan(err_pright_cued), 0, err_pright_cued)

        regressors.append(glm.regressors.ParametricRegressor(name = 'err pleft neutral', values = err_pleft_neut, preproc = None, num_observations = nobs))
        regressors.append(glm.regressors.ParametricRegressor(name = 'err pleft cued', values = err_pleft_cued, preproc = None, num_observations = nobs))
        regressors.append(glm.regressors.ParametricRegressor(name = 'err right neutral', values = err_pright_neut, preproc = None, num_observations = nobs))
        regressors.append(glm.regressors.ParametricRegressor(name = 'err pright cued', values = err_pright_cued, preproc = None, num_observations = nobs))


        #same for confidence now
        cw_pleft_neut = np.log(np.where(np.logical_and(pside == 1, cues == 0), cw, np.nan))
        cw_pleft_cued = np.log(np.where(np.logical_and(pside == 1, cues == 1), cw, np.nan))

        cw_pright_neut = np.log(np.where(np.logical_and(pside == -1, cues == 0), cw, np.nan))
        cw_pright_cued = np.log(np.where(np.logical_and(pside == -1, cues == 1), cw, np.nan))

        #now we need to zscore these before setting nans to 0 for the glm

        cw_pleft_neut = np.divide(np.subtract(cw_pleft_neut, np.nanmean(cw_pleft_neut)), np.nanstd(cw_pleft_neut))
        cw_pleft_cued = np.divide(np.subtract(cw_pleft_cued, np.nanmean(cw_pleft_cued)), np.nanstd(cw_pleft_cued))

        cw_pright_neut = np.divide(np.subtract(cw_pright_neut, np.nanmean(cw_pright_neut)), np.nanstd(cw_pright_neut))
        cw_pright_cued = np.divide(np.subtract(cw_pright_cued, np.nanmean(cw_pright_cued)), np.nanstd(cw_pright_cued))


        cw_pleft_neut  = np.where(np.isnan(cw_pleft_neut), 0, cw_pleft_neut)
        cw_pleft_cued  = np.where(np.isnan(cw_pleft_cued), 0, cw_pleft_cued)
        cw_pright_neut = np.where(np.isnan(cw_pright_neut), 0, cw_pright_neut)
        cw_pright_cued = np.where(np.isnan(cw_pright_cued), 0, cw_pright_cued)

        regressors.append(glm.regressors.ParametricRegressor(name = 'conf pleft neutral', values = cw_pleft_neut,  preproc = None, num_observations = nobs))
        regressors.append(glm.regressors.ParametricRegressor(name = 'conf pleft cued',    values = cw_pleft_cued,  preproc = None, num_observations = nobs))
        regressors.append(glm.regressors.ParametricRegressor(name = 'conf right neutral', values = cw_pright_neut, preproc = None, num_observations = nobs))
        regressors.append(glm.regressors.ParametricRegressor(name = 'conf pright cued',   values = cw_pright_cued, preproc = None, num_observations = nobs))


        contrasts = list()
        contrasts.append(glm.design.Contrast([ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'grand mean')            )
        contrasts.append(glm.design.Contrast([ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pleft_neutral')         )
        contrasts.append(glm.design.Contrast([ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pleft_cued')            )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pright_neutral')        )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pright_cued')           )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'dt_pleft_neutral')      )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'dt_pleft_cued')         )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'dt_pright_neutral')     )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'dt_pright_cued')        )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'error_pleft_neutral')   )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'error_pleft_cued')      )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'error_pright_neutral')  )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'error_pright_cued')     )
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'conf_pleft_neutral'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'conf_pleft_cued'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'conf_pright_neutral'))
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'conf_pright_cued'))      #16
        contrasts.append(glm.design.Contrast([ 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pleft_cvsn')            )#17
        contrasts.append(glm.design.Contrast([ 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'pright_cvsn')           )#18
        contrasts.append(glm.design.Contrast([ 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'neutral')               )#19
        contrasts.append(glm.design.Contrast([ 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'cued')                  )#20
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'dt_pleft_cvsn')         )#21
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'dt_pright_cvsn')        )#22
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'dt_neutral')            )#23
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'dt_cued')               )#24
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0], 'dt_cued_lvsr')          )#25
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0, 0, 0], 'error_pleft_cvsn')      )#26
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 0, 0], 'error_pright_cvsn')     )#27
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0], 'error_neutral')         )#28
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], 'error_cued')            )#29
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,-1, 0, 0, 0, 0], 'error_cued_lvsr')       )#30
        contrasts.append(glm.design.Contrast([ 0,-1, 1, 1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'plvsr_cvsn')            )#31
        contrasts.append(glm.design.Contrast([ 0, 0, 1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'plvsr_cued')            )#32
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0], 'conf_pleft_cvsn')       )#33
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1], 'conf_pright_cvsn')      )#34
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0], 'conf_neutral')          )#35
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], 'conf_cued')             )#36
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,-1], 'conf_cued_lvsr')        )#37
        contrasts.append(glm.design.Contrast([ 0,-1, 1,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'cued vs neutral')       )#38
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0,-1, 1,-1, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'dt_cued vs neutral')    )#39
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1,-1, 1, 0, 0, 0, 0], 'error_cued vs neutral') )#40
        contrasts.append(glm.design.Contrast([ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 1,-1, 1], 'conf_cued vs neutral')  )#41

        glmdes = glm.design.GLMDesign.initialise(regressors, contrasts); #designs.append(glmdes)
        #if iglm == 0:
        #    glmdes.plot_summary()

        total_nave = len(tfr)
        neut_nave  = len(tfr['cue==0'])
        cued_nave  = len(tfr['cue==1'])
        neut_pleft_nave = len(tfr['cuetrig == 11'])
        cued_pleft_nave = len(tfr['cuetrig == 13'])
        neut_pright_nave = len(tfr['cuetrig == 12'])
        cued_pright_nave = len(tfr['cuetrig == 14'])
        times = tfr.times
        freqs = tfr.freqs
        info = tfr.info

        del(tfr)
        print('\n - - - - -  running glm - - - - - \n')
        model = glm.fit.OLSModel( glmdes, glmdata )

        del(glmdata) #clear from RAM as not used from now on really
        names = glmdes.contrast_names

        for iname in range(len(glmdes.contrast_names)):
            name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name
            if iname in [19, 23, 28, 35]:
                nave = neut_nave
            elif iname in [20, 21, 24, 25, 29, 30 ,32, 36, 37]:
                nave = cued_nave
            elif iname in [1, 5, 9, 13]:
                nave = neut_pleft_nave
            elif iname in [2, 6, 10, 14]:
                nave = cued_pleft_nave
            elif iname in [3, 7, 11, 15]:
                nave = neut_pright_nave
            elif iname in [4, 8, 12, 16]:
                nave = cued_pright_nave
            elif iname in [17, 21, 26, 33]:
                nave = neut_pleft_nave + cued_pleft_nave
            elif iname in [18, 22, 27, 34]:
                nave = neut_pright_nave + cued_pright_nave
            else:
                nave = total_nave


            tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.copes[iname,:,:,:]))
#            tfr_betas.plot_joint(title = '%s, betas'%(name),
#                                 timefreqs = {
#                                         (-1.1, 10):(.4, 4), (-.9,10):(.4,4),(-.7,10):(.4,4), #some cue locked topos
#                                         (.4, 10):(.4, 4),(.6, 10):(.4, 4),(.8, 10):(.4, 4)}, #baseline = (-2, -1.8),
#                                 topomap_args = dict(outlines = 'head', contours = 0,
#                                                     vmin = np.divide(np.min(tfr_betas.data),10), vmax = np.divide(np.min(tfr_betas.data), -10)))
            tfr_betas.save(fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm_2_conf', 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_betas)

            tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.get_tstats()[iname,:,:,:]))
#            deepcopy(tfr_tstats).drop_channels(['RM']).plot_joint(title = '%s, tstats'%(name),
#                                 timefreqs = {
#                                         (-1.1, 10):(.4, 4), (-.9,10):(.4,4),(-.7,10):(.4,4), #some cue locked topos
#                                         (.4, 10):(.4, 4),(.6, 10):(.4, 4),(.8, 10):(.4, 4)}, #baseline = (-2, -1.8),
#                                 topomap_args = dict(outlines = 'head', contours = 0,
#                                                     vmin = -2, vmax = 2))
            tfr_tstats.save(fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm_2_conf', 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_tstats)

            tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
            tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'probe', 'tfr_glm_2_conf', 'wmc_' + param['subid'] + '_probelocked_tfr_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_varcopes)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)
#%%
