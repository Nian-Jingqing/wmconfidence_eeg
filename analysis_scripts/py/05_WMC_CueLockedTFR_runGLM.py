#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:37:29 2019

@author: sammirc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:06:09 2019

@author: sammirc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:54:35 2019

@author: sammirc
"""


import numpy as np
import scipy as sp
import pandas as pd
import mne
from copy import deepcopy
import os
import os.path as op
import sys
from matplotlib import pyplot as plt

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts')
from wmConfidence_funcs import get_subject_info_wmConfidence
from wmConfidence_funcs import gesd, plot_AR

sys.path.insert(0, '/home/sammirc/Desktop/DPhil/glm')
import glmtools as glm

wd = '/home/sammirc/Desktop/DPhil/wmConfidence' #workstation wd
os.chdir(wd)


subs = np.array([1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
#subs = np.array([11,12,13,14])
#subs = np.array([11, 15])

glms2run = 2 #1 with no baseline, one where tfr input data is baselined
for i in subs:
    for iglm in range(glms2run):
        print('\n\nrunning glm %d/%d'%(iglm+1, glms2run))
        print('-- working on subject ' + str(i) +' --\n\n')
        sub = dict(loc = 'workstation', id = i)
        param = get_subject_info_wmConfidence(sub)

        #get tfr
        tfr = mne.time_frequency.read_tfrs(fname=param['cuelocked_tfr']); tfr=tfr[0]
        tfr.metadata = pd.read_csv(param['cuelocked_tfr_meta'], index_col=None) #read in and attach metadata

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
           tfr = tfr.apply_baseline((-0.5, -0.3))

        glmdata         = glm.data.TrialGLMData(data = tfr.data, time_dim = 3, sample_rate = 100)


        cues      = tfr.metadata.cue.to_numpy() #cue condition for trials
        absrdif   = tfr.metadata.absrdif.to_numpy() #response error on trial (lower is better)
        confwidth = tfr.metadata.confwidth.to_numpy() #confidence width in degrees, higher = less confident
        conf = np.radians(np.multiply(confwidth, -1)) #reverse the sign so higher (less negative) = more confident, then convert to radians so same scale as error
        cuedside  = np.where(tfr.metadata.cuetrig==14, -1, cues) #cued left trials = 1, cued right = -1 (we implicitly code lateralisation in this regressor by flipping signs)
        DT        = tfr.metadata.DT.to_numpy() #decision time (time until pressing space to start response phase) on each trial

        regressors = list()
        regressors.append( glm.regressors.ParametricRegressor(name = 'trials', values = np.ones(glmdata.num_observations), preproc=None, num_observations = glmdata.num_observations))
        regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 0, name = 'neutral') )
        regressors.append( glm.regressors.CategoricalRegressor(category_list = cues, codes = 1, name = 'cued') )
        regressors.append( glm.regressors.ParametricRegressor(name = 'cuedside',          values = cuedside,                         preproc = None, num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'error',             values = absrdif,                          preproc = 'z',  num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'confidence',        values = conf,                        preproc = 'z',  num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'DT',                values = DT,                               preproc = 'z',  num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'error x side',      values = np.multiply(cuedside, absrdif),   preproc = 'z',  num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'confidence x side', values = np.multiply(cuedside, conf), preproc = 'z',  num_observations = glmdata.num_observations) )
        regressors.append( glm.regressors.ParametricRegressor(name = 'DT x side',         values = np.multiply(cuedside, DT),        preproc = 'z',  num_observations = glmdata.num_observations) )

        contrasts = list()
        contrasts.append( glm.design.Contrast([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'grand mean') )
        contrasts.append( glm.design.Contrast([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'neutral') )
        contrasts.append( glm.design.Contrast([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'cued') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'cued side') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'error') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'confidence') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'DT') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'error x side') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'confidence x side') )
        contrasts.append( glm.design.Contrast([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 'DT x side') )
        contrasts.append( glm.design.Contrast([0,-1, 1, 0, 0, 0, 0, 0, 0, 0], 'cued vs neutral') )


        glmdes = glm.design.GLMDesign.initialise(regressors, contrasts)
        #if iglm == 0:
        #    glmdes.plot_summary()

        total_nave = len(tfr)
        cued_nave  = len(tfr['cue==1'])
        neut_nave  = len(tfr['cue==0'])
        times = tfr.times
        freqs = tfr.freqs
        info = tfr.info

        del(tfr)


        print('\n - - - - -  running glm - - - - - \n')
        model = glm.fit.OLSModel( glmdes, glmdata)

        del(glmdata) #clear from RAM as not used from now on really

        for iname in range(len(glmdes.contrast_names)):
            name = glmdes.contrast_names[iname].replace(' ','') #remove whitespace in the contrast name
            if iname == 1:
                nave = neut_nave
            elif iname in [2,3,7,8,9]:
                nave = cued_nave
            else:
                nave = total_nave


            tfr_betas = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.copes[iname,:,:,:]))
            tfr_betas.save(fname = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_'+ name + '_betas' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_betas)

            tfr_tstats = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.get_tstats()[iname,:,:,:]))
            tfr_tstats.save(fname = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_'+ name + '_tstats' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_tstats)

            tfr_varcopes = mne.time_frequency.AverageTFR(info = info, times = times, freqs = freqs, nave = nave,
                                                      data = np.squeeze(model.varcopes[iname,:,:,:]))
            tfr_varcopes.save(fname = op.join(param['path'], 'glms', 'cue_period', 'wmConfidence_' + param['subid'] + '_cuelocked_tfr_'+ name + '_varcopes' + addtopath + '-tfr.h5'), overwrite = True)
            del(tfr_varcopes)

        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        del(glmdes)
        del(model)
