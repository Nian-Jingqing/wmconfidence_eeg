function param = getSubjectInfo_wmConf(subj)

param.path = '/home/sammirc/Desktop/DPhil/wmConfidence/data';

if subj == 1;
    param.subid                     = 's01';
    param.behaviour                 = '/datafiles/s01/wmConfidence_eegfmri_outsidescanner_S01_allData_preprocessed.csv';
    param.rawdata                   = '/eeg/s01/wmConfidence_s01_12062019.cdt';
    param.rawset                    = '/eeg/s01/wmConfidence_s01_12062019.set';
    param.cuelock_noref             = '/eeg/s01/wmConfidence_s01_cuelock_noref.mat';
    param.cuelock_car               = '/eeg/s01/wmConfidence_s01_cuelock_common.mat';
    param.cuelock_mast              = '/eeg/s01/wmConfidence_s01_cuelock_mast.mat';
    param.cuelock_noref_laplacian   = '/eeg/s01/wmConfidence_s01_cuelock_noref_laplacian.mat';
    param.cuelock_car_laplacian     = '/eeg/s01/wmConfidence_s01_cuelock_common_laplacian.mat';
    param.cuelock_mast_laplacian    = '/eeg/s01/wmConfidence_s01_cuelock_mast_laplacian.mat';
    param.includechans = {'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','T7','C5','C3','C1','CZ', 'C4','C6','T8','TP7', ... %C2 excluded broken channel
                          'CP5','CP3','CP1','CPZ', 'CP2','CP4','CP6','TP8','M1','M2','P7','P5','P3','P1', 'PZ', 'P2','P4','P6','P8','PO7','PO3','POZ','PO4','PO8','O1','OZ','O2','VEO','HEO'}; 
    param.car_chans = {'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','T7','C5','C3','C1','CZ', 'C4','C6', ... %C2 excluded broken channel
                          'T8','TP7','CP5','CP3','CP1','CPZ', 'CP2','CP4','CP6','TP8','P7','P5','P3','P1', 'PZ', 'P2','P4','P6','P8','PO7','PO3','POZ','PO4','PO8','O1','OZ','O2'}; 
elseif subj == 2;
    param.subid                     = 's02';
    param.behaviour                 = '/datafiles/s02/wmConfidence_eegfmri_outsidescanner_S02_allData_preprocessed.csv';
    param.rawdata                   = '/eeg/s02/wmConfidence_s02_12062019.cdt';
    param.rawset                    = '/eeg/s02/wmConfidence_s02_12062019.set';
    param.cuelock_noref             = '/eeg/s02/wmConfidence_s02_cuelock_noref.mat';
    param.cuelock_car               = '/eeg/s02/wmConfidence_s02_cuelock_common.mat';
    param.cuelock_mast              = '/eeg/s02/wmConfidence_s02_cuelock_mast.mat';
    param.cuelock_noref_laplacian   = '/eeg/s02/wmConfidence_s02_cuelock_noref_laplacian.mat';
    param.cuelock_car_laplacian     = '/eeg/s02/wmConfidence_s02_cuelock_common_laplacian.mat';
    param.cuelock_mast_laplacian    = '/eeg/s02/wmConfidence_s02_cuelock_mast_laplacian.mat';
    param.includechans = {'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','T7','C5','C3','C1','CZ', 'C4','C6','T8','TP7', ... %C2 excluded broken channel
                          'CP5','CP3','CP1','CPZ', 'CP2','CP4','CP6','TP8','M1','M2','P7','P5','P3','P1', 'PZ', 'P2','P4','P6','P8','PO7','PO3','POZ','PO4','PO8','O1','OZ','O2','VEO','HEO'}; 
    param.car_chans = {'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','T7','C5','C3','C1','CZ', 'C4','C6', ... %C2 excluded broken channel
                          'T8','TP7','CP5','CP3','CP1','CPZ', 'CP2','CP4','CP6','TP8','P7','P5','P3','P1', 'PZ', 'P2','P4','P6','P8','PO7','PO3','POZ','PO4','PO8','O1','OZ','O2'}; 
    

end
end
