function param = getSubjectInfo_wmConf(subj)

param.path = '/home/sammirc/Desktop/DPhil/wmConfidence/data';

if subj == 1 ;
    param.rawdata = '/eeg/s01/wmConfidence_s01_12062019.cdt';
    param.rawset  = '/eeg/s01/wmConfidence_s01_12062019.set';
elseif subj == 2;
    param.rawdata = '/eeg/s02/wmConfidence_s02_12062019.cdt';
    param.rawset  = '/eeg/s02/wmConfidence_s02_12062019.set';
elseif subj == 3;
    param.rawdata = '/eeg/s03/wmConfidence_s03a_24062019.cdt';
    param.rawset  = '/eeg/s03/wmConfidence_s03a_24062019.set';
elseif subj == 4;
    param.rawdata = '/eeg/s04/wmConfidence_s04a_24062019.cdt';
    param.rawset  = '/eeg/s04/wmConfidence_s04a_24062019.set';
    param.rawdata2 = '/eeg/s04/wmConfidence_s04b_24062019.cdt';
    param.rawset2  = '/eeg/s04/wmConfidence_s04b_24062019.set';
elseif subj == 5;
    param.rawdata = '/eeg/s05/wmConfidence_s05a_25062019.cdt';
    param.rawset  = '/eeg/s05/wmConfidence_s05a_25062019.set';
    param.rawdata2 = '/eeg/s05/wmConfidence_s05b_25062019.cdt';
    param.rawset2  = '/eeg/s05/wmConfidence_s05b_25062019.set';
elseif subj == 6;
    param.rawdata = '/eeg/s06/wmConfidence_s06a_26062019.cdt';
    param.rawset  = '/eeg/s06/wmConfidence_s06a_26062019.set';
    param.rawdata2 = '/eeg/s06/wmConfidence_s06b_26062019.cdt';
    param.rawset2  = '/eeg/s06/wmConfidence_s06b_26062019.set';
elseif subj == 7;
    param.rawdata = '/eeg/s07/wmConfidence_s07a_26062019.cdt';
    param.rawset  = '/eeg/s07/wmConfidence_s07a_26062019.set';
    param.rawdata2 = '/eeg/s07/wmConfidence_s07b_26062019.cdt';
    param.rawset2  = '/eeg/s07/wmConfidence_s07b_26062019.set';
 
end
end
