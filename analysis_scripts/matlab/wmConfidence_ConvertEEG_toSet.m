clear all
close all
restoredefaultpath;
clc

%%
eeglabpath = '/home/sammirc/eeglab14_1_1b';

wd = '/home/sammirc/Desktop/DPhil/wmConfidence';
chdir(wd);

addpath(genpath(eeglabpath));
addpath([ '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts/toolbox' ]);
%%

subs = [1 2];


for i = 1:length(subs)
    sub = subs(i);
    
    param = getSubjectInfo_wmConf(sub);
    
    d = pop_loadcurry([param.path, param.rawdata]);
    pop_saveset(d, 'filename', param.rawset, 'filepath', param.path);
    clear d;
end
