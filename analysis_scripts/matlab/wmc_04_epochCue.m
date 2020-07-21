clear all
close all
restoredefaultpath;
clc

%%
% eeglabpath = '/home/sammirc/eeglab14_1_1b';
ftpath     = '/home/sammirc/fieldtrip-20180115';

wd = '/home/sammirc/Desktop/DPhil/wmConfidence';
chdir(wd);

% addpath(genpath(eeglabpath));
addpath([ '/home/sammirc/Desktop/DPhil/wmConfidence/analysis_scripts/toolbox' ]);
addpath(ftpath);
ft_defaults;
%%

subs = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22, 23, 24, 25, 26];
subs = [      4 5 6 7 8 9    11 12 13 14 15 16 17 18    20 21 22,     24, 25, 26];

for i = 1:length(subs);
    sub = subs(i);
    
    sprintf('working on subject %02d', sub)
    if ~exist(sprintf('%s/data/epoched_matlab/wmc_s%02d_CueLocked.mat', wd, subs(i)));
        param = getSubjectInfo_wmConf(sub);

        raw1 = load(sprintf('%s/rawcleaned_matlab/wmc_s%02d_part1_rawcleaned.mat', param.path, subs(i))); raw1 = raw1.raw;
        raw2 = load(sprintf('%s/rawcleaned_matlab/wmc_s%02d_part2_rawcleaned.mat', param.path, subs(i))); raw2 = raw2.raw;

        %epoch first session
        cfg = [];
        cfg.dataset             = [param.path param.rawset];
        cfg.trialfun            = 'ft_trialfun_general';
        cfg.trialdef.eventtype  = 'trigger';
        cfg.trialdef.eventvalue = [11,12,13,14]; %retrocue trigger
        cfg.trialdef.prestim    = 2.000;
        cfg.trialdef.poststim   = 2.250;
        cfg                     = ft_definetrial(cfg);
        data1 = ft_redefinetrial(cfg,raw1);

        cfg = [];
        cfg.offset = 26;
        data1 = ft_redefinetrial(cfg,data1);

        %epoch second session
        cfg = [];
        cfg.dataset             = [param.path param.rawset2];
        cfg.trialfun            = 'ft_trialfun_general';
        cfg.trialdef.eventtype  = 'trigger';
        cfg.trialdef.eventvalue = [11,12,13,14]; %retrocue trigger
        cfg.trialdef.prestim    = 2.000;
        cfg.trialdef.poststim   = 2.250;
        cfg                     = ft_definetrial(cfg);
        data2 = ft_redefinetrial(cfg,raw2);

        cfg = [];
        cfg.offset = 26;
        data2 = ft_redefinetrial(cfg,data2);

        cfg = [];
        data = ft_appenddata(cfg, data1, data2);


        %read in behavioural datafiles and add to the data structures
        bpath = sprintf('%s/data/datafiles/s%02d/wmConfidence_S%02d_gathered_preprocessed.csv', wd, sub, sub);
        bdata = readtable(bpath);

        %column1 in trialinfo is the trigger for that event
        %add behavioural data into the trial info structure
        data.trialinfo(:,2)  = bdata.trialnum;
        data.trialinfo(:,3)  = bdata.cue;
        data.trialinfo(:,4)  = bdata.pside;
        data.trialinfo(:,5)  = bdata.ori1;
        data.trialinfo(:,6)  = bdata.ori2;
        data.trialinfo(:,7)  = bdata.targori;
        data.trialinfo(:,8)  = bdata.nontargori;
        data.trialinfo(:,9)  = bdata.DT;
        data.trialinfo(:,10) = bdata.absrdif;
        data.trialinfo(:,11) = bdata.confwidth;
        data.trialinfo(:,12) = bdata.confdiff;
        if iscell(bdata.CT);
            bdata.CT(strcmp(bdata.CT, 'NA'))         = {'NaN'}; bdata.CT = double(cellfun(@str2num, bdata.CT));
        end
        if iscell(bdata.confCT);
            bdata.confCT(strcmp(bdata.confCT, 'NA')) = {'NaN'}; bdata.confCT = double(cellfun(@str2num, bdata.confCT));
        end
        
        data.trialinfo(:,13) = bdata.clickresp;
        data.trialinfo(:,14) = bdata.CT;
        data.trialinfo(:,15) = bdata.DTcheck;

        incl = find(data.trialinfo(:,13)==1 & data.trialinfo(:,15) == 0);
        cfg = [];
        cfg.trials = incl; %remove trials where didnt click to respond, and DT outside 2.5 SDs of subject mean
        data = ft_selectdata(cfg, data); 
        
        


        %now want to do some trial rejection based on noise in data
        %semi-automatic trial rejection
        cfg = [];
        cfg.method = 'summary';
        cfg.keepchannel = 'yes';
        cfg.keeptrial  = 'nan'; %throw out bad trials, as behavioural data already added into trialinfo
        data = ft_rejectvisual(cfg, data);

        tmp = cellfun(@sum, cellfun(@sum, cellfun(@sum,data.trial,'UniformOutput',0), 'UniformOutput',0), 'UniformOutput',0);
        throwouts = find(cellfun(@isnan,tmp) == 1);
        trls = 1:size(data.trial,2); trls(throwouts)=[]; %get trials marked as bad from data
        cfg  = []; cfg.trials = trls; data = ft_selectdata(cfg,data); %exclude them

        save(sprintf('%s/data/epoched_matlab/wmc_s%02d_CueLocked.mat', wd, sub), 'data');
    
        clear raw1
        clear raw2
        clear data1
        clear data2
        clear data
        clear bdata
    end 
end
