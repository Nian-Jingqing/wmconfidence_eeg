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

for i = 1:length(subs)
    sub = subs(i);
    param = getSubjectInfo_wmConf(sub);
    for ipart = 1:2
        if ~exist(sprintf('%s/rawcleaned_matlab/wmc_s%02d_part%01d_rawcleaned.mat', param.path, subs(i), ipart));
            sprintf('working on subject %2d part %01d/2', subs(i), ipart)

            raw   = load(sprintf('%s/data/rawcleaned_matlab/wmc_s%02d_part%01d_rawcleaned_preica.mat', wd, sub, ipart)); raw = raw.raw;
            comps = load(sprintf('%s/data/ica_comps/wmc_s%02d_part%01d_icacomps.mat', wd, sub, ipart)); comps = comps.comps;

            cfg = [];   
            cfg.demean = 'yes';
            cfg.channel = {'HEOG', 'VEOG'};
            eyed = ft_selectdata(cfg, raw);

            cfg = [];
            cfg.component = [1:length(comps.label)];
            cfg.layout = './easycap-M1.lay';
            cfg.comment = 'no';
            figure; ft_topoplotIC(cfg, comps);

            cfg = [];
            cfg.keeptrials = 'yes';
            tleyed = ft_timelockanalysis(cfg, eyed);
            tlcomp = ft_timelockanalysis(cfg, comps);

            x   = tleyed.trial(:,1,:); %blinks
            xx  = tleyed.trial(:,2,:); %saccades
            %xxx = tleyed.trial(:,3,:); %ECG %NB: not used in the piloting of EEG in the eeglab because shouldn't cause excessive artefacts like in scanner

            for c = 1:size(tlcomp.trial,2)
                y = tlcomp.trial(:,c,:); %loop over components
                rblink(c) = corr(y(:),x(:));
                rsac(c)   = corr(y(:),xx(:));
            end

            figure; %plot corr of ICA components with blinks and saccades
            subplot(2,1,1); bar(1:c, abs(rsac)  , 'r'); title('correlation with saccades');
            subplot(2,1,2); bar(1:c, abs(rblink), 'r'); title('correlation with blinks');
            xlabel('component number');

            comp2rem = input('bad components are: ');
            %s04 part1: 12 16 23
            %s04 part2: 5, 13, 27, 31
            save(sprintf('%s/rawcleaned_matlab/wmc_s%02d_part%01d_comps2rem.mat', param.path, subs(i), ipart), 'comp2rem');

            %s05 part1:
            %s05 part2:

            cfg = [];
            cfg.component = sort(comp2rem);
            raw = ft_rejectcomponent(cfg, comps, raw);

            %save this preprocessed data

            save(sprintf('%s/rawcleaned_matlab/wmc_s%02d_part%01d_rawcleaned.mat', param.path, subs(i), ipart), 'raw');        
        end
        close all
        clear rblink
        clear rsac
    end
end

        
        
        
        
        