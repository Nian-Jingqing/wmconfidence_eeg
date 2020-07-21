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
    for ipart = 1:2
        
        sprintf('working on subject %2d part %01d/2', subs(i), ipart)
    
        
        sub = subs(i);
        param = getSubjectInfo_wmConf(sub);
        
        cfg = [];
        if ipart==1;
            cfg.dataset = [param.path param.rawset];
        elseif ipart ==2;
            cfg.dataset = [param.path param.rawset2];
        end
        cfg.continuous = 'yes';
        raw = ft_preprocessing(cfg);
        
        %reref data --> filter data ->> ica
        
        %rename the right mastoid (RM) to M2
        chanindx = find(strcmp(raw.label, 'RM'));
        raw.label{chanindx} = 'M2';
        
        %rereference to average of the mastoids
        sprintf('rereferencing the data')
        cfg = [];
        cfg.reref = 'yes';
        cfg.refchannel = 'M2';
        cfg.implicitref = 'M1';
        raw = ft_preprocessing(cfg,raw);
        
        sprintf('filtering the data')
        %filter the data now
        cfg = [];
        cfg.bpfilter = 'yes';
        cfg.dftfilter = 'yes'; %removes 50hz line noise
        cfg.bpfreq = [1 40];
        raw                = ft_preprocessing(cfg, raw);
        save(sprintf('%s/data/rawcleaned_matlab/wmc_s%02d_part%01d_rawcleaned_preica.m', wd, sub, ipart), 'raw');
        
        % just to look at the data if you want to %%
%         cfg = []; cfg.layout = 'easycap-M1';
%         cfg.elec = raw.elec;
%         cfg.viewmode = 'vertical';
%         cfg = ft_databrowser(cfg, raw);
        cfg = []; cfg.demean = 'yes'; raw = ft_preprocessing(cfg, raw); %demean data pre-ica

        %ica for visual artefacts
        cfg = [];
        cfg.method = 'fastica';
        comps = ft_componentanalysis(cfg, raw);
        save(sprintf('%s/data/ica_comps/wmc_s%02d_part%01d_icacomps.m', wd, sub, ipart), 'comps');
        
%         ncomps = size(comps.trial{1}, 1);
        
        %quickly just create layout
%         cfg=[]; cfg.elec = raw.elec; cfg.rotate = 90;
%         cfg.center = 'yes';
%         cfg.channel = {'all', '-VEOG', '-HEOG'};
%         cfg.output = 'easycap-M1.lay';
%         layout = ft_prepare_layout(cfg);
%         ft_plot_lay(layout);
    end
end

%%
%%%% the stuff below here can be in a separate script after reading in raw cleaned pre-ica and the components from ica ...
%%%% this would just allow you to automate the bulk of it and then click
%%%% through each subject in one go a bit more easily
% 
%         cfg = [];
%         cfg.demean = 'yes';
%         cfg.channel = {'HEOG', 'VEOG'};
%         eyed = ft_selectdata(cfg, raw);
% 
%         cfg = [];
%         cfg.component = [1:length(comps.label)];
%         cfg.layout = layout;
%         cfg.comment = 'no';
%         figure; ft_topoplotIC(cfg, comps);
% 
%         cfg = [];
%         cfg.keeptrials = 'yes';
%         tleyed = ft_timelockanalysis(cfg, eyed);
%         tlcomp = ft_timelockanalysis(cfg, comps);
% 
%         x   = tleyed.trial(:,1,:); %blinks
%         xx  = tleyed.trial(:,2,:); %saccades
%         %xxx = tleyed.trial(:,3,:); %ECG %NB: not used in the piloting of EEG in the eeglab because shouldn't cause excessive artefacts like in scanner
% 
%         for c = 1:size(tlcomp.trial,2)
%             y = tlcomp.trial(:,c,:); %loop over components
%             rblink(c) = corr(y(:),x(:));
%             rsac(c)   = corr(y(:),xx(:));
%         end
% 
%         figure; %plot corr of ICA components with blinks and saccades
%         subplot(2,1,1); bar(1:c, abs(rsac)  , 'r'); title('correlation with saccades');
%         subplot(2,1,2); bar(1:c, abs(rblink), 'r'); title('correlation with blinks');
%         xlabel('component number');
% 
%         comp2rem = input('bad components are: ');
%         %s04 part1:
%         %s04 part2: 5, 13, 27, 31
% 
%         cfg = [];
%         cfg.component = sort(comp2rem);
%         raw = ft_rejectcomponent(cfg, comps, raw);
%         
%         %save this preprocessed data
%         
%         save(sprintf('%s/rawcleaned_matlab/wmc_s%02d_part%01d_rawcleaned.m', param.path, i, ipart), 'raw');
%         
%         
% %         %laplacian then save the laplacian data too
% %         do_laplacian = 1;
% %         if do_laplacian;
% %             cfg            = [];
% %             cfg.elec       = raw.elec;
% %             raw_laplacian  = ft_scalpcurrentdensity(cfg,raw);
% % 
% %         end
% %         
% %         save(sprintf('%s/rawcleaned_matlab/wmc_s%02d_part%01d_rawcleaned_laplacian.m', param.path, i, ipart), 'raw_laplacian');
% end

    
    
    
    
    
    
    
    
    