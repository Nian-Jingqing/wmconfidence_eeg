#import libraries
library(tidyverse) # general data manipulation & plotting
library(magrittr)  # allows use of more pipes
library(afex)      # anovas etc
library(RePsychLing)
library(MASS)

loadfonts = F
if(loadfonts){
  library(extrafont)
  font_import() #yes to this
  loadfonts(device = "pdf");
  loadfonts(device = 'postscript')
}

#set theme for plots to standardise them and make cleaner
theme_set(theme_bw() +
            theme(
              axis.text        = element_text(size=12),
              legend.text      = element_text(size=12),
              strip.text.x     = element_text(size=12),
              strip.background = element_blank()
            ))
cuedcol <- '#3182bd' #blue, use for cued
neutcol <- '#bdbdbd' #light-ish grey, use for neutral
diffcol <- '#B2DF8A' #for colouring of the bars of difference plots, third distinct colour

se <- function(x) sd(x)/sqrt(length(x))


wd <- '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri'
setwd(wd)

dpath <- paste0(wd, '/data') #path to folder with behavioural data
figpath <- paste0(wd, '/figures')

#get my block of test data
fpath <- paste0(dpath, '/wmSelection_BehaviouralData_All_Preprocessed.csv')
#fpath <- paste0(wd, '/EEG/data/s01/behaviour/wmSelection_EEG_S01_allData.csv') #look at Alex's EEG session behavioural data
fpath <- '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence_eegfmri/data/datafiles/outside_pilot1/wmConfidence_eegfmri_outsidescanner_S01_allData_preprocessed.csv'



df <- read.csv(fpath, header = T, as.is = T, sep = ',') %>% dplyr::select(-X) # str(df) if you want to see the columns and values etc
nsubs = length(unique(df$subid))
subs2use <- c(1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,19,20,21,22)


df %>%
  dplyr::select(subid, confclicked, cond) %>%
  group_by(subid, cond) %>%
  dplyr::mutate(confclicked = (confclicked -1)*-1) %>%
  dplyr::summarise_at(.vars = c('confclicked'), .funs = c('sum')) %>%
  as.data.frame() %>%
  dplyr::mutate(clickresp = round((confclicked/128)*100,1))


df %<>% dplyr::filter(subid %in% subs2use) %>% #keep only subjects safe to use based on clicking to confirm response (this is a failsafe in case it's not coded later on)
  dplyr::filter(clickresp == 1) %>% dplyr::filter(confclicked == 1) %>% dplyr::filter(DTcheck == 0) #hard code this now
#trials excluded if didn't click to confirm response or confidence judgement, and DT outside a couple sd's of mean per subject and condition

dim(df) 

df %>%
  dplyr::filter(clickresp == 1) %>% dplyr::filter(confclicked==1) %>% dplyr::filter(DTcheck == 0) %>%
  ggplot(aes(x = confwidth, fill = cond)) +
  geom_density(alpha = .5, adjust = 1) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'confidence report width') +
  facet_wrap(~subid)

#think this is getting towards what nils wanted to see, about whether the distribution of confidence and responses overlap
df %>%
  dplyr::filter(clickresp == 1) %>% dplyr::filter(DTcheck == 0) %>%
  dplyr::filter(cond == 'neutral') %>% droplevels() %>%
  ggplot() +
  geom_density(aes(x = confdiff, fill = neutcol), alpha = .5, adjust = 1.5) +
  geom_density(aes(x = rdif, fill = cuedcol), alpha = .5, adjust = 1.5) +
  scale_fill_manual(values = c('#bdbdbd' = neutcol, '#3182bd' = cuedcol), labels = c('confidence', 'response')) +
  facet_wrap(~subid)


df %>%
  dplyr::filter(clickresp == 1) %>%
  filter(DTcheck == 0) %>%
  ggplot(aes(x = confwidth, fill = cond)) +
  geom_histogram(stat = 'bin', binwidth = 3) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'confidence report width') +
  facet_wrap(subid ~ cond, ncol = 6) +
  theme(strip.text.x = element_text(size = 6),
        axis.text.x  = element_text(size = 8),
        axis.text.y  = element_text(size = 6))
ggsave(filename = paste0(figpath, '/confwidth_hist_13subs_pilot.pdf'), device = 'pdf', dpi = 600, height = 10, width = 10)
ggsave(filename = paste0(figpath, '/confwidth_hist_13subs_pilot.eps'), device = 'eps', dpi = 600, height = 10, width = 10)

df %>%
  dplyr::filter(clickresp == 1, confclicked == 1) %>% filter(DTcheck == 0) %>%
  ggplot(aes(x = confwidth, fill = cond)) +
  geom_density(adjust = 1.2)+#, alpha = .4) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'confidence report width', y = '') +
  facet_wrap(~subid, ncol = 5) +
  theme(strip.text.x = element_text(size = 10),
        axis.text.x  = element_text(size = 10),
        axis.text.y  = element_text(size = 10))
ggsave(filename = paste0(figpath, '/confwidth_density.pdf'), device = 'pdf', dpi = 600, height = 10, width = 10)
ggsave(filename = paste0(figpath, '/confwidth_density.eps'), device = 'eps', dpi = 600, height = 10, width = 10)

df %>%
  dplyr::filter(clickresp == 1, confclicked == 1) %>% filter(DTcheck == 0) %>%
  ggplot(aes(x = confdiff, fill = cond)) +
  geom_histogram(aes(y = ..count../sum(..count..)*100),
                 binwidth = 4, #this sets a realistic resolution of people's use of the mouse
                 center = 0, #centre the bins around 0
                 alpha = 1, #no transparency for exporting to sketch
                 position = position_identity()) + #position_identity stops them stacking, so they're on same y scale (makes it easier to see equivalence
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  geom_vline(xintercept = 0, linetype = 'dashed') +
  labs(x = 'confidence report width', y = '') +
  facet_wrap(~subid, ncol = 5) +
  theme(strip.text.x = element_text(size = 10),
        axis.text.x  = element_text(size = 10),
        axis.text.y  = element_text(size = 10))
ggsave(filename = paste0(figpath, '/confdiff_hist.pdf'), device = 'pdf', dpi = 600, height = 10, width = 10)
ggsave(filename = paste0(figpath, '/confdiff_hist.eps'), device = 'eps', dpi = 600, height = 10, width = 10)


#look at proportion of trials where the target orientation is actually in the confidence interval
#i.e.e whether they were at all aware of how accurate they were (calibrated)

df.confacc <- df %>%
  dplyr::mutate(targinconf = ifelse(confdiff <= 0, 1, 0)) %>% #if included target, 1, else 0
  dplyr::group_by(subid, cond) %>%
  summarise_at(.vars = 'targinconf', .funs = 'mean') %>%
  dplyr::mutate(targinconf = targinconf*100)

df.confacc.diffs <- df.confacc %>%
  dplyr::group_by(subid) %>%
  dplyr::summarise(diff = last(targinconf) - first(targinconf))


#in this plot, negative numbers means greater percentage of trials with target inside confint for cued
df.confacc.diffs %>%
  dplyr::summarise_at(.vars = 'diff', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = 1, y = mean, ymin = mean-se, ymax = mean + se)) +
  geom_bar(stat = 'identity', width = .4, fill = diffcol) + 
  geom_errorbar(stat = 'identity', width = .15, size = .4) +
  geom_point(data = df.confacc.diffs, aes(x = 1, y = diff), inherit.aes = F,size = .5, position = position_jitter(.01)) + 
  geom_hline(yintercept = 0, linetype = 'dashed', color = '#000000', size = .2) +
  labs(x = '', y = 'difference in % trials where target inside\nconfidence interval (neutral - cued)') +
  xlim(c(0.5,1.5)) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
ggsave(filename = paste0(figpath, '/targinconf_groupaverage_diffs.eps'), device = 'eps', dpi = 600, height = 5, width = 5)
ggsave(filename = paste0(figpath, '/targinconf_groupaverage_diffs.pdf'), device = 'pdf', dpi = 600, height = 5, width = 5)



#plot group average with individual data points
df.confacc %>%
  dplyr::group_by(cond) %>%
  dplyr::summarise_at(.vars = 'targinconf', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = cond, y = mean, fill = cond)) +
  geom_bar(stat = 'identity', width = .7 ) +
  geom_errorbar(aes(ymin = mean-se, ymax = mean+se), width = .35) +
  geom_point(inherit.aes = F, data = df.confacc, aes(x = cond, y = targinconf), size = .5) +
  geom_line(inherit.aes = F, data = df.confacc, aes(x = cond, y = targinconf, group = subid), size = .1) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = '', y = '% trials where target inside confidence interval') +
  theme(legend.position = 'none')
ggsave(filename = paste0(figpath, '/targinconf_groupaverage.eps'), device = 'eps', dpi = 600, height = 5, width = 5)
ggsave(filename = paste0(figpath, '/targinconf_groupaverage.pdf'), device = 'pdf', dpi = 600, height = 5, width = 5)
  
aov_confacc = afex::aov_ez(id = 'subid', data = df.confacc, 'targinconf', within = 'cond')
nice(aov_confacc, es = 'pes')
t.test(data = df.confacc, targinconf ~ cond, paired = T)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

library(circular)
df.confmu <- df %>%
  dplyr::mutate(cond = as.factor(cond)) %>%
  dplyr::mutate(absconfdiff = abs(confdiff)) %>% #need this to get mean absolute confidence error
  dplyr::group_by(subid, cond) %>%
  dplyr::summarise_at(.vars = 'absconfdiff', .funs = mean)

df.confmu.separate <- df %>%
  dplyr::mutate(cond = as.factor(cond)) %>%
  dplyr::mutate(targinconf = ifelse(confdiff <= 0, 'Target Inside', 'Target Outside')) %>%
  dplyr::mutate(targinconf = as.factor(targinconf)) %>%
  dplyr::mutate(absconfdiff = abs(confdiff)) %>% #need this to get mean absolute confidence error
  dplyr::group_by(subid, cond, targinconf) %>%
  dplyr::summarise_at(.vars = 'absconfdiff', .funs = mean)
  

df.confmu.diffs <- df.confmu %>%
  dplyr::group_by(subid) %>%
  dplyr::summarise(diff = last(absconfdiff) - first(absconfdiff))
#here, positive values mean that neutral was higher than cued (i.e. on average less accurate with confidence)

#plot as one difference bar, rather than separate condition bars
df.confmu.diffs %>%
  dplyr::summarise_at(.vars = 'diff', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = 1, y = mean, ymin = mean-se, ymax = mean + se)) +
  geom_bar(stat = 'identity', width = .4, fill = diffcol) + 
  geom_errorbar(stat = 'identity', width = .15, size = .4) +
  geom_point(data = df.confmu.diffs, aes(x = 1, y = diff), inherit.aes = F,size = .5, position = position_jitter(.01)) + 
  geom_hline(yintercept = 0, linetype = 'dashed', color = '#000000', size = .2) +
  labs(x = '', y = 'difference in mean confidence interval deviation between neutral and cued') +
  xlim(c(0.5,1.5)) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
ggsave(filename = paste0(figpath, '/confmu_groupaverage_diffs.eps'), device = 'eps', dpi = 600, height = 5, width = 5)
ggsave(filename = paste0(figpath, '/confmu_groupaverage_diffs.pdf'), device = 'pdf', dpi = 600, height = 5, width = 5)

df.confmu %>%
  dplyr::group_by(cond) %>%
  dplyr::summarise_at(.vars = 'absconfdiff', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = cond, y = mean, ymin = mean-se, ymax = mean+se, fill = cond)) +
  geom_bar(stat = 'identity', width = .7) +
  geom_errorbar(stat = 'identity', width = .35) +
  geom_point(inherit.aes = F, data = df.confmu, aes(x = cond, y = absconfdiff), size = .5) +
  geom_line(inherit.aes = F, data = df.confmu, aes(x = cond, y = absconfdiff, group = subid), size = .1) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = '', y = '(group) mean confidence interval deviation') +
  theme(legend.position = 'none')
ggsave(filename = paste0(figpath, '/confmu_groupaverage.pdf'), device = 'pdf', dpi = 600, height = 5, width = 5)
ggsave(filename = paste0(figpath, '/confmu_groupaverage.eps'), device = 'eps', dpi = 600, height = 5, width = 5)

anova_confmu <- afex::aov_ez(id = 'subid', data = df.confmu, 'absconfdiff', within = c('cond'))
nice(anova_confmu, es = 'pes') #significant main effect of cue (i.e. significant condition difference in confmu)

t.test(data = df.confmu, absconfdiff ~ cond, paired = TRUE)



#plot separately for trials where people were over vs underconfident
df.confmu.separate %>%
  dplyr::group_by(cond, targinconf) %>%
  dplyr::summarise_at(.vars = 'absconfdiff', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = cond, y = mean, ymin = mean-se, ymax = mean+se, fill = cond)) +
  geom_bar(stat = 'identity', width = .7) +
  geom_errorbar(stat = 'identity', width = .35) +
  geom_point(inherit.aes = F, data = df.confmu.separate, aes(x = cond, y = absconfdiff), size = .5) +
  geom_line(inherit.aes = F, data = df.confmu.separate, aes(x = cond, y = absconfdiff, group = subid), size = .1) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = '', y = '(group) mean confidence interval deviation') +
  theme(legend.position = 'none') +
  facet_wrap(~targinconf)
ggsave(filename = paste0(figpath, '/confmu_separate_groupaverage.pdf'), device = 'pdf', dpi = 600, height = 5, width = 11)
ggsave(filename = paste0(figpath, '/confmu_separate_groupaverage.eps'), device = 'eps', dpi = 600, height = 5, width = 11)

df.confmu.separate.diffs1 <- 
  df.confmu.separate %>%
  dplyr::filter(targinconf == 'Target Inside') %>% #dplyr::select(-targinconf) %>%
  dplyr::group_by(subid) %>%
  dplyr::summarise(diff = last(absconfdiff) - first(absconfdiff)) %>% #neutral - cued
  dplyr::mutate(targinconf = 'Target Inside')

df.confmu.separate.diffs2 <-
  df.confmu.separate %>%
  dplyr::filter(targinconf == 'Target Outside') %>%
  dplyr::group_by(subid) %>%
  dplyr::summarise(diff = last(absconfdiff) - first(absconfdiff)) %>% #neutral - cued
  dplyr::mutate(targinconf = 'Target Outside')
  
df.confmu.separate.diffs <- dplyr::bind_rows(df.confmu.separate.diffs1,df.confmu.separate.diffs2) %>%
  dplyr::mutate(targinconf = as.factor(targinconf), #make factor for grouping in plot
                subid = as.factor(subid)) 

#plot of differences (shows distribution of effects more clearly than line paths between points)
df.confmu.separate.diffs %>%
  dplyr::group_by(targinconf) %>% #to get the group average...
  dplyr::summarise_at(.vars = 'diff', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = targinconf, y = mean, ymin = mean-se, ymax = mean+se)) +
  geom_bar(stat = 'identity', width = .7, fill = diffcol) +
  geom_errorbar(width = .35, size = 1) +
  geom_point(inherit.aes = F, data = df.confmu.separate.diffs, aes(x = targinconf, y = diff), position = position_jitter(.05)) +
  geom_hline(yintercept = 0, linetype = 'dashed', color = '#000000', size = 1) +
  labs(x='', y = 'Difference in mean confidence interval deviation (neutral - cued)')
ggsave(filename = paste0(figpath, '/confmu_separate_groupaverage_diffs.pdf'), device = 'pdf', dpi = 600, height = 5, width = 5)
ggsave(filename = paste0(figpath, '/confmu_separate_groupaverage_diffs.eps'), device = 'eps', dpi = 600, height = 5, width = 5)
  



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

df.confvar <- df %>% dplyr::filter(clickresp == 1) %>%
  dplyr::filter(DTcheck == 0) %>%
  dplyr::group_by(subid, cond) %>%
  dplyr::mutate(rconfwidth = rad(confwidth)) %>%
  summarise_at(.vars = c('rconfwidth'), .funs = c('sd')) %>% as.data.frame() %>%
  dplyr::mutate(acc_confwidth = 1/rconfwidth) %>% select(-rconfwidth)

# - - - --  - - - -

#quickly look at change of behaviour across trials
xlines = seq(1,256,by=32)
#this is a messy version where values are zscored
df %>%
  dplyr::mutate(cond2 = cond) %>%
  dplyr::mutate(cond2 = ifelse(cond2 == 'neutral', 'neutral2', 'cued2')) %>% dplyr::mutate(cond2 = as.factor(cond2)) %>%
  dplyr::group_by(subid, cond) %>%
  dplyr::mutate(absrdif = scale(absrdif), confdiff = scale(confdiff)) %>% #this just zscores to make things comparable
  #but i dont want this comparable, i just want to look at confdiff across trials for now
  ggplot() +
  geom_vline(xintercept = xlines, linetype = 'dashed') +
  #geom_smooth(aes(x = trialnum, y = absrdif, color = cond), formula = y~s(x, bs = 'cr'), size = .5, method = 'gam') +
  #scale_color_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  geom_smooth(aes(x = trialnum, y = confdiff, color = cond2), formula = y~s(x, bs = 'cr'), size = .5, method = 'gam') +
  scale_color_manual(values = c('neutral' = neutcol, 'cued' = cuedcol, 'neutral2' = '#9e9ac8', 'cued2' = '#54278f')) + #change colouring of neutral/cued just for confdiff
  facet_wrap(subid~cond, scales = 'free')

#this plot will look at confdiff across trials, unscaled (so sign and value still has meaning)
df %>%
  dplyr::mutate(cond = ifelse(cond == 'neutral', 'Neutral', 'Cued')) %>% #just makes relabelling the legend a lot easier
  ggplot() +
  geom_vline(xintercept = xlines, linetype = 'dashed', colour = '#000000', size = .2) + #marks when people have received feedback so you can see changepoints
  #geom_line(aes(x = trialnum, y = confdiff, color = cond), size = .5) +
  geom_smooth(aes(x = trialnum, y = confdiff, color = cond), method = 'auto', size = .5, fill = '#bdbdbd', span=.2) +
  #stat_smooth(aes(x = trialnum, y = confdiff, color = cond), method = 'gam', formula = y~s(x, bs = 'cr'), fill = '#bdbdbd') +
  scale_color_manual(values = c('Neutral' = neutcol, 'Cued' = cuedcol), name = 'Condition') +
  facet_wrap(~subid, nrow = 4, ncol = 5) +
  labs(x = 'Trial Number', y = 'Confidence interval deviation') +
  theme(
    axis.text    = element_text(family = 'Source Sans Pro', colour = '#000000', size = 18),
    axis.title   = element_text(family = 'Source Sans Pro', colour = '#000000', size = 30),
    panel.border = element_rect(size = 1.5, color = '#000000'),
    legend.title = element_text(family = 'Source Sans Pro', colour = '#000000', size = 24),
    legend.text  = element_text(family = 'Source Sans Pro', colour = '#000000', size = 24),
    strip.text.x = element_blank(),
    panel.spacing.x = unit(1, 'lines')
  )
ggsave(filename = paste0(figpath, '/confdiff_trialwise.pdf'), device = cairo_pdf, dpi = 600, height = 12, width = 18)
ggsave(filename = paste0(figpath, '/confdiff_trialwise.eps'), device = cairo_ps , dpi = 600, height = 12, width = 18)



#each subject, confdiff (i.e. negative is overconfident, positive underconfident) across blocks
df %>% 
  dplyr::filter(clickresp ==1, confclicked == 1, DTcheck ==0) %>%
  dplyr::group_by(subid, cond, block) %>%
  dplyr::summarise_at(.vars = 'confdiff', .funs = 'mean') %>%
  ggplot(aes(x = block, y = confdiff, color = cond)) +
  geom_point(size = .5, alpha = .5) +
  geom_line(size = .5) +
  scale_color_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  facet_wrap(~subid)

#group confdiff across blocks
df %>% 
  dplyr::group_by(subid, cond, block) %>%
  dplyr::summarise_at(.vars = 'confdiff', .funs = 'mean') %>%
  dplyr::group_by(cond, block) %>%
  summarise_at(.vars = 'confdiff', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = block, y = mean, ymin = mean-se, ymax = mean+se, color = cond)) +
  geom_point(size = 1) +
  geom_line(size = 1) +
  scale_color_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  geom_errorbar(width = .1, size = .5, color = '#000000') +
  labs(x = 'block number', y = 'average confidence deviation across the block')
ggsave(filename = paste0(figpath, '/confdiff_blockwise.pdf'), device = 'pdf', dpi = 600, height = 10, width = 10)
ggsave(filename = paste0(figpath, '/confdiff_blockwise.eps'), device = 'eps', dpi = 600, height = 10, width = 10)


df %>% 
  dplyr::filter(clickresp ==1, confclicked == 1, DTcheck ==0) %>%
  dplyr::group_by(subid, cond, block) %>%
  dplyr::summarise_at(.vars = 'confdiff', .funs = 'mean') %>%
  ggplot(aes(x = block, y = confdiff, color = cond)) +
  geom_point(size = .5, alpha = .5) +
  geom_line(size = .5) +
  scale_color_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  facet_wrap(~subid, nrow = 4, ncol = 5)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


lmm.data <- df %>%
  dplyr::filter(subid != 12 & subid != 8) %>% droplevels(.) %>%
  dplyr::mutate(confwidth = rad(confwidth), absrdif = rad(absrdif)) %>%
  dplyr::filter(clickresp == 1, DTcheck == 0, confclicked == 1) %>% #just to be sure, but it should be fine
  dplyr::mutate(subid = as.factor(subid)) %>%
  dplyr::mutate(cond = as.factor(cond)) %>%
  dplyr::mutate(cond = relevel(cond, ref = 'neutral'))

#make the plot of raw data from here
#need to use ribbons and lines if you want to export to sketch as eps files cant output transparency
lmm.data %>%
  ggplot(aes(x = absrdif, y = confwidth)) +
  geom_ribbon(stat = 'smooth', method = 'lm', aes(fill = cond)) +
  geom_line(  stat = 'smooth', method = 'lm', aes(colour = cond)) +
  geom_point(size = .5, aes(colour = cond)) +
  #geom_smooth(method = 'lm', size = .5) +
  scale_color_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  scale_fill_manual(values = c('neutral' = '#d9d9d9', 'cued' = '#d9d9d9')) +
  geom_abline(intercept = 0, slope = 1, linetype = 'dashed', color = '#636363') +
  facet_wrap(~subid,  nrow = 4, ncol = 5)
ggsave(filename = paste0(figpath, '/absrdif~confwidth.pdf'), device = 'pdf', dpi = 600, height = 10, width = 10)
ggsave(filename = paste0(figpath, '/absrdif~confwidth.eps'), device = 'eps', dpi = 600, height = 10, width = 10)

contrasts(lmm.data$cond) <- contr.sum(2)

# are my DV normally distr
# determine lambda for confwidth
# temp fix for 0 values in the confwidth
lmm.data$confwidth[lmm.data$confwidth==0] <- 0.0001
lambdaList <- boxcox(confwidth~cond, data=lmm.data)
(lambda <- lambdaList$x[which.max(lambdaList$y)])

# - - - - - - - - - - - - - - - - - - - - - - 
fullmodel_log <- lme4::lmer(data = lmm.data, log(confwidth) ~ absrdif + cond + absrdif:cond +
                              (1 + absrdif + cond + absrdif:cond|subid))
summary(rePCA(fullmodel_log))#check if model is degenerate

simpmodel1_log <-lme4::lmer(data = lmm.data, log(confwidth) ~ absrdif + cond + absrdif:cond  + (1 + absrdif + absrdif:cond|subid))
simpmodel2_log <- lme4::lmer(data = lmm.data, log(confwidth) ~ absrdif + cond + absrdif:cond + (1 + cond    + absrdif:cond|subid))
simpmodel3_log <- lme4::lmer(data = lmm.data, log(confwidth) ~ absrdif + cond + absrdif:cond + (1 + absrdif + cond|subid))
simpmodel4_log <- lme4::lmer(data = lmm.data, log(confwidth) ~ absrdif + cond + absrdif:cond + (1 + absrdif|subid))
simpmodel5_log <- lme4::lmer(data = lmm.data, log(confwidth) ~ absrdif + cond + absrdif:cond + (1 + cond|subid))
minmodel_log  <- lme4::lmer(data = lmm.data, log(confwidth) ~ absrdif + cond + absrdif:cond  + (1|subid))

summary(rePCA(simpmodel1_log))
summary(rePCA(simpmodel2_log))
summary(rePCA(simpmodel3_log))
summary(rePCA(simpmodel4_log)) #this is ok
summary(rePCA(simpmodel5_log)) #this is less ok
summary(rePCA(minmodel_log))

anova(fullmodel_log, simpmodel1_log) #simpmodel1 better?
anova(simpmodel1_log, simpmodel2_log) #simp2 better?
anova(simpmodel2_log, simpmodel3_log) #simp3 better?
anova(simpmodel3_log, simpmodel4_log) #simp4 better
anova(simpmodel4_log, simpmodel5_log) #simp5 better
anova(simpmodel5_log, minmodel_log) #minmodel better
anova(fullmodel_log, minmodel_log)

anova(fullmodel_log, simpmodel1_log, simpmodel2_log, simpmodel3_log, simpmodel4_log, simpmodel5_log)
                             
summary(fullmodel_log)
summary(minmodel_log)

library(remef)
fit <- keepef(fullmodel_log, fix = 'absrdif:cond1', grouping = TRUE)

lmm.data$fitted <- fit


lmm.data %>%
  dplyr::mutate(cond = ifelse(cond=='neutral', 'Neutral', 'Cued')) %>%
  ggplot(aes(x = absrdif, y = exp(fitted))) +
  geom_point(size = 1, aes(colour = cond)) +
  geom_ribbon(stat = 'smooth', method = 'lm', aes(fill = cond)) +
  geom_line(stat = 'smooth', method = 'lm', aes(colour=cond)) +
  scale_fill_manual(values = c('Neutral' = '#d9d9d9', 'Cued' = '#d9d9d9')) +
  scale_color_manual('Condition', values = c('Neutral' = neutcol, 'Cued' = cuedcol)) +
  labs(x = 'absolute deviation of response from target (radians)',
       y = 'value of confidence interval width from the model (radians)') +
  geom_abline(intercept = 0, slope = 1, linetype = 'dashed', color = '#636363')
ggsave(filename = paste0(figpath, '/absrdif~confwidth_fitted.pdf'), device = 'pdf', dpi = 600, height = 10, width = 10)
ggsave(filename = paste0(figpath, '/absrdif~confwidth_fitted.eps'), device = 'eps', dpi = 600, height = 10, width = 10)



#rerun the LMM with limited data, to allow us to look at data where response errors weren't massive (and could skew the interaction)
#plot empirical cumulative density function of the data to see where most of the data lays
cumsum <- stats::ecdf(lmm.data$absrdif)
maxval_cdf <- quantile(cumsum, .98)

lmm.data %>% 
  ggplot(aes(x = absrdif)) +
  stat_ecdf() +
  geom_hline(yintercept = .98, linetype = 'dashed') +
  geom_vline(xintercept = maxval_cdf, linetype = 'dashed') +
  labs(x = 'absolute deviation of response from target (radians)', y = 'cumulative density')

lmm.data2 <- lmm.data %>%
  dplyr::filter(absrdif <= maxval_cdf)

fullmodel_log_limited <- lme4::lmer(data = lmm.data2, log(confwidth) ~ absrdif + cond + absrdif:cond +
                                      (1 + absrdif + cond + absrdif:cond|subid))
summary(rePCA(fullmodel_log_limited))#check if model is degenerate

library(remef)
fit2 <- keepef(fullmodel_log_limited, fix = 'absrdif:cond1', grouping = TRUE)

lmm.data2$fitted <- fit2

# plot the group level effect in one plot (all trials all subjects, showing model effect)
lmm.data2 %>%
  dplyr::mutate(cond = ifelse(cond=='neutral', 'Neutral', 'Cued')) %>%
  ggplot(aes(x = absrdif, y = exp(fitted))) +
  geom_point(size = 1, aes(colour = cond)) +
  geom_ribbon(stat = 'smooth', method = 'lm', aes(fill = cond)) +
  geom_line(stat = 'smooth', method = 'lm', aes(colour=cond)) +
  scale_fill_manual(values = c('Neutral' = '#d9d9d9', 'Cued' = '#d9d9d9')) +
  scale_color_manual('Condition', values = c('Neutral' = neutcol, 'Cued' = cuedcol)) +
  labs(x = 'absolute deviation of response from target (radians)',
       y = 'value of confidence interval width from the model (radians)') +
  geom_abline(intercept = 0, slope = 1, linetype = 'dashed', color = '#636363')
ggsave(filename = paste0(figpath, '/absrdif~confwidth_fitted_limited.pdf'), device = 'pdf', dpi = 600, height = 10, width = 10)
ggsave(filename = paste0(figpath, '/absrdif~confwidth_fitted_limited.eps'), device = 'eps', dpi = 600, height = 10, width = 10)
dev.off()

#and just plot the raw data
lmm.data2 %>%
  ggplot(aes(x = absrdif, y = confwidth)) +
  geom_ribbon(stat = 'smooth', method = 'lm', aes(fill = cond)) +
  geom_line(  stat = 'smooth', method = 'lm', aes(colour = cond)) +
  geom_point(size = .5, aes(colour = cond)) +
  scale_color_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  scale_fill_manual(values = c('neutral' = '#d9d9d9', 'cued' = '#d9d9d9')) +
  geom_abline(intercept = 0, slope = 1, linetype = 'dashed', color = '#636363') +
  labs(x = 'absolute deviation of response from target (radians)', y = 'width of confidence interval (radians)') +
  theme(
    axis.text    = element_text(family = 'Source Sans Pro', colour = '#000000', size = 18),
    axis.title   = element_text(family = 'Source Sans Pro', colour = '#000000', size = 24),
    panel.border = element_rect(size = 1, color = '#000000'),
    legend.title = element_text(family = 'Source Sans Pro', colour = '#000000', size = 16),
    legend.text  = element_text(family = 'Source Sans Pro', colour = '#000000', size = 14),
    strip.text   = element_text(family = 'Source Sans Pro', colour = '#000000', size = 16) 
  ) +
  facet_wrap(~subid,  nrow = 4, ncol = 5)
ggsave(filename = paste0(figpath, '/absrdif~confwidth_limited.pdf'), device = cairo_pdf, dpi = 600, height = 10, width = 10)
ggsave(filename = paste0(figpath, '/absrdif~confwidth_limited.eps'), device = cairo_ps , dpi = 600, height = 10, width = 10)
  