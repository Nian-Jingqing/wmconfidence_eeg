#import libraries
library(tidyverse) # general data manipulation & plotting
library(magrittr)  # allows use of more pipes
library(afex)      # anovas etc
library(RePsychLing)
library(MASS)

loadfonts = T
if(loadfonts){
  library(extrafont)
  font_import() #yes to this
  # font_import(paths='/home/sammirc/Desktop/fonts')
  fonts()
  loadfonts(device = "pdf");
  loadfonts(device = 'postscript')
}

#set theme for plots to standardise them and make cleaner
theme_set(theme_bw() +
            theme(
              strip.background = element_blank(),
              axis.text    = element_text(family = 'Source Sans Pro', colour = '#000000', size = 18),
              axis.title   = element_text(family = 'Source Sans Pro', colour = '#000000', size = 18),
              panel.border = element_rect(size = 2, color = '#000000'),
              legend.title = element_text(family = 'Source Sans Pro', colour = '#000000', size = 16),
              legend.text  = element_text(family = 'Source Sans Pro', colour = '#000000', size = 14),
              strip.text   = element_text(family = 'Source Sans Pro', colour = '#000000', size = 16)
            ) 
)
cuedcol <- '#3182bd' #blue, use for cued
neutcol <- '#bdbdbd' #light-ish grey, use for neutral
diffcol <- '#B2DF8A' #for colouring of the bars of difference plots, third distinct colour

se <- function(x) sd(x)/sqrt(length(x))


wd <- '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'
setwd(wd)

dpath <- paste0(wd, '/data') #path to folder with behavioural data
figpath <- paste0(wd, '/figures/behaviour')

#get behavioural data
fpath <- paste0(dpath, '/datafiles/wmConfidence_BehaviouralData_All.csv')



df <- read.csv(fpath, header = T, as.is = T, sep = ',') 
nsubs <- length(unique(df$subid))
subs2use <- c(4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 24, 25, 26) #these subs are included in eeg analyses, so just work with these for behavioural analyses too
#these subs are included in eeg analyses, so just work with these for behavioural analyses too


df %>%
  dplyr::select(subid, confclicked, cond) %>%
  group_by(subid, cond) %>%
  dplyr::mutate(confclicked = (confclicked -1)*-1) %>%
  dplyr::summarise_at(.vars = c('confclicked'), .funs = c('sum')) %>%
  as.data.frame() %>%
  dplyr::mutate(confclicked = ifelse(subid %in% c(1,2), (confclicked/160)*100, confclicked)) %>%
  dplyr::mutate(confclicked = ifelse(subid == 3, (confclicked/128)*100, confclicked)) %>%
  dplyr::mutate(confclicked = ifelse(subid > 3, (confclicked/256)*100, confclicked)) %>%
  dplyr::mutate(confclicked = round(confclicked,1) )
#confclicked is the percentage of trials, within condition, where the participant did not click to confirm their confidence response

subs2use <- c(4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 24, 25, 26) #these subs are included in eeg analyses, so just work with these for behavioural analyses too
#these subs are included in eeg analyses, so just work with these for behavioural analyses too


df %<>% dplyr::filter(subid %in% subs2use) %>% #keep only subjects safe to use based on clicking to confirm response (this is a failsafe in case it's not coded later on)
  dplyr::filter(clickresp == 1) %>%
  dplyr::filter(confclicked == 1) %>%
  dplyr::filter(DTcheck == 0) #hard code this now
#trials excluded if didn't click to confirm response or confidence judgement, and DT outside a couple sd's of mean per subject and condition

dim(df) #9517 x 47

df %>%
  ggplot(aes(x = confwidth, fill = NULL, color = cond)) +
  geom_density(alpha = .5, adjust = 1, size=1.2, outline.type = 'upper') +
  scale_color_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'confidence report width') +
  facet_wrap(~subid)

#this is also a bit of a shit figure to be honest
df %>%
  ggplot(aes(x = confwidth, fill = cond)) +
  geom_histogram(stat = 'bin', binwidth = 3) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'confidence report width') +
  facet_wrap(subid ~ cond, ncol = 6) +
  theme(strip.text.x = element_text(size = 6),
        axis.text.x  = element_text(size = 8),
        axis.text.y  = element_text(size = 6))
ggsave(filename = paste0(figpath, '/confwidth_hist_20subs.pdf'), device = cairo_pdf, dpi = 600, height = 10, width = 10)
ggsave(filename = paste0(figpath, '/confwidth_hist_20subs.eps'), device =  cairo_ps, dpi = 600, height = 10, width = 10)

df %>%
  ggplot(aes(x = confwidth, fill = cond)) +
  geom_density(adjust = 1, outline.type = 'upper')+#, alpha = .4) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'confidence report width', y = '') +
  facet_wrap(~subid, ncol = 5) +
  theme(strip.text.x = element_text(size = 10),
        axis.text.x  = element_text(size = 10),
        axis.text.y  = element_text(size = 10))
ggsave(filename = paste0(figpath, '/confwidth_density_20subs.pdf'), device = cairo_pdf, dpi = 600, height = 10, width = 10)
ggsave(filename = paste0(figpath, '/confwidth_density_20subs.eps'), device = cairo_ps, dpi = 600, height = 10, width = 10)

#just look at raw confidence width here
df %>%
  dplyr::group_by(subid, cond) %>%
  summarise_at(.vars = 'confwidth', .funs = c('mean', 'sd')) %>%
  dplyr::group_by(cond) %>%
  summarise_at(.vars = c('mean', 'sd'), .funs = c('mean', 'se')) %>%
  ggplot(aes(x = cond, y = mean_mean, fill = cond)) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  geom_bar(stat = 'identity') +
  geom_errorbar(aes(ymin = mean_mean - mean_se, ymax = mean_mean + mean_se), width = .3, size=1, color = '#000000') +
  labs(y = 'confidence width (degrees)', x = 'cue condition') + 
  theme(legend.position = 'none')
ggsave(filename = paste0(figpath, '/gave_confwidthmean.pdf'), device = cairo_pdf, dpi = 600, height = 9, width = 9)
ggsave(filename = paste0(figpath, '/gave_confwidthmean.eps'), device = cairo_ps, dpi = 600, height = 9, width = 9)

#mean confidence width
df.cwmean <- df %>%
  dplyr::group_by(subid, cond) %>%
  summarise_at(.vars = 'confwidth', .funs = c('mean'))
aov.cwmean <- afex::aov_ez(id = 'subid', data = df.cwmean, 'confwidth', within = 'cond')
nice(aov.cwmean, es = 'pes')

# Anova Table (Type 3 tests)
# 
# Response: confwidth
# Effect    df  MSE         F pes p.value
# 1   cond 1, 19 0.37 17.84 *** .48   .0005
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘+’ 0.1 ‘ ’ 1

t.test(data = df.cwmean, confwidth ~ cond, paired = T)

# Paired t-test
# 
# data:  confwidth by cond
# t = -4.2241, df = 19, p-value = 0.0004593
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   -1.223208 -0.412652
# sample estimates:
#   mean of the differences 
# -0.8179302 


df.cwmean %>%
  dplyr::group_by(cond) %>%
  summarise_at(.vars = 'confwidth', .funs = c('mean', 'se'))


df %>%
  dplyr::group_by(subid, cond) %>%
  summarise_at(.vars = 'confwidth', .funs = c('mean', 'sd')) %>% as.data.frame(.) %>%
  dplyr::group_by(cond) %>%
  summarise_at(.vars = c('mean', 'sd'), .funs = c('mean', 'se')) %>%
  ggplot(aes(x = cond, y = sd_mean, fill = cond)) +
  geom_bar(stat = 'identity') +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  geom_errorbar(aes(ymin = sd_mean - sd_se, ymax = sd_mean + sd_se), width = .25, size = 1, color = '#000000') +
  labs(y = 'mean confidence width variability (SD)', x = 'cue condition')



df %>%
  ggplot(aes(x = confdiff, fill = cond)) +
  geom_histogram(aes(y = ..count../sum(..count..)*100),
                 binwidth = 4, #this sets a realistic resolution of people's use of the mouse
                 center = 0, #centre the bins around 0
                 alpha = 1, #no transparency for exporting to sketch
                 position = position_identity()) + #position_identity stops them stacking, so they're on same y scale (makes it easier to see equivalence
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  geom_vline(xintercept = 0, linetype = 'dashed') +
  labs(x = 'confidence error', y = '') +
  facet_wrap(~subid, ncol = 5) +
  theme(strip.text.x = element_text(size = 10),
        axis.text.x  = element_text(size = 10),
        axis.text.y  = element_text(size = 10))
ggsave(filename = paste0(figpath, '/confdiff_hist.pdf'), device = cairo_pdf, dpi = 600, height = 10, width = 10)
ggsave(filename = paste0(figpath, '/confdiff_hist.eps'), device = cairo_ps, dpi = 600, height = 10, width = 10)


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
  geom_errorbar(stat = 'identity', width = .15, size = 1) +
  geom_point(data = df.confacc.diffs, aes(x = 1, y = diff), inherit.aes = F,size = 1, position = position_jitter(width=.2,height=0)) + 
  geom_hline(yintercept = 0, linetype = 'dashed', color = '#000000', size = .5) +
  labs(x = '', y = 'difference in % trials where target inside\nconfidence interval (neutral - cued)') +
  xlim(c(0.5,1.5)) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
ggsave(filename = paste0(figpath, '/gave_targinconf_diffs.eps'), device = cairo_ps, dpi = 600, height = 8, width = 8)
ggsave(filename = paste0(figpath, '/gave_targinconf_diffs.pdf'), device = cairo_pdf, dpi = 600, height = 8, width = 8)



#plot group average with individual data points
df.confacc %>%
  dplyr::group_by(cond) %>%
  dplyr::summarise_at(.vars = 'targinconf', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = cond, y = mean, fill = cond)) +
  geom_bar(stat = 'identity', width = .7 ) +
  geom_errorbar(aes(ymin = mean-se, ymax = mean+se), width = .35, size = 1) +
  #geom_point(inherit.aes = F, data = df.confacc, aes(x = cond, y = targinconf), size = 1) +
  #geom_line(inherit.aes = F, data = df.confacc, aes(x = cond, y = targinconf, group = subid), size = .5) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = '', y = '% trials where target\ninside confidence interval') +
  theme(legend.position = 'none')
ggsave(filename = paste0(figpath, '/gave_targinconf.eps'), device = cairo_ps, dpi = 600, height = 9, width = 9)
ggsave(filename = paste0(figpath, '/gave_targinconf.pdf'), device = cairo_pdf, dpi = 600, height = 9, width = 9)
  
aov_confacc = afex::aov_ez(id = 'subid', data = df.confacc, 'targinconf', within = 'cond')
nice(aov_confacc, es = 'pes')

# Anova Table (Type 3 tests)
# 
# Response: targinconf
# Effect    df  MSE        F pes p.value
# 1   cond 1, 19 7.91 12.51 ** .40    .002
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘+’ 0.1 ‘ ’ 1


t.test(data = df.confacc, targinconf ~ cond, paired = T)

# Paired t-test
# 
# data:  targinconf by cond
# t = 3.5369, df = 19, p-value = 0.002203
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   1.284096 5.006855
# sample estimates:
#   mean of the differences 
# 3.145476 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

library(circular)
df.confdiffmu <- df %>%
  dplyr::mutate(cond = as.factor(cond)) %>%
  dplyr::mutate(absconfdiff = abs(confdiff)) %>% #need this to get mean absolute confidence error
  dplyr::group_by(subid, cond) %>%
  dplyr::summarise_at(.vars = 'absconfdiff', .funs = mean)

df.confdiffmu.separate <- df %>%
  dplyr::mutate(cond = as.factor(cond)) %>%
  dplyr::mutate(targinconf = ifelse(confdiff <= 0, 'Target Inside', 'Target Outside')) %>%
  dplyr::mutate(targinconf = as.factor(targinconf)) %>%
  dplyr::mutate(subid = as.factor(subid)) %>%
  dplyr::mutate(absconfdiff = abs(confdiff)) %>% #need this to get mean absolute confidence error
  dplyr::group_by(subid, cond, targinconf) %>%
  dplyr::summarise_at(.vars = 'absconfdiff', .funs = mean)
  

df.confdiffmu.diffs <- df.confdiffmu %>%
  dplyr::group_by(subid) %>%
  dplyr::summarise(diff = last(absconfdiff) - first(absconfdiff))
#here, positive values mean that neutral was higher than cued (i.e. on average less accurate with confidence)

#plot as one difference bar, rather than separate condition bars
df.confdiffmu.diffs %>%
  dplyr::summarise_at(.vars = 'diff', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = 1, y = mean, ymin = mean-se, ymax = mean + se)) +
  geom_bar(stat = 'identity', width = .4, fill = diffcol) + 
  geom_errorbar(stat = 'identity', width = .15, size = 1) +
  geom_point(data = df.confdiffmu.diffs, aes(x = 1, y = diff), inherit.aes = F,size = 1, position = position_jitter(width=.2,height=0)) + 
  geom_hline(yintercept = 0, linetype = 'dashed', color = '#000000', size = .5) +
  labs(x = '', y = 'difference in mean confidence error\nbetween neutral and cued') +
  xlim(c(0.5,1.5)) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
ggsave(filename = paste0(figpath, '/gave_confdiffmean_diffs.eps'), device = cairo_ps,  dpi = 600, height = 9, width = 9)
ggsave(filename = paste0(figpath, '/gave_confdiffmean_diffs.pdf'), device = cairo_pdf, dpi = 600, height = 9, width = 9)

df.confdiffmu %>%
  dplyr::group_by(cond) %>%
  dplyr::summarise_at(.vars = 'absconfdiff', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = cond, y = mean, ymin = mean-se, ymax = mean+se, fill = cond)) +
  geom_bar(stat = 'identity', width = .7) +
  geom_errorbar(stat = 'identity', width = .35, size = 1) +
  #geom_point(inherit.aes = F, data = df.confmu, aes(x = cond, y = absconfdiff), size = 1) +
  #geom_line(inherit.aes = F, data = df.confmu, aes(x = cond, y = absconfdiff, group = subid), size = .5) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = '', y = 'mean confidence error') +
  theme(legend.position = 'none')
ggsave(filename = paste0(figpath, '/gave_confdiffmu.pdf'), device = cairo_pdf, dpi = 600, height = 9, width = 9)
ggsave(filename = paste0(figpath, '/gave_confdiffmu.eps'), device = cairo_ps,  dpi = 600, height = 9, width = 9)

anova_confdiffmu <- afex::aov_ez(id = 'subid', data = df.confdiffmu, 'absconfdiff', within = c('cond'))
nice(anova_confdiffmu, es = 'pes') #significant main effect of cue (i.e. significant condition difference in confmu)

# Anova Table (Type 3 tests)
# 
# Response: absconfdiff
# Effect    df  MSE      F pes p.value
# 1   cond 1, 19 0.31 6.69 * .26     .02
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘+’ 0.1 ‘ ’ 1

t.test(data = df.confdiffmu, absconfdiff ~ cond, paired = TRUE)

# Paired t-test
# 
# data:  absconfdiff by cond
# t = -2.5869, df = 19, p-value = 0.01808
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#   -0.82989088 -0.08758045
# sample estimates:
#   mean of the differences 
# -0.4587357 



#plot separately for trials where people were over vs underconfident
df.confdiffmu.separate %>%
  dplyr::group_by(cond, targinconf) %>%
  dplyr::summarise_at(.vars = 'absconfdiff', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = cond, y = mean, ymin = mean-se, ymax = mean+se, fill = cond)) +
  geom_bar(stat = 'identity', width = .7) +
  geom_errorbar(stat = 'identity', width = .35, size = 1) +
  #geom_point(inherit.aes = F, data = df.confmu.separate, aes(x = cond, y = absconfdiff), size = 1) +
  #geom_line(inherit.aes = F, data = df.confmu.separate, aes(x = cond, y = absconfdiff, group = subid), size = .5) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = '', y = 'mean confidence error') +
  theme(legend.position = 'none') +
  facet_wrap(~targinconf)
ggsave(filename = paste0(figpath, '/gave_confdiffmu_separate.pdf'), device = cairo_pdf, dpi = 600, height = 9, width = 20)
ggsave(filename = paste0(figpath, '/gave_confdiffmu_separate.eps'), device = cairo_ps, dpi = 600,  height = 9, width = 20)

df.confdiffmu.separate.diffs1 <- 
  df.confdiffmu.separate %>%
  dplyr::filter(targinconf == 'Target Inside') %>% #dplyr::select(-targinconf) %>%
  dplyr::group_by(subid) %>%
  dplyr::summarise(diff = last(absconfdiff) - first(absconfdiff)) %>% #neutral - cued
  dplyr::mutate(targinconf = 'Target Inside')

df.confdiffmu.separate.diffs2 <-
  df.confdiffmu.separate %>%
  dplyr::filter(targinconf == 'Target Outside') %>%
  dplyr::group_by(subid) %>%
  dplyr::summarise(diff = last(absconfdiff) - first(absconfdiff)) %>% #neutral - cued
  dplyr::mutate(targinconf = 'Target Outside')
  
df.confdiffmu.separate.diffs <- dplyr::bind_rows(df.confdiffmu.separate.diffs1, df.confdiffmu.separate.diffs2) %>%
  dplyr::mutate(targinconf = as.factor(targinconf), #make factor for grouping in plot
                subid = as.factor(subid)) 

#plot of differences (shows distribution of effects more clearly than line paths between points)
df.confdiffmu.separate.diffs %>%
  dplyr::group_by(targinconf) %>% #to get the group average...
  dplyr::summarise_at(.vars = 'diff', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = targinconf, y = mean, ymin = mean-se, ymax = mean+se)) +
  geom_bar(stat = 'identity', width = .7, fill = diffcol) +
  geom_errorbar(width = .35, size = 1) +
  geom_point(inherit.aes = F, data = df.confdiffmu.separate.diffs, aes(x = targinconf, y = diff), position = position_jitter(width=.2,height=0)) +
  geom_hline(yintercept = 0, linetype = 'dashed', color = '#000000', size = 1) +
  labs(x='', y = 'Difference in mean confidence interval deviation\n(neutral - cued)')
ggsave(filename = paste0(figpath, '/gave_confdiffmu_separate_diffs.pdf'), device = cairo_pdf, dpi = 600, height = 9, width = 9)
ggsave(filename = paste0(figpath, '/gave_confdiffmu_separate_diffs.eps'), device = cairo_ps,  dpi = 600, height = 9, width = 9)
  
anova_dfconfdiffmu <- afex::aov_ez('subid', data = df.confdiffmu.separate, 'absconfdiff', within = c('targinconf', 'cond'))
nice(anova_dfconfdiffmu, es = 'pes')
# Anova Table (Type 3 tests)
# Effect    df   MSE       F  pes p.value
# 1      targinconf 1, 19 23.31 9.75 **  .34    .006
# 2            cond 1, 19  2.93    2.52  .12     .13
# 3 targinconf:cond 1, 19  2.62    0.09 .005     .77
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘+’ 0.1 ‘ ’ 1





library(BayesFactor)
aovbf.data <- as.data.frame(df.confdiffmu.separate) %>% dplyr::mutate(subid = factor(subid))
aovbf_dfconfdiffmu <-  BayesFactor::anovaBF(absconfdiff ~ subid + cond + targinconf + cond:targinconf + subid:cond + subid:targinconf, data = aovbf.data,
                                        whichRandom = 'subid', iterations = 100000)
aovbf_dfconfdiffmu[4]/aovbf_dfconfdiffmu[3] #strong evidence against the 2 way interaction here (i.e. it doesn't exist really )
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


lmm.data <- df %>%
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
  facet_wrap(~subid,  nrow = 5)
ggsave(filename = paste0(figpath, '/absrdif~confwidth.pdf'), device = cairo_pdf, dpi = 600, height = 10, width = 10)
ggsave(filename = paste0(figpath, '/absrdif~confwidth.eps'), device = cairo_ps, dpi = 600, height = 10, width = 10)

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
minmodel_log  <- lme4::lmer(data = lmm.data, log(confwidth) ~ absrdif + cond + absrdif:cond  + (1|subid))
summary(rePCA(minmodel_log))
anova(fullmodel_log, minmodel_log)
                             
summary(fullmodel_log) # condition x error interaction: absrdif:cond1 -0.047282   0.024187  -1.955 (beta, std err, t value)
summary(minmodel_log)  # condition x error interaction: absrdif:cond1 -0.067720   0.022982  -2.947 (beta, std err, t value)

# > summary(minmodel_log)  # condition x error interaction: absrdif:cond1 -0.067720   0.022982  -2.947 (beta, std err, t value)
# Linear mixed model fit by REML ['lmerMod']
# Formula: log(confwidth) ~ absrdif + cond + absrdif:cond + (1 | subid)
# Data: lmm.data
# 
# REML criterion at convergence: 8906.4
# 
# Scaled residuals: 
#   Min       1Q   Median       3Q      Max 
# -22.3562  -0.5616   0.0136   0.6022   4.4983 
# 
# Random effects:
#   Groups   Name        Variance Std.Dev.
# subid    (Intercept) 0.08245  0.2871  
# Residual             0.14721  0.3837  
# Number of obs: 9517, groups:  subid, 20
# 
# Fixed effects:
#   Estimate Std. Error t value
# (Intercept)   -1.367456   0.064454 -21.216
# absrdif        0.538525   0.023817  22.611
# cond1          0.024759   0.005538   4.471
# absrdif:cond1 -0.067720   0.022982  -2.947
# 
# Correlation of Fixed Effects:
#   (Intr) absrdf cond1 
# absrdif     -0.063              
# cond1       -0.001  0.049       
# absrdf:cnd1  0.005 -0.129 -0.703



library(remef)
fit <- keepef(minmodel_log, fix = 'absrdif:cond1', grouping = TRUE)

lmm.data$fitted <- fit


lmm.data %>%
  dplyr::mutate(cond = ifelse(cond=='neutral', 'Neutral', 'Cued')) %>%
  ggplot(aes(x = absrdif, y = confwidth)) + #y = exp(fitted))) +
  #geom_point(size = 1, aes(colour = cond)) +
  #geom_ribbon(stat = 'smooth', method = 'lm', aes(fill = cond), alpha=.5) +
  #geom_line(stat = 'smooth', method = 'lm', aes(colour=cond)) +
  geom_ribbon(inherit.aes = F, aes(x = absrdif, y = exp(fitted), fill = cond), stat = 'smooth', method = 'lm') +
  geom_line(inherit.aes = F, aes(x = absrdif, y = exp(fitted), color = cond), stat = 'smooth', method = 'lm', size = 1) +
  scale_fill_manual(values = c('Neutral' = '#d9d9d9', 'Cued' = '#d9d9d9'), name = NULL, labels = NULL, guide = "none") +
  scale_color_manual('Condition', values = c('Neutral' = neutcol, 'Cued' = cuedcol)) +
  labs(x = 'absolute reponse error (radians)',
       y = 'confidence interval width (radians)') +
  geom_abline(intercept = 0, slope = 1, linetype = 'dashed', color = '#636363') +
  coord_cartesian(xlim = c(0, 1.6), ylim = c(0,1.6)) + theme(legend.position = 'top')
ggsave(filename = paste0(figpath, '/absrdif~confwidth_fittedline_fromMinModel.pdf'), device = cairo_pdf, dpi = 600, height = 10, width = 10)
ggsave(filename = paste0(figpath, '/absrdif~confwidth_fittedline_fromMinModel.eps'), device = cairo_ps, dpi = 600, height = 10, width = 10)


#%%  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
df %>%
  dplyr::mutate(trladj = confwidth - prevtrlcw) %>%
  dplyr::filter(trialnum != 1) %>% #exclude first trial of every session of each participant as this doesn't have a previous trial
  dplyr::mutate(prevtrlinconf = ifelse(prevtrlconfdiff <= 0, 'underconfident','overconfident')) %>%
  dplyr::mutate(as.factor(prevtrlinconf)) %>%
  ggplot(aes(x=prevtrlconfdiff, y = trladj)) +
  geom_point(size=.5) +
  geom_smooth(method = 'lm', aes(color = prevtrlinconf)) + 
  labs(y = 'confidence adjustment (currtrial - prevtrl confwidth',
       x = 'previous trial confidence error') +
  facet_wrap(~subid)

df %>%
  dplyr::mutate(trladj = absrdif - prevtrlabsrdif) %>%
  dplyr::filter(trialnum != 1) %>%
  dplyr::mutate(prevtrlinconf = ifelse(prevtrlconfdiff <= 0, 1, 0)) %>%
  dplyr::mutate(prevtrlinconf = as.factor(prevtrlinconf)) %>%
  dplyr::mutate(prevtrlinconf = ifelse(prevtrlinconf == 1, 'underconfident', 'overconfident')) %>%
  dplyr::mutate(prevtrlinconf = as.factor(prevtrlinconf)) %>%
  ggplot(aes(x=prevtrlconfdiff, y = trladj)) +
  geom_point(size=.5) +
  geom_smooth(method = 'lm', aes(color = prevtrlinconf)) + 
  scale_color_manual(values = c('underconfident' = '#4daf4a', 'overconfident' = '#e41a1c')) +
  labs(y = 'error adjustment (currtrial - prevtrl absolute error)',
       x = 'previous trial confidence error') +
  facet_wrap(~subid)


#this is just to check if it is normally distributed or not
df %>%
  dplyr::filter(trialnum != 1) %>% #exclude first trial of every session of each participant as this doesn't have a previous trial
  dplyr::mutate(trladj = confwidth -prevtrlcw) %>%
  ggplot(aes(x=trladj)) + geom_density() + facet_wrap(~subid)

lmm.trladjcw.data <- df %>%
  dplyr::mutate(prevtrlinconf = ifelse(prevtrlconfdiff <= 0, 1, 0)) %>%
  dplyr::mutate(prevtrlinconf = as.factor(prevtrlinconf)) %>%
  dplyr::mutate(trladj = confwidth - prevtrlcw) %>% # difference in current trials confidence compared to the previous trial
  dplyr::filter(trialnum != 1) %>% #exclude first trial of each session as no prev trial for it (these vals would be NA anyway)
  dplyr::filter(!is.na(prevtrlconfdiff))


#check correlation between confidence width and the update
# this should be correlated really, because the computation of the adjustment includes the previous trial confidence width anyways
lmm.trladjcw.data %>% ggplot(aes(x = prevtrlcw, y = trladj)) +
  geom_point(size=.3) +
  geom_smooth(method='lm') +
  facet_wrap(~subid) +
  labs(y = 'trialwise adjustment in confidence/n(negative = become more confident)',
       x = 'previous trial confidence width')


lmm.trladjerr.data <- df %>%
  dplyr::mutate(prevtrlinconf = ifelse(prevtrlconfdiff <= 0, 1, 0)) %>%
  dplyr::mutate(prevtrlinconf = as.factor(prevtrlinconf)) %>%
  dplyr::mutate(trladj = absrdif - prevtrlabsrdif) %>% # difference in current trials confidence compared to the previous trial
  dplyr::filter(trialnum != 1) %>% #exclude first trial of each session as no prev trial for it (these vals would be NA anyway)
  dplyr::filter(!is.na(prevtrlconfdiff))

# lmm.trladjcw.data %>%
#   group_by(subid) %>%
#   dplyr::mutate(prevtrlinconf = ifelse(prevtrlconfdiff <= 0, 1, 0)) %>%
#   dplyr::mutate(prevtrlinconf = as.factor(prevtrlinconf)) %>%
#   summarise(cor = cor(prevtrlconfdiff, trladj)) %>%
#   summarise_at(.vars='cor', .funs = c('mean', 'se'))

contrasts(lmm.trladjcw.data$prevtrlinconf) <- contr.sum(2)
contrasts(lmm.trladjerr.data$prevtrlinconf) <- contr.sum(2)

lmm.trladjcw.full <- lme4::lmer(data = lmm.trladjcw.data,
                                trladj ~ prevtrlconfdiff + prevtrlcw +
                                (1 + prevtrlconfdiff + prevtrlcw | subid))
lmm.trladjcw.min  <- lme4::lmer(data = lmm.trladjcw.data,
                                trladj ~ prevtrlconfdiff + prevtrlcw + (1|subid))
summary(lmm.trladjcw.full)
# > summary(lmm.trladjcw.full)
# Linear mixed model fit by REML ['lmerMod']
# Formula: trladj ~ prevtrlconfdiff + prevtrlcw + (1 + prevtrlconfdiff +      prevtrlcw | subid)
# Data: lmm.trladjcw.data
# 
# REML criterion at convergence: 65613.7
# 
# Scaled residuals: 
#   Min      1Q  Median      3Q     Max 
# -4.2043 -0.5277 -0.1291  0.3628  8.5922 
# 
# Random effects:
#   Groups   Name            Variance  Std.Dev. Corr       
# subid    (Intercept)     3.724e+01 6.10206             
# prevtrlconfdiff 1.291e-04 0.01136   0.99      
# prevtrlcw       6.514e-03 0.08071  -0.24 -0.20
# Residual                 5.783e+01 7.60428             
# Number of obs: 9494, groups:  subid, 20
# 
# Fixed effects:
#   Estimate Std. Error t value
# (Intercept)     15.393630   1.380431  11.151
# prevtrlconfdiff  0.043461   0.008396   5.176
# prevtrlcw       -0.840411   0.021777 -38.592
# 
# Correlation of Fixed Effects:
#   (Intr) prvtrlcn
# prvtrlcnfdf  0.279         
# prevtrlcw   -0.272  0.165  
# convergence code: 0
# unable to evaluate scaled gradient
# Model failed to converge: degenerate  Hessian with 1 negative eigenvalues


summary(lmm.trladjcw.min)
# > summary(lmm.trladjcw.min)
# Linear mixed model fit by REML ['lmerMod']
# Formula: trladj ~ prevtrlconfdiff + prevtrlcw + (1 | subid)
# Data: lmm.trladjcw.data
# 
# REML criterion at convergence: 65666.6
# 
# Scaled residuals: 
#   Min      1Q  Median      3Q     Max 
# -4.0324 -0.5284 -0.1314  0.3653  8.6031 
# 
# Random effects:
#   Groups   Name        Variance Std.Dev.
# subid    (Intercept) 24.18    4.917   
# Residual             58.35    7.639   
# Number of obs: 9494, groups:  subid, 20
# 
# Fixed effects:
#   Estimate Std. Error t value
# (Intercept)     15.471133   1.115593  13.868
# prevtrlconfdiff  0.043767   0.007981   5.484
# prevtrlcw       -0.844348   0.010457 -80.745
# 
# Correlation of Fixed Effects:
#   (Intr) prvtrlcn
# prvtrlcnfdf -0.021         
# prevtrlcw   -0.145  0.460  

fit.trladjcw <- keepef(lmm.trladjcw.full, fix = c('prevtrlconfdiff', 'prevtrlcw'), grouping=T)
lmm.trladjcw.data$fitted <- fit.trladjcw

#plot the fitted trialwise adjustments from this lmm
#this will get the fitted values per trial across subjects, having controlled for the within and between subject variance of the model
lmm.trladjcw.data %>%
  ggplot(aes(x = prevtrlconfdiff, y = trladj)) +
  #geom_point(size = .5, color = '#bdbdbd') +
  #geom_smooth(inherit.aes = F, aes(x = prevtrlconfdiff, y = fitted), method = 'lm', color = '#756bb1') +
  geom_ribbon(inherit.aes = F, aes(x = prevtrlconfdiff, y = fitted), stat = 'smooth', method = 'lm', color = '#756bb1') +
  geom_line(inherit.aes = F, aes(x = prevtrlconfdiff, y = fitted), stat = 'smooth', method = 'lm', color = '#756bb1', size = 1) +
  labs(y = 'confidence adjustment\n(current trial - previous trial confidence width',
       x = 'previous trial confidence error') +
  coord_cartesian(ylim = c(-80, 80), xlim = c(-80, 80)) + #only add theme_bw if changing the device from cairo to normal to export into sketch
ggsave(filename = paste0(figpath, '/trladjustment_confidence_prevtrlconferr_agg.pdf'), device = cairo_pdf, dpi = 600, height = 9, width = 9)
ggsave(filename = paste0(figpath, '/trladjustment_confidence_prevtrlconferr_agg.eps'), device = cairo_ps, dpi = 600, height = 9, width = 9)

lmm.trladjcw.data %>%
  ggplot(aes(x = prevtrlconfdiff, y = trladj)) +
  geom_point(size = .5, color = '#bdbdbd') +
  geom_smooth(inherit.aes = F, aes(x = prevtrlconfdiff, y = fitted), method = 'lm', color = '#756bb1') +
  labs(y = 'confidence adjustment\n(current trial - previous trial confidence width',
       x = 'previous trial confidence error')#only add theme_bw if changing the device from cairo to normal to export into sketch

lmm.trladjerr.full <- lme4::lmer(data = lmm.trladjerr.data,
                                 trladj ~ prevtrlconfdiff + prevtrlabsrdif + 
                                   (1 + prevtrlconfdiff + prevtrlabsrdif | subid))
lmm.trladjerr.min <- lme4::lmer(data = lmm.trladjerr.data,
                                trladj ~ prevtrlconfdiff + prevtrlabsrdif + (1 + prevtrlconfdiff + prevtrlabsrdif |  subid))


summary(lmm.trladjerr.full)
summary(lmm.trladjerr.min)

fit.trladjerr <- keepef(lmm.trladjerr.full, fix = c('prevtrlconfdiff', 'prevtrlabsrdif'), grouping=T)
lmm.trladjerr.data$fitted <- fit.trladjerr


lmm.trladjerr.data %>%
  dplyr::mutate(prevtrlinconf = ifelse(prevtrlconfdiff <= 0, 1, 0)) %>%
  dplyr::mutate(prevtrlinconf = as.factor(prevtrlinconf)) %>%
  dplyr::mutate(prevtrlinconf = ifelse(prevtrlinconf == 1, 'underconfident', 'overconfident')) %>%
  dplyr::mutate(prevtrlinconf = as.factor(prevtrlinconf)) %>%
  ggplot(aes(x=prevtrlconfdiff, y = trladj)) +
  geom_point(size=.5, color = '#bdbdbd', alpha = .5) +
  geom_smooth(method = 'lm', size=.7, color = '#000000') +
  scale_color_manual(values = c('underconfident' = '#4daf4a', 'overconfident' = '#e41a1c')) +
  labs(y = 'error adjustment\n(current trial - previous trial error)',
       x = 'previous trial confidence error') +
  facet_wrap(~subid)
ggsave(filename = paste0(figpath, '/trladjustment_error_prevtrlconferr_persub.pdf'), device = cairo_pdf, dpi = 600, height = 12, width = 18)
ggsave(filename = paste0(figpath, '/trladjustment_error_prevtrlconferr_persub.eps'), device = cairo_ps , dpi = 600, height = 12, width = 18)

lmm.trladjerr.data %>%
  dplyr::mutate(prevtrlinconf = ifelse(prevtrlconfdiff <= 0, 1, 0)) %>%
  dplyr::mutate(prevtrlinconf = as.factor(prevtrlinconf)) %>%
  dplyr::mutate(prevtrlinconf = ifelse(prevtrlinconf == 1, 'underconfident', 'overconfident')) %>%
  dplyr::mutate(prevtrlinconf = as.factor(prevtrlinconf)) %>%
  ggplot(aes(x=prevtrlconfdiff, y = trladj)) +
  geom_point(size=.5, color = '#bdbdbd', alpha = .5) +
  geom_smooth(method = 'lm', size=.7, color = '#000000') +
  scale_color_manual(values = c('underconfident' = '#4daf4a', 'overconfident' = '#e41a1c')) +
  labs(y = 'error adjustment\n(current trial - previous trial error)',
       x = 'previous trial confidence error') 
ggsave(filename = paste0(figpath, '/trladjustment_error_prevtrlconferr_20subs.pdf'), device = cairo_pdf, dpi = 600, height = 12, width = 18)
ggsave(filename = paste0(figpath, '/trladjustment_error_prevtrlconferr_20subs.eps'), device = cairo_ps , dpi = 600, height = 12, width = 18)

lmm.trladjerr.data %>%
  dplyr::mutate(prevtrlinconf = ifelse(prevtrlconfdiff <= 0, 1, 0)) %>%
  dplyr::mutate(prevtrlinconf = as.factor(prevtrlinconf)) %>%
  dplyr::mutate(prevtrlinconf = ifelse(prevtrlinconf == 1, 'underconfident', 'overconfident')) %>%
  dplyr::mutate(prevtrlinconf = as.factor(prevtrlinconf)) %>%
  ggplot(aes(x=prevtrlabsrdif, y = trladj)) +
  geom_point(size=.5, color = '#bdbdbd', alpha = .5) +
  geom_smooth(method = 'lm', size=.7, color = '#000000') +
  scale_color_manual(values = c('underconfident' = '#4daf4a', 'overconfident' = '#e41a1c')) +
  labs(y = 'error adjustment\n(current trial - previous trial error)',
       x = 'previous trial response error') 



