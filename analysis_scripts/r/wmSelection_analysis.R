#import libraries
library(tidyverse) # general data manipulation & plotting
library(magrittr)  # allows use of more pipes
library(afex)      # anovas etc

#set theme for plots to standardise them and make cleaner
theme_set(theme_bw() +
            theme(
              axis.text        = element_text(size=12),
              legend.text      = element_text(size=12),
              strip.text.x     = element_text(size=12),
              strip.background = element_blank()
            ))

#hex codes for colours to use in figures:
# #3182bd -- blue, use for cued
# #bdbdbd -- light-ish grey, use for neutral

cuedcol <- '#3182bd'
neutcol <- '#bdbdbd'

se <- function(x) sd(x)/sqrt(length(x))


wd <- '/Users/sammi/Desktop/Experiments/DPhil/wmSelection'
setwd(wd)

dpath <- paste0(wd, '/data') #path to folder with behavioural data
figpath <- paste0(wd, '/figures')

#get my block of test data
fpath <- paste0(dpath, '/wmSelection_BehaviouralData_All_Preprocessed.csv')
#fpath <- paste0(wd, '/EEG/data/s01/behaviour/wmSelection_EEG_S01_allData.csv') #look at Alex's EEG session behavioural data


df <- read.csv(fpath, header = T, as.is = T, sep = ',') %>% select(-X) # str(df) if you want to see the columns and values etc
nsubs = length(unique(df$subid))

#just quickly look at how many trials aren't clickresped/DTchecked
df %>%
  select(subid, clickresp, DTcheck, cond) %>%
  group_by(subid, cond) %>%
  dplyr::mutate(clickresp = (clickresp -1)*-1) %>%
  dplyr::summarise_at(.vars = c('clickresp', 'DTcheck'), .funs = c('sum')) %>%
  as.data.frame() %>%
  dplyr::mutate(clickresp = round((clickresp/128)*100,1))
#clickresp -- percentage of trials in condition where they didn't click to respond
#DTcheck   -- number of trials outside 2.5 SDs of the within condition mean

# may need to replace some subjects on the basis of this, as some subjects are missing a lot of trials (like 30-40% get thrown out because of clickresp)


#this section won't change df in place, just do stuff to it without change
df %>%
  dplyr::filter(DTcheck == 0) %>% dplyr::filter(clickresp == 1) %>%
  ggplot(aes(x = DT, fill = cond)) + 
  geom_histogram(stat = 'bin', binwidth = .05) + #DT is measure in seconds, not milliseconds, so a .1 bin width is 100ms bins
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'Decision Time (s)') +
  facet_wrap(cond ~ subid, nrow = 2, ncol = nsubs)
ggsave(filename = paste0(figpath, '/DT_hist_2subs_pilot.pdf'),
       dpi = 600, height = 10, width = 10)

#density plot of decision times (time taken to press space, after the probe appears)
df %>%
  dplyr::filter(DTcheck ==0) %>% dplyr::filter(clickresp == 1) %>%
  ggplot(aes(x = DT, fill = cond)) +
  geom_density(alpha = .5, adjust = 2) + #adjust sets the kernel width of the gaussian smoothing
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'Decision Time (s)') +
  facet_wrap(~subid)
ggsave(filename = paste0(figpath, '/DT_density_2subs_pilot.pdf'),
       dpi = 600, height = 10, width = 10)


library(circular)
wrap <- function(x) (x+180)%%360 - 180 #function to wrap data between +/- 90 degrees. keeps sign of the response (-ve leftwards, +ve rightwards)
wrap90 <- function(x) (x+90)%%180 - 90

df %<>% dplyr::mutate(rdif = wrap90(resp-targori)) #bound response deviation between -90 and 90 because of 180 degree symmetry
df %>%
  dplyr::filter(DTcheck == 0) %>%
  dplyr::filter(clickresp == 1) %>%
  ggplot(aes(x = rdif, fill = cond)) +
  geom_histogram(stat = 'bin', binwidth = 3) +
  geom_vline(aes(xintercept = 0), linetype = 'dashed', color = '#000000') +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'response deviation (degrees)') +
  facet_wrap(cond ~ subid, nrow = 2, ncol = nsubs)

#or plot as a geom_density ..
df %>%
  dplyr::filter(clickresp == 1) %>%
  dplyr::filter(DTcheck == 0) %>% #only take trials where DT within 2 sd's of the mean (throw out outliers)
  ggplot(aes(x = rdif, fill = cond)) +
  geom_density(alpha = .4, adjust = 2) + 
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'response deviation (degrees)') +
  facet_wrap(cond~subid,  nrow = 2, ncol = nsubs)
ggsave(filename = paste0(figpath, '/rdif_density_2subs_pilot.eps'),
       device = 'eps', dpi = 600, width = 10, height = 10)
ggsave(filename = paste0(figpath, '/rdif_density_2subs_pilot.pdf'),
       dpi = 600, height = 10, width = 10)


#look at mean abs dev & accuracy (1/SD) measures too

df.mad <- df %>%
  dplyr::filter(clickresp == 1) %>%
  dplyr::mutate(absrdif = abs(rdif)) %>%
  dplyr::group_by(subid, cond) %>%
  summarise_at(.vars = c('absrdif'), .funs = c('mean')) %>% as.data.frame()

#now plot mean absolute deviation in subjects (subplot for each subject)
df.mad %>%
  ggplot(aes(x = cond, y = absrdif, fill = cond)) +
  geom_bar(stat = 'identity', width = .7) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'cue condition', y = 'mean absolute deviation') +
  facet_wrap(~subid)
ggsave(filename = paste0(figpath, '/MeanAbsDev_2subs_pilot.pdf'),
       device = 'pdf', dpi = 600,width = 10, height = 10)


df.acc <- df %>%
  dplyr::filter(clickresp == 1) %>%
  dplyr::mutate(rabsrdif = rad(abs(rdif))) %>%
  dplyr::group_by(subid, cond) %>%
  summarise_at(.vars = c('rabsrdif'), .funs = c('sd')) %>% as.data.frame() %>%
  dplyr::mutate(acc = 1/rabsrdif) %>% #get 1/standard deviation of responses for condition
  dplyr::select(-rabsrdif) #get rid of the standard deviation

#plot accuracy (1/SD)
df.acc %>%
  ggplot(aes(x = cond, y = acc, fill = cond)) +
  geom_bar(stat = 'identity', width = .7) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'cue condition', y = 'accuracy (1/SD)') +
  facet_wrap(~subid) +
  theme(legend.position = 'none')
ggsave(filename = paste0(figpath, '/accuracy_2subs_pilot.pdf'),
       device = 'pdf', dpi = 600,width = 10, height = 10)


#look at confidence data now

df %<>% dplyr::mutate(confwidth = abs(wrap90(confang-resp)))

df %>%
  dplyr::filter(clickresp == 1) %>%
  filter(DTcheck == 0) %>%
  ggplot(aes(x = confwidth, fill = cond)) +
  geom_density(alpha = .5, adjust = 1) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'confidence report width') +
  facet_wrap(~subid)#cond)

df %>%
  dplyr::filter(clickresp == 1) %>%
  filter(DTcheck == 0) %>%
  ggplot(aes(x = confwidth, fill = cond)) +
  geom_histogram(stat = 'bin', binwidth = 3) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'confidence report width') +
  facet_wrap(cond ~ subid,  nrow = 2, ncol = nsubs)
ggsave(filename = paste0(figpath, '/confwidth_hist_2subs_pilot.pdf'),
       device = 'pdf', dpi = 600, height = 10, width = 10)


df.confvar <- df %>% dplyr::filter(clickresp == 1) %>%
  dplyr::filter(DTcheck == 0) %>%
  dplyr::group_by(subid, cond) %>%
  dplyr::mutate(rconfwidth = rad(confwidth)) %>%
  summarise_at(.vars = c('rconfwidth'), .funs = c('sd')) %>% as.data.frame() %>%
  dplyr::mutate(acc_confwidth = 1/rconfwidth) %>% select(-rconfwidth)
  
df.confvar %>%
  ggplot(aes(x = cond, y = acc_confwidth, fill = cond)) +
  geom_bar(stat = 'identity', width = .7) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'cue condition', y = '1/SD of confidence width') +
  facet_wrap(~subid)
  
# - - - --  - - - -

df %>%
  dplyr::filter(clickresp == 1) %>%
  dplyr::filter(DTcheck == 0) %>%
  dplyr::mutate(absrdif = abs(rdif)) %>%
  ggplot(aes(x = absrdif, y = confwidth, colour = cond)) +
  geom_point(size = 1) +
  geom_smooth(method = 'lm') +
  scale_color_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  geom_abline(intercept = 0, slope = 1, linetype = 'dashed', color = '#636363') +
  facet_wrap(cond ~ subid,  nrow = 2, ncol = nsubs)
ggsave(filename = paste0(figpath, '/absrdif~confwidth_2subs_pilot.pdf'),
       device = 'pdf', dpi = 600, height = 10, width = 10)

df %>% 
  dplyr::filter(clickresp == 1) %>%
  dplyr::filter(DTcheck == 0) %>%
  dplyr::mutate(absrdif = abs(rdif)) %>%
  dplyr::mutate(confdiff = absrdif - confwidth) %>%
  ggplot(aes(x = confdiff, fill = cond)) +
  geom_density(alpha = .5, adjust = 2) +
  geom_vline(aes(xintercept = 0), linetype = 'dashed', color = '#000000') +
  facet_grid(subid ~ cond)

df %>% 
  dplyr::filter(clickresp == 1) %>%
  dplyr::filter(DTcheck == 0) %>%
  dplyr::mutate(absrdif = abs(rdif)) %>%
  dplyr::mutate(confdiff = absrdif - confwidth) %>%
  ggplot(aes(x = confdiff, fill = cond)) +
  geom_histogram(stat = 'bin', binwidth = 2) +
  scale_fill_manual(values = c('neutral'= neutcol, 'cued' = cuedcol)) +
  geom_vline(aes(xintercept = 0), linetype = 'dashed', color = '#000000') +
  facet_wrap(cond ~ subid, nrow = 2, ncol = nsubs)
ggsave(filename = paste0(figpath, '/confprecision_2subs_pilot.pdf'),
       device = 'pdf', dpi = 600, width = 10, height = 10)
