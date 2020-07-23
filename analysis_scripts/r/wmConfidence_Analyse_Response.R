#import libraries
library(tidyverse) # general data manipulation & plotting
library(magrittr)  # allows use of more pipes
library(afex)      # anovas etc

loadfonts = T
if(loadfonts){
  library(extrafont)
  font_import() #yes to this
  font_import(paths='/home/sammirc/Desktop/fonts')
  fonts()
  loadfonts(device = "pdf");
  loadfonts(device = 'postscript')
}

#set theme for plots to standardise them and make cleaner
theme_set(theme_bw() +
            theme(
              strip.background = element_blank(),
              axis.text    = element_text(family = 'Source Sans Pro', colour = '#000000', size = 18),
              axis.title   = element_text(family = 'Source Sans Pro', colour = '#000000', size = 24),
              panel.border = element_rect(size = 1, color = '#000000'),
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
#wd <- '/home/sammirc/Desktop/DPhil/wmConfidence'
setwd(wd)

dpath <- paste0(wd, '/data') #path to folder with behavioural data
figpath <- paste0(wd, '/figures/behaviour')
fpath <- paste0(dpath, '/datafiles/wmConfidence_BehaviouralData_All.csv')

#get my block of test data
df <- read.csv(fpath, header = T, as.is = T, sep = ',') # str(df) if you want to see the columns and values etc
nsubs <- length(unique(df$subid))

#just quickly look at how many trials aren't clickresped/DTchecked
df %>%
  dplyr::select(subid, clickresp, DTcheck, cond) %>%
  group_by(subid, cond) %>%
  dplyr::mutate(clickresp = (clickresp -1)*-1) %>%
  dplyr::summarise_at(.vars = c('clickresp', 'DTcheck'), .funs = c('sum')) %>%
  as.data.frame() %>%
  dplyr::mutate(clickresp = ifelse(subid %in% c(1,2), (clickresp/320)*100, clickresp)) %>%
  dplyr::mutate(clickresp = ifelse(subid %in% c(3,10, 19), (clickresp/256)*100, clickresp)) %>%
  dplyr::mutate(clickresp = ifelse(subid >3 & subid != 10 & subid != 19, (clickresp/512)*100, clickresp)) %>%
  dplyr::mutate(clickresp = round(clickresp,1) )
# clickresp -- percentage of trials in condition where they didn't click to respond
# DTcheck   -- number of trials outside 2.5 SDs of the within condition mean
# may need to replace some subjects on the basis of this, as some subjects are missing a lot of trials (like 30-40% get thrown out because of clickresp)

#note:
#subjects 1 & 2 have 320 trials (1 session of 10 blocks)
#subject 3 has one session of 8 blocks (256 trials)
#subjects 4 onwards have 2 sessions of 8 blocks (512 trials in total)

subs2use <- c(4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 24, 25, 26) #these subs are included in eeg analyses, so just work with these for behavioural analyses too

df  %<>% # shape = 12128, 45 at this point
  dplyr::group_by(subid, session) %>%
  dplyr::mutate(prevtrlconfdiff = lag(confdiff)) %>%
  as.data.frame(.)

df %<>% dplyr::filter(subid %in% subs2use) %>% #keep only subjects safe to use based on clicking to confirm response (this is a failsafe in case it's not coded later on)
  dplyr::filter(clickresp == 1) %>%
  dplyr::filter(DTcheck == 0) #hard code this now

dim(df) #9579, 45 at this point


#quickly look to see if there might be any proof of behavioural change following poor error awareness on the previous trial

#this section won't change df in place, just do stuff to it without change
df %>%
  ggplot(aes(x = DT, fill = cond)) +
  geom_histogram(aes(y = ..count../sum(..count..)),stat = 'bin', binwidth = .1) + #DT is measure in seconds, not milliseconds, so a .1 bin width is 100ms bins (this line does normalised bins rel to trial number)
  #geom_histogram(stat = 'bin', binwidth = .2) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'Decision Time (s)') +
  facet_wrap(subid~cond, ncol = 6, scales= 'free')
ggsave(filename = paste0(figpath, '/DT_hist.pdf'),
       dpi = 600, height = 10, width = 10)

#density plot of decision times (time taken to press space, after the probe appears)
df %>%
  ggplot(aes(x = DT, fill = cond)) +
  geom_density(alpha = .5, adjust = 2) + #adjust sets the kernel width of the gaussian smoothing
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'Decision Time (s)') +
  facet_wrap(~subid, scales = 'free')
ggsave(filename = paste0(figpath, '/DT_density_20subs.pdf'), device = cairo_pdf, dpi = 600, height = 15, width = 15)
ggsave(filename = paste0(figpath, '/DT_density_20subs.eps'), device = cairo_ps, dpi = 600, height = 15, width = 15)

df.dt <- df %>%
  dplyr::group_by(subid, cond) %>%
  dplyr::summarise_at(.vars = 'DT', .funs = 'mean') #gets within subject, within condition averages

df.dt.diffs <- df.dt %>% 
  dplyr::group_by(subid) %>%
  dplyr::summarise(diff = last(DT)- first(DT)) #neutral - cued decision time (positive means lower for cued)

df.dt.diffs %>%
  dplyr::summarise_at(.vars = 'diff', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = 1, y = mean, ymin = mean-se, ymax = mean + se)) +
  geom_bar(stat = 'identity', width = .4, fill = diffcol) + 
  geom_errorbar(stat = 'identity', width = .15, size = 1) +
  geom_point(data = df.dt.diffs, aes(x = 1, y = diff), inherit.aes = F,size = 1, position = position_jitter(width=.1,height=0)) + 
  geom_hline(yintercept = 0, linetype = 'dashed', color = '#000000', size = .2) +
  labs(x = '', y = 'Difference in DT between\nneutral and cued condition (s)') +
  xlim(c(0.5,1.5)) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
ggsave(filename = paste0(figpath, '/DT_groupaverage_diffs_20subs.eps'), device = cairo_ps , dpi = 600, height = 9, width = 9)
ggsave(filename = paste0(figpath, '/DT_groupaverage_diffs_20subs.pdf'), device = cairo_pdf, dpi = 600, height = 9, width = 9)


df.dt %>% #not changing in place (will use dframe for stats, mutating in place for plotting)
  dplyr::group_by(cond) %>%
  dplyr::summarise_at(.vars = 'DT', .funs = c('mean', 'se')) %>% #gets group average for conditions
  ggplot(aes(x = cond, y = mean, ymin = mean-se, ymax = mean+se, fill = cond)) +
  geom_bar(stat = 'identity', width = .7) +
  geom_errorbar(stat = 'identity', width = .2, size = 1) +
  #geom_point(inherit.aes=F, data = df.dt, aes(x = cond, y = DT), size = 1) +
  #geom_line(inherit.aes=F, data = df.dt, aes(x = cond,y = DT, group = subid), size = .5) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = '', y = 'mean decision time (s)') +
  theme(legend.position = 'none')
ggsave(filename = paste0(figpath, '/DT_groupaverage_20subs.eps'), device = cairo_ps, dpi = 600, height = 9, width = 9)
ggsave(filename = paste0(figpath, '/DT_groupaverage_20subs.pdf'), device = cairo_pdf, dpi = 600, height = 9, width = 9)




library(circular)
wrap <- function(x) (x+180)%%360 - 180 #function to wrap data between +/- 90 degrees. keeps sign of the response (-ve leftwards, +ve rightwards)
wrap90 <- function(x) (x+90)%%180 - 90

#check for uniformity (if u wanna)
test <- df %>%
  dplyr::group_by(subid,cond) %>%
  dplyr::select(rdif) %>%
  group_map(~ rayleigh.test(.$rdif, mu = circular(0)))

pvals     = vector(length = length(test), mode = 'numeric')
teststats = vector(length = length(test), mode = 'numeric')
for(i in seq(1,length(test),1)){
  pvals[i] = test[[i]]$p.value
  teststats[i] = test[[i]]$statistic
}

rtest <- df %>% 
  dplyr::group_by(subid, cond) %>%
  dplyr::select(rdif) %>%
  summarise(mean(abs(rdif))) %>% ungroup() %>% as.data.frame() %>%
  dplyr::mutate(rayleigh_p = pvals)


df %>%
  ggplot(aes(x = rdif, fill = cond)) +
  geom_histogram(stat = 'bin', binwidth = 3) +
  geom_vline(aes(xintercept = 0), linetype = 'dashed', color = '#000000') +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'response deviation (degrees)') +
  facet_wrap(cond ~ subid, nrow = 2, ncol = length(subs2use))

#or plot as a geom_density ..
df %>%
  dplyr::filter(subid %in% subs2use) %>%
  ggplot(aes(x = rdif, color = cond, fill=NULL)) +
  geom_density(alpha = .4, adjust = 1, size=1.5) +
  geom_vline(xintercept = 0, linetype = 'dashed', color = '#000000', size = .2) +
  scale_color_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'response deviation (degrees)') +
  facet_wrap(~subid, ncol = 4)
ggsave(filename = paste0(figpath, '/rdif_density.eps'), device = cairo_ps, dpi = 600, height = 15, width = 15)
ggsave(filename = paste0(figpath, '/rdif_density.pdf'), device = cairo_pdf, dpi = 600, height = 15, width = 15)


 #look at mean abs dev & accuracy (1/SD) measures too

df.mad <- df %>%
  dplyr::filter(subid %in% subs2use) %>%
  dplyr::filter(clickresp == 1) %>%
  dplyr::mutate(absrdif = abs(rdif)) %>%
  dplyr::group_by(subid, cond) %>%
  summarise_at(.vars = c('absrdif'), .funs = c('mean')) %>% as.data.frame()

df.mad.diffs <- df.mad %>%
  dplyr::group_by(subid) %>%
  dplyr::summarise(diff = last(absrdif) - first(absrdif)) #neutral - cued (negative values where cued MAD worse than neutral MAD)

df.mad.diffs %>%
  dplyr::summarise_at(.vars = 'diff', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = 1, y = mean, ymin = mean-se, ymax = mean + se)) +
  geom_bar(stat = 'identity', width = .4, fill = diffcol) + 
  geom_errorbar(stat = 'identity', width = .1, size = 1) +
  geom_point(data = df.mad.diffs, aes(x = 1, y = diff), inherit.aes = F,size = 1, position = position_jitter(width=.1,height=0)) + 
  geom_hline(yintercept = 0, linetype = 'dashed', color = '#000000', size = .5) +
  labs(x = '', y = 'Difference in MAD between\nneutral and cued condition (degrees)') +
  xlim(c(0.5,1.5)) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
ggsave(filename = paste0(figpath, '/MAD_groupaverage_diffs_20subs.eps'), device = cairo_ps , dpi = 600, height = 9, width = 9)
ggsave(filename = paste0(figpath, '/MAD_groupaverage_diffs_20subs.pdf'), device = cairo_pdf, dpi = 600, height = 9, width = 9)


#now plot mean absolute deviation in subjects (subplot for each subject)
df.mad %>%
  ggplot(aes(x = cond, y = absrdif, fill = cond)) +
  geom_bar(stat = 'identity', width = .7) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'cue condition', y = 'mean absolute deviation') +
  facet_wrap(~subid)
ggsave(filename = paste0(figpath, '/MeanAbsDev_20subs.pdf'), device = cairo_pdf, dpi = 600,width = 10, height = 10)
ggsave(filename = paste0(figpath, '/MeanAbsDev_20subs.eps'), device = cairo_ps , dpi = 600,width = 10, height = 10)

df.mad %>%
  dplyr::group_by(cond) %>%
  dplyr::summarise_at(.vars = c('absrdif'), .funs = c('mean', 'se')) %>%
  ggplot(aes(x = cond, y = mean, ymin = mean-se, ymax = mean+se, fill = cond)) +
  geom_bar(stat = 'identity', width = .7) +
  geom_errorbar(stat = 'identity', position = position_dodge(), width = 0.35, size = 1) +
  #geom_point(inherit.aes=F, data = df.mad, aes(x = cond, y = absrdif), size = 1) +
  #geom_line(inherit.aes=F, data = df.mad, aes(x = cond,y = absrdif, group = subid), size = .5) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = '', y = 'error (degrees)') +
  theme(legend.position = 'none')
ggsave(filename = paste0(figpath, '/MAD_groupaverage_20subs.eps'), device = cairo_ps, dpi = 600, height = 9, width = 9)
ggsave(filename = paste0(figpath, '/MAD_groupaverage_20subs.pdf'), device = cairo_pdf, dpi = 600, height = 9, width =9)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

df.acc <- df %>% 
  dplyr::filter(subid %in% subs2use) %>%
  dplyr::filter(clickresp == 1) %>%
  dplyr::mutate(rabsrdif = rad(abs(rdif))) %>%
  dplyr::group_by(subid, cond) %>%
  summarise_at(.vars = c('rabsrdif'), .funs = c(sd.circular)) %>% as.data.frame() %>%
  dplyr::mutate(acc = rabsrdif) %>% #get standard deviation of responses for condition
  dplyr::select(-rabsrdif) #get rid of the standard deviation

#plot response variability (SD of error)
df.acc %>%
  ggplot(aes(x = cond, y = acc, fill = cond)) +
  geom_bar(stat = 'identity', width = .7) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = 'cue condition', y = 'error variability (SD)') +
  facet_wrap(~subid) +
  theme(legend.position = 'none')
ggsave(filename = paste0(figpath, '/errorvar_20subs.pdf'), device = cairo_pdf, dpi = 600,width = 15, height = 15)

#plot as a single difference value rather than separate bars
df.acc.diffs <- df.acc %>%
  dplyr::group_by(subid) %>%
  dplyr::summarise(diff = last(acc) - first(acc)) #neutral - cued (negative values where cued MAD worse than neutral MAD)

df.acc.diffs %>%
  dplyr::summarise_at(.vars = 'diff', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = 1, y = mean, ymin = mean-se, ymax = mean + se)) +
  geom_bar(stat = 'identity', width = .4, fill = diffcol) + 
  geom_errorbar(stat = 'identity', width = .1, size = 1) +
  geom_point(data = df.acc.diffs, aes(x = 1, y = diff), inherit.aes = F,size = 1, position = position_jitter(width=.051,height=0)) + 
  geom_hline(yintercept = 0, linetype = 'dashed', color = '#000000', size = .5) +
  labs(x = '', y = 'difference in error variability (SD)\nbetween neutral and cued condition ') +
  xlim(c(0.5,1.5)) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())
ggsave(filename = paste0(figpath, '/errorvar_groupaverage_20subs.eps'), device = cairo_ps, dpi = 600, height = 9, width = 9)
ggsave(filename = paste0(figpath, '/errorvar_groupaverage_20subs.pdf'), device = cairo_pdf, dpi = 600, height = 9, width = 9)



#plot group average
df.acc %>%
  dplyr::group_by(cond) %>%
  dplyr::summarise_at(.vars = 'acc', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = cond, y = mean, ymin = mean-se, ymax = mean+se, fill = cond)) +
  geom_bar(stat = 'identity', width = .7) +
  geom_errorbar(stat = 'identity', width = .35, size = 1) +
  #geom_point(inherit.aes=F, data = df.acc, aes(x = cond, y = acc), size = 1) +
  #geom_line(inherit.aes=F, data = df.acc, aes(x = cond,y = acc, group = subid), size = .5) +
  scale_fill_manual(values = c('neutral' = neutcol, 'cued' = cuedcol)) +
  labs(x = '', y = 'error variability (SD)') +
  theme(legend.position = 'none')
ggsave(filename = paste0(figpath, '/errorvar_groupaverage_20subs.eps'), device = cairo_ps, dpi = 600, height = 8, width = 8)
ggsave(filename = paste0(figpath, '/errorvar_groupaverage_20subs.pdf'), device = cairo_pdf, dpi = 600, height = 8, width = 8)


#one-way anova on decision time
anova_dt  <- afex::aov_ez(id = 'subid', data = df.dt , 'DT' , within = 'cond')
nice(anova_dt, es = 'pes') #significant main effect of condition (cue) on decision time
t.test(DT~cond, data = df.dt, paired = T) #mean diff output here


anova_mad <- afex::aov_ez(id = 'subid', data = df.mad, 'absrdif', within = 'cond')
nice(anova_mad, es = 'pes') #signif main effect of condition (cue) on mean absolute deviation
t.test(absrdif ~ cond, df.mad, paired = T)

#one-way anova on accuracy(1/sd measure)
anova_acc <- afex::aov_ez(id = 'subid', data = df.acc, 'acc', within = 'cond')
nice(anova_acc, es = 'pes') 
t.test(acc ~ cond, df.acc, paired = T)


#are people better at oblique angles?

df_oblique <- df %>%
  dplyr::mutate(dist_to_45 = targori - 45) %>%
  dplyr::mutate(dist_to_90 = targori - 90) %>%
  dplyr::mutate(dist_to_135 = targori - 135) %>%
  dplyr::select(cond, pside, subid, ori1, ori2, targori, nontargori, pstartang, DT, resp, rdif, confwidth, absrdif, confdiff, dist_to_45, dist_to_90, dist_to_135)

df_oblique %<>%
  rowwise() %>%
  dplyr::mutate(dist_oblique = min(abs(c(dist_to_45, dist_to_135)))) %>% as.data.frame()

df_oblique %>%
  ggplot(aes(x = dist_oblique, y = absrdif)) +
  geom_point(size = .5) +
  geom_smooth(method = 'gam', formula = y ~s(x, bs='cr')) +
  facet_wrap(~subid)

#doesn't really look like any oblique effects here across subjects to be honest

