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

dim(df) #9488 x 47 ## actualy 9517 x 47 ??

tmp <- df
# df <- tmp




#post-error slowing?
df %<>% dplyr::mutate(prevtrlcorr = ifelse(prevtrlconfdiff <= 0, 'correct', 'incorrect')) %>%
  dplyr::filter(!is.na(prevtrlconfdiff)) %>%
  dplyr::mutate(prevtrlcorr = as.factor(prevtrlcorr)) 

df %>%
  dplyr::group_by(subid, prevtrlcorr) %>%
  dplyr::select(DT) %>%
  dplyr::summarise_at(.vars = 'DT', .funs = 'mean') %>% dplyr::ungroup() %>%
  dplyr::group_by(prevtrlcorr) %>%
  dplyr::summarise_at(.vars = 'DT', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = prevtrlcorr, y = mean, fill = prevtrlcorr)) +
  geom_bar(stat = 'identity', aes(fill = prevtrlcorr), width = .7, position = position_dodge(.3)) +
  geom_errorbar(inherit.aes = F, aes(x = prevtrlcorr, ymin = mean-se, ymax = mean+se), position = position_dodge(.3), width = .3, size=1) +
  labs(x = 'previous trial correctness',
       y = 'average decision time (s)\nfor current trial') + theme(legend.position = 'none')

#bar chart not overwhelming, but its about binary correctness and not necessarily error
#check as geom_point to view all the data

df %>%
  dplyr::mutate(currTrlCorr = ifelse(confdiff<=0, 'correct', 'incorrect')) %>% dplyr::mutate(currTrlCorr = as.factor(currTrlCorr)) %>%
  ggplot(aes(x = prevtrlabsrdif, y = log(DT))) +#, color = currTrlCorr)) +
  geom_point(alpha=.5, size=.5) +
  geom_smooth(formula = y ~ x, method = 'lm', alpha = .5, se = T, size=1) +
  facet_wrap(~subid) + labs(x = 'previous trial error (degrees)', y = 'log Decision Time (s)')


#can look at other things changing after an incorrect trial too though (confidence width or error?)

df %>%
  dplyr::group_by(subid, prevtrlcorr) %>%
  dplyr::select(absrdif) %>%
  dplyr::summarise_at(.vars = 'absrdif', .funs = 'mean') %>% dplyr::ungroup() %>% #ungroup after within subject averaging
  dplyr::group_by(prevtrlcorr) %>%
  dplyr::summarise_at(.vars = 'absrdif', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = prevtrlcorr, y = mean, fill = prevtrlcorr)) +
  geom_bar(stat = 'identity', aes(fill=prevtrlcorr), width = .7, position = position_dodge(.3)) +
  geom_errorbar(inherit.aes=F, aes(x = prevtrlcorr, ymin = mean-se, ymax = mean+se), position = position_dodge(.3), size = 1, width = .3) +
  labs(x = 'previous trial correctness', y = 'mean absolute response error (degrees)') +
  theme(legend.position = 'none')

df %>%
  dplyr::group_by(subid, prevtrlcorr) %>%
  dplyr::select(confwidth) %>%
  dplyr::summarise_at(.vars = 'confwidth', .funs = 'mean') %>% dplyr::ungroup() %>% #ungroup after within subject averaging
  dplyr::group_by(prevtrlcorr) %>%
  dplyr::summarise_at(.vars = 'confwidth', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = prevtrlcorr, y = mean, fill = prevtrlcorr)) +
  geom_bar(stat = 'identity', aes(fill=prevtrlcorr), width = .7, position = position_dodge(.3)) +
  geom_errorbar(inherit.aes=F, aes(x = prevtrlcorr, ymin = mean-se, ymax = mean+se), position = position_dodge(.3), size = 1, width = .3) +
  labs(x = 'previous trial correctness', y = 'mean confidence width (degrees)') +
  theme(legend.position = 'none')

#can look at these two in non-binary fashions too, is the degree of error important in the change?
df %>%
  dplyr::mutate(currTrlCorr = ifelse(confdiff<=0, 'correct', 'incorrect')) %>%
  dplyr::mutate(currTrlCorr = as.factor(currTrlCorr)) %>%
  ggplot(aes(x = prevtrlabsrdif, y = absrdif)) +
  geom_point(alpha=.5, size=.5) +
  geom_smooth(formula = y ~ x, method = 'lm', alpha = .5, se = T, size=1) +
  facet_wrap(~subid)


df %>%
  dplyr::mutate(currTrlCorr = ifelse(confdiff <= 0, 'correct', 'incorrect')) %>%
  dplyr::mutate(currTrlCorr = as.factor(currTrlCorr)) %>%
  ggplot(aes(x = prevtrlabsrdif, y = confwidth)) +
  geom_point(alpha = .5, size = .5) +
  geom_smooth(formula = y ~ x, method = 'lm', alpha = .5, size = 1, se = T) +
  facet_wrap(~subid)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

#look at post error changes in behaviour: starting with response error on trial N+1 after response on trial N
# i.e. if you're bad on trial N, are you subsequently better on trial N+1 (post error change in behaviour)

lmm.posterr.error <- df %>%
  dplyr::filter(trialnum != 1) #first trial doesn't have a preceding trial

lmm.posterr.absrdif <- lme4::lmer(data = lmm.posterr.error,
                                  absrdif ~ prevtrlabsrdif + (1 + prevtrlabsrdif| subid))
summary(lmm.posterr.absrdif)

lmm.posterr.conf <- df %>% dplyr::filter(trialnum != 1)
lmm.posterr.confwidth <- lme4::lmer(data = lmm.posterr.conf,
                                    confwidth ~ prevtrlabsrdif + (1 + prevtrlabsrdif | subid))
summary(lmm.posterr.confwidth)


lmm.trladjcw.data <- df %>%
  dplyr::mutate(prevtrlinconf = ifelse(prevtrlconfdiff <= 0, 1, 0)) %>%
  dplyr::mutate(prevtrlinconf = as.factor(prevtrlinconf)) %>%
  dplyr::mutate(trladj = confwidth - prevtrlcw) %>% # difference in current trials confidence compared to the previous trial
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
summary(lmm.trladjcw.full)

library(remef)

fit.trladjcw <- keepef(lmm.trladjcw.full, fix = c('prevtrlconfdiff', 'prevtrlcw'), grouping=T)
lmm.trladjcw.data$fitted <- fit.trladjcw




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#trialwise adjustments of confidence

lmm.trladjcw.data <- df %>%
  dplyr::mutate(prevtrlinconf = ifelse(prevtrlconfdiff <= 0, 1, 0)) %>%
  dplyr::mutate(prevtrlinconf = as.factor(prevtrlinconf)) %>%
  dplyr::mutate(trladj = confwidth - prevtrlcw) %>% # difference in current trials confidence compared to the previous trial
  dplyr::filter(trialnum != 1) %>% #exclude first trial of each session as no prev trial for it (these vals would be NA anyway)
  dplyr::filter(!is.na(prevtrlconfdiff))


#check correlation between confidence width and the update
lmm.trladjcw.data %>% ggplot(aes(x = prevtrlcw, y = trladj)) +
  geom_point(size=.3) +
  geom_smooth(method='lm') +
  facet_wrap(~subid) +
  labs(y = 'trialwise adjustment in confidence\n(negative = become more confident)',
       x = 'previous trial confidence width')

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
summary(lmm.trladjcw.full)

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
  theme_bw()
# ggsave(filename = paste0(figpath, '/trladjustment_confidence_prevtrlconferr_20subs_agg.pdf'), device = 'pdf', dpi = 600, height = 9, width = 9)
# ggsave(filename = paste0(figpath, '/trladjustment_confidence_prevtrlconferr_20subs_agg.eps'), device = 'eps', dpi = 600, height = 9, width = 9)

lmm.trladjcw.data %>%
  ggplot(aes(x = prevtrlconfdiff, y = trladj)) +
  geom_point(size = .5, color = '#bdbdbd') +
  geom_smooth(inherit.aes = F, aes(x = prevtrlconfdiff, y = fitted), method = 'lm', color = '#756bb1') +
  labs(y = 'confidence adjustment\n(current trial - previous trial confidence width',
       x = 'previous trial confidence error')#only add theme_bw if changing the device from cairo to normal to export into sketch


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


fpath <- paste0(dpath, '/datafiles/wmConfidence_BehaviouralData_All.csv')
dat <- read.csv(fpath, header = T, as.is = T, sep = ',') 
dat %>% group_by(subid) %>% count() %>% as.data.frame() # see how many trials per participant 
dat %<>% dplyr::filter(subid %in% subs2use) #remove some subjects who are rly bad or dont have full datasets



#one thing we also want to do for a specific analysis is to get the last 'relevant' confidence report
#i.e we want to know what the last confidence report for a cued trial was, and what the last neutral trial confidence was
#to see if people "multi-task" and treat cued/neutral trials as different tasks and so update their confidence (independently) for the two trial types
#so we will get the original data (without having removed any trials) and get this info

newdat <- list(NULL)
count <- 1
for(sub in subs2use){
  tmp <- dplyr::filter(dat, subid == sub) #get single subject data
  ntrials <- dim(tmp)[1]
  
  tmp %<>% dplyr::mutate(lastNeutralConf = NaN, lastCuedConf = NaN)
  for(trial in seq(1,ntrials-1,by=1)){
    curTrialConf <- tmp$confwidth[trial]
    curTrialType <- tmp$cond[trial]
    
    tmp$lastNeutralConf[trial+1] <- ifelse(curTrialType=='neutral', curTrialConf, tmp$lastNeutralConf[trial])
    tmp$lastCuedConf[trial+1]    <- ifelse(curTrialType=='cued',    curTrialConf, tmp$lastCuedConf[trial])
    }
  
  newdat[[count]] <- tmp
  count <- count + 1
  
}

dat <- do.call('rbind', newdat)
dat %<>% dplyr::mutate(lastRelevConf = NaN) %>%
  dplyr::mutate(lastRelevConf = ifelse(cond == 'cued', lastCuedConf, lastNeutralConf))


dat %<>% dplyr::mutate(trladjustRelev = ifelse(cond=='cued', confwidth - lastCuedConf, confwidth - lastNeutralConf)) %>%
  dplyr::mutate(trladjust = confwidth - prevtrlcw)


lmmdat <- dat %>%
          dplyr::filter(!is.na(trladjust)) %>% #cant have nans in the glm
          dplyr::filter(!is.na(lastRelevConf)) %>% #still cant have nans in the data
          dplyr::filter(clickresp == 1) %>%    #only take trials where they clicked to respond
          dplyr::filter(confclicked == 1) %>%  # only take trials where they confirmed their confidence judgement
          dplyr::filter(DTcheck == 0)    #and remove trials that are outliers based on primary response reaction time


for(sub in subs2use){
  glmdat <- dplyr::filter(lmmdat, subid == sub)
  
  model <- glm(glmdat, trladjust ~ confwidth + prevtrlcw)
}

lmmdat %<>% dplyr::group_by(subid) %>%
  dplyr::mutate(zLastRelevConf = scale(lastRelevConf) ) %>% ungroup()

lmmdat %<>% dplyr::mutate(confwidth = ifelse(confwidth ==0, 0.0001, confwidth))

lambdaList <- boxcox(confwidth~1, data=lmmdat)
(lambda <- lambdaList$x[which.max(lambdaList$y)])

model <- lme4::lmer(data = lmmdat, log(confwidth) ~ prevtrlcw + lastRelevConf + trialnum + (1 + prevtrlcw + lastRelevConf + trialnum |subid))
summary(model)


lmmdat %>%
  ggplot(aes(x = lastRelevConf, y = confwidth)) +
  geom_point(size = .5) + geom_smooth(method = 'lm', size = .7, se = T, alpha = .5) +
  facet_wrap(~subid)

lmmdat %>%
  ggplot(aes(x = prevtrlcw, y = confwidth)) +
  geom_point(size = .5) + geom_smooth(method = 'lm', size = .7, se = T, alpha = .5) +
  facet_wrap(~subid)

lmmdat %>%
  ggplot(aes(x = prevtrlcw, y = lastRelevConf)) +
  geom_point(size=.5) + geom_smooth(method='lm', size = .7, alpha = .5, se = T) +
  facet_wrap(~subid)


# - - - - - - - - - - - - - - - - - - - - - - 
fullmodel_log <- lme4::lmer(data = lmm.data, log(confwidth) ~ absrdif + cond + absrdif:cond +
                              (1 + absrdif + cond + absrdif:cond|subid))
summary(rePCA(fullmodel_log))#check if model is degenerate
minmodel_log  <- lme4::lmer(data = lmm.data, log(confwidth) ~ absrdif + cond + absrdif:cond  + (1|subid))
summary(rePCA(minmodel_log))
anova(fullmodel_log, minmodel_log)
