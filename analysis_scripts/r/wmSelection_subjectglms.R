#import libraries
library(tidyverse) # general data manipulation & plotting
library(magrittr)  # allows use of more pipes
library(afex)      # anovas etc
library(RePsychLing)
library(MASS)
library(broom)

loadfonts = F
if(loadfonts){
  library(extrafont)
  font_import() #yes to this
  loadfonts(device = "pdf");
  loadfonts(device = 'postscript')
}

library(circular)
wrap <- function(x) (x+180)%%360 - 180 #function to wrap data between +/- 90 degrees. keeps sign of the response (-ve leftwards, +ve rightwards)
wrap90 <- function(x) (x+90)%%180 - 90




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

wd <- '/Users/sammi/Desktop/Experiments/DPhil/wmSelection'
setwd(wd)

dpath <- paste0(wd, '/data') #path to folder with behavioural data
figpath <- paste0(wd, '/figures')

#get my block of test data
fpath <- paste0(dpath, '/wmSelection_BehaviouralData_All_Preprocessed.csv')
#fpath <- paste0(wd, '/EEG/data/s01/behaviour/wmSelection_EEG_S01_allData.csv') #look at Alex's EEG session behavioural data


df <- read.csv(fpath, header = T, as.is = T, sep = ',') %>% dplyr::select(-X) # str(df) if you want to see the columns and values etc
nsubs = length(unique(df$subid))
subs2use <- c(1,2,3,4,5,6,7,9,10,11,13,14,15,16,17,18,19,20,21,22)

df %<>% dplyr::filter(subid %in% subs2use) %>% #keep only subjects safe to use based on clicking to confirm response (this is a failsafe in case it's not coded later on)
  dplyr::filter(clickresp == 1) %>% dplyr::filter(confclicked == 1) %>% dplyr::filter(DTcheck == 0) #hard code this now
#trials excluded if didn't click to confirm response or confidence judgement, and DT outside a couple sd's of mean per subject and condition

dim(df) #should be 4531 trials (across all ppts)

df$confwidth[df$confwidth==0] <- 0.0001
lambdaList <- boxcox(confwidth~cond, data=df)
(lambda <- lambdaList$x[which.max(lambdaList$y)])

subsummaries <- data.frame('term'='0', 'estimate'=0, 'std.error'=0, 'statistic'=0, 'p.value'=0, 'subid'=0)

for(sub in subs2use){
  
  tmp <- dplyr::filter(df, subid == sub)
  blockwise_fb <- tmp %>%
    dplyr::group_by(block) %>%
    dplyr::summarise_at(.vars = 'confdiff', .funs = 'mean') %>%
    dplyr::mutate(block = as.integer(block+1)) %>%
    dplyr::mutate(confdiff = round(confdiff, digits = 3 )) %>%
    dplyr::rename(prevblock_confdiff = confdiff) %>% dplyr::filter(block %in% seq(2,8))
  
  tmp %<>% dplyr::right_join(blockwise_fb, by = 'block')
  
  subglm <- glm(data = tmp, formula = log(confwidth) ~ log(trialnum) + prevblock_confdiff + absrdif)
  
  tidytable <- tidy(subglm) %>% dplyr::mutate(subject = sub)
  subsummaries %<>% dplyr::bind_rows(tidytable)
}

subsummaries %>%
  dplyr::filter(term != '0' & term != '(Intercept)') %>% dplyr::mutate(term = as.factor(term)) %>%
  dplyr::group_by(term) %>%
  dplyr::summarise_at(.vars = 'estimate', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = term, y = mean, ymin = mean-se, ymax = mean+se)) +
  geom_bar(stat = 'identity', width = .4) +
  geom_errorbar(width = .2) +
  geom_hline(yintercept = 0, linetype = 'dashed', size = .75)

  

#look at confdiff

lambdaList <- boxcox(confdiff~cond, data=df)
(lambda <- lambdaList$x[which.max(lambdaList$y)])



subsummaries2 <- data.frame('term'='0', 'estimate'=0, 'std.error'=0, 'statistic'=0, 'p.value'=0, 'subid'=0)

for(sub in subs2use){
  
  tmp <- dplyr::filter(df, subid == sub)
  blockwise_fb <- tmp %>%
    dplyr::group_by(block) %>%
    dplyr::summarise_at(.vars = 'confdiff', .funs = 'mean') %>%
    dplyr::mutate(block = as.integer(block+1)) %>%
    dplyr::mutate(confdiff = rad(round(confdiff, digits = 3 ))) %>% 
    dplyr::rename(prevblock_confdiff = confdiff) %>% dplyr::filter(block %in% seq(2,8))

  
  tmp %<>% dplyr::right_join(blockwise_fb, by = 'block') %>%
    dplyr::mutate(nontargdev = rad(wrap90(nontargori - targori))) %>%
    dplyr::mutate(confdiff = rad(confdiff)) %>%
    dplyr::mutate(cond = as.factor(cond)) %>% dplyr::mutate(cond = relevel(cond, ref = 'neutral'))
  
  
  subglm <- glm(data = tmp, formula = confdiff ~ log(trialnum) + prevblock_confdiff + absrdif + nontargdev + cond)
  
  tidytable <- tidy(subglm) %>% dplyr::mutate(subject = sub)
  subsummaries2 %<>% dplyr::bind_rows(tidytable)
}

subsummaries2 %>%
  dplyr::filter(term != '0' & term != '(Intercept)') %>% dplyr::mutate(term = as.factor(term)) %>%
  dplyr::group_by(term) %>%
  dplyr::summarise_at(.vars = 'estimate', .funs = c('mean', 'se')) %>%
  ggplot(aes(x = term, y = mean, ymin = mean-se, ymax = mean+se)) +
  geom_bar(stat = 'identity', width = .4, aes(fill = term)) +
  geom_errorbar(width = .2) +
  geom_hline(yintercept = 0, linetype = 'dashed', size = .75)


