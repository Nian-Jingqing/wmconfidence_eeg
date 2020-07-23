library(tidyverse)
library(magrittr)
library(lubridate)

#get demographics

wd <- "/Users/sammi/Desktop/Experiments/DPhil/wmConfidence"
setwd(wd)

infodir <- paste0(wd, '/data/info')
fnames  <- sort(list.files(path = infodir, pattern = '_info.csv'))
files  <- list(NULL)
count <- 1
for(i in fnames){
  path <- paste0(infodir, '/', i)
  split = str_split(path, 'wmConfidence', simplify = T)[3]
  
  
  session <- ifelse(str_detect(split, 'b'), 2, 1)
  tmp  <- read.csv(path, header = F, as.is = T, sep = ',') %>% mutate(session = session)
  files[[count]] <- tmp
  count <- count+1
}

dat <- do.call('rbind', files)

dat %<>% dplyr::rename('subid' = V1,
                'age'   = V2,
                'sex'   = V3,
                'handedness' = V4,
                'date'  = V5) %>%
  dplyr::select(-date) %>%
  dplyr::mutate(sex = as.factor(sex), handedness = str_to_upper(handedness)) %>%
  dplyr::mutate(sex = str_to_upper(sex)) %>% dplyr::mutate(handedness = as.factor(handedness)) %>%
  dplyr::mutate(sex = ifelse(sex == 'FALSE', 'F', sex)) %>% dplyr::mutate(sex = factor(sex))


dat %>% filter(session == 1) %>% summary()
