library(dplyr)
getwd()
dir <- '/home/sammirc/Desktop/DPhil/wmConfidence'
dir <- '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'
setwd(dir)

datapath <- paste0(dir, '/data/datafiles')

# only doing this for specific subjects really, for now at least
sublist <- seq(1, 2, by = 1)
dataFiles = list(NULL)

for(sub in sublist){
  subpath <- paste0(datapath, sprintf('/s%02d/', sub))
  
  fpath <- paste0(subpath, sprintf('wmConfidence_S%02d_allData_preprocessed.csv', sub))
  df <- read.csv(fpath, header = T, as.is = T, sep = ',') %>% select(-X)
  dataFiles[[sub]] <- df
  
}

data <- do.call('rbind', dataFiles)
fname <- paste0(dir, '/data/datafiles/wmConfidence_BehaviouralData_All.csv')
write.csv(data, file=fname, sep = ',', eol = '\n', dec = '.', col.names=T, row.names = F)