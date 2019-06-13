library(dplyr)
getwd()
dir <- '/home/sammirc/Desktop/DPhil/wmConfidence'
dir <- '/Users/sammi/Desktop/Experiments/DPhil/wmConfidence'
setwd(dir)

datapath <- paste0(dir, '/data/datafiles')

# only doing this for specific subjects really, for now at least
sublist <- seq(1, 2, by = 1)
for(sub in sublist){
  subpath <- paste0(datapath, sprintf('/s%02d/', sub))
  
  fileList  <- sort(list.files(path = subpath, pattern = '.csv'))
  dataFiles <- list(NULL)
  count <- 1
  for(i in fileList){
    path <- paste0(subpath, '/', i)
    tmp <- read.csv(path, header = T, as.is = T, sep = ',')
    dataFiles[[count]] <- tmp
    count <- count + 1
  }
  
  df <- do.call('rbind', dataFiles)
  fname <- paste0(subpath, sprintf('wmConfidence_S%02d_allData.csv', sub))
  write.csv(df, file = fname, eol = '\n',col.names = T)
}
