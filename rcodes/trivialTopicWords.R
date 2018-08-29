#to take command line arguments
args<-commandArgs(TRUE)

#free up memory from previous runs
gc()

library(tm)
library(dplyr)
library(Matrix)
library(tidytext)

#inputFile = args[1]
topicWords = read.csv('outputFiles/TOPIC_TERMS.csv',header = TRUE,sep = ',',colClasses = "character",nrows = 50)

#nrow(topicWords)

#trivialTopics <- unlist(strsplit(args[1],","))
trivialTopics = c(2,19,28,40,41,45,47,48,50)
trivialTopics
for(i in trivialTopics){
  #gets the 50 most popular words in a topic
  #exclude only if it is not an important word in other topics?? will do later
  topic = as.integer(i)
  topic
  topicTerms = topicWords[,topic]
  write.table(topicTerms, file = 'trivialTopicTerms.txt',row.names = F,quote = F,append = T)
}


trivialWords = read.csv('trivialTopicTerms.txt',
                        header = TRUE,sep = ',',colClasses = "character")


