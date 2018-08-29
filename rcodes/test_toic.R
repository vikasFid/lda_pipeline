library(tm)
library(topicmodels)
library(tokenizers)
library(stringr)
library(sqldf)
library(LDAvis)
library(tidyr)
library(dplyr)
library(tidytext)

#load RData from model building code
load(file = "outputFiles/LDAVis_JSON.RData")
load(file = "outputFiles/myAPdata.RData")
load(file = "outputFiles/ldaModel.RData")


new.order = RJSONIO::fromJSON(json)$topic.order
prevJson = RJSONIO::fromJSON(json)
topic.frequency = colSums(myAPdata$theta * myAPdata$doc.length)
topic.proportion = topic.frequency/sum(topic.frequency)
term.topic.frequency = myAPdata$phi * topic.frequency
term.frequency = colSums(term.topic.frequency)
term.proportion = term.frequency/sum(term.frequency)


test = as.data.frame(term.topic.frequency)
colnames(test) = myAPdata$vocab
test$topic_num = c(1:50)

temp = as.data.frame(test[c("topic_num","bit")])

new_temp = as.data.frame(temp[order(-temp[,2]),])

new_temp
