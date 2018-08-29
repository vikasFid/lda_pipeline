library(topicmodels)
library(tidytext)
library(tm)
library(stringr)
library(tidyr)
library(reshape2)
library(dplyr)

load(file = "outputFiles/myAPdata.RData")
load(file = "outputFiles/ldaModel.RData")


probTopics = tidy(modelTrain,matrix = 'beta')
ap_top_terms <- probTopics %>%
  group_by(topic) %>%
  top_n(50, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

write.csv(ap_top_terms,'TOPIC_WORDS.csv')






