args<-commandArgs(TRUE)
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
#print(new.order)
orig.myAPdata = myAPdata
myAPdata$phi = myAPdata$phi[new.order, ]
myAPdata$theta = myAPdata$theta[, new.order]
word_topic_triplet = tidy(modelTrain)
word_topic_triplet = as.data.frame(word_topic_triplet)

#set lambda as per best fit from Viz, default 0.6
selected_lambda = as.numeric(args[1])
#nTopic = as.numeric(args[2])
topic.frequency = colSums(myAPdata$theta * myAPdata$doc.length)
topic.proportion = topic.frequency/sum(topic.frequency)
term.topic.frequency = myAPdata$phi * topic.frequency
term.frequency = colSums(term.topic.frequency)
term.proportion = term.frequency/sum(term.frequency)

#getting # of topics in model:
num_topics = length(new.order)


#original order calculations:
orig.topic.frequency = colSums(orig.myAPdata$theta * orig.myAPdata$doc.length)
orig.topic.proportion = orig.topic.frequency/sum(orig.topic.frequency)
orig.term.topic.frequency = orig.myAPdata$phi * orig.topic.frequency


orig.term.topic.frequency = as.data.frame(orig.term.topic.frequency)
colnames(orig.term.topic.frequency) = orig.myAPdata$vocab
orig.term.topic.frequency$topic_num = c(1:num_topics)	


#writing output files

nTerms = args[2]
selected_topics = c(4,17,20,31)
#writing terms in order of relevance
lambda = selected_lambda
output = c()
relTerms = c()
added_words = c()
num_selected_topics = length(selected_topics)
item_count = 0
for (i in selected_topics){

  	#picking topic in a loop here, calculating relevance order one topic at a time!
  	lift = myAPdata$phi[i,]/term.proportion

  	rel = as.data.frame(lambda*log(myAPdata$phi[i,]) + (1 - lambda)*log(lift))

  	idx = apply(rel, 2, function(x) order(x, decreasing = TRUE))
  	# for matrices, we pick out elements by their row/column index
  	#indices = cbind(c(idx))
  	myAPdata$relevance = data.frame(Term = myAPdata$vocab[idx],
                                   logprob = round(log(myAPdata$phi[idx]), 4),
                                   loglift = round(log(lift[idx]), 4),
                                   stringsAsFactors = FALSE)
  

	


	print(i)
	for(term in myAPdata$relevance[,1]){

		if(!term %in% relTerms){
			
			print(term)

			relevant_term_flag = 1

			#get the top topics for this word, the word should have either of the selected topics as top K, K being # of selcted topics, K here is num_selected_topics
			#print(paste("Checking for term:",term))
			top_topic_set = c()
			#print(word_topic_triplet)
			word_topic_dis = as.data.frame(orig.term.topic.frequency[,c("topic_num",term)])

			#fetch the topics that satisfy the freq cutoff >= 0.5
			word_topic_dis = word_topic_dis[word_topic_dis[term] >= 0.5,]

			#sort the word_topic_dis by the conditional prob. value
			word_topic_dis = as.data.frame(word_topic_dis[order(-word_topic_dis[,2]),])
			#print(nrow(word_topic_dis))

			#getting top topics for that term
			for(topic in word_topic_dis[1:num_selected_topics,]$topic_num){
				if(!is.na(topic)){
					topic_count = length(top_topic_set)
					top_topic_set[topic_count + 1] = which(prevJson$topic.order == topic)
				}
			}

			#now check if the top K topics are the selected topics in any order


				
			#collecting terms for a topic combination, the terms should be top term in each of the selected topics:	
		
			for(t in selected_topics){	
				if(nrow(word_topic_dis) >= num_topics/2 || (!t %in% top_topic_set)){
					relevant_term_flag = 0
				}

			}

			#found relevant word for the topic, add it as an exclusive word for the topic
			#write term beta to a separate file as well
			if(relevant_term_flag == 1){			
    				item_count = length(relTerms)
    				relTerms[item_count + 1] = term
			}




		}


  
  
	}

}

#note that we have already written the Doc-Topic assignement, topic proportion and topic probability files by now
#write relevance as well here
#write.csv(output,file = 'LDA_FILES/TOPIC_TERMS.csv',row.names = F, quote =  F)
write.csv(relTerms,file = 'outputFiles/related_topic_list.csv',
          row.names = F, quote =  F)

#warnings()
