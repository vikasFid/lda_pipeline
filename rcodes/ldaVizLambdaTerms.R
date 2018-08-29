args<-commandArgs(TRUE)
library(tm)
library(topicmodels)
library(tokenizers)
library(stringr)
library(sqldf)
library(LDAvis)

#load RData from model building code
load(file = "outputFiles/LDAVis_JSON.RData")
load(file = "outputFiles/myAPdata.RData")

new.order = RJSONIO::fromJSON(json)$topic.order
myAPdata$phi = myAPdata$phi[new.order, ]
myAPdata$theta = myAPdata$theta[, new.order]

#set lambda as per best fit from Viz, default 0.6
selected_lambda = as.numeric(args[1])
nTopic = as.numeric(args[2])
topic.frequency = colSums(myAPdata$theta * myAPdata$doc.length)
topic.proportion = topic.frequency/sum(topic.frequency)
term.topic.frequency = myAPdata$phi * topic.frequency
term.frequency = colSums(term.topic.frequency)
term.proportion = term.frequency/sum(term.frequency)


#writing output files

nTerms = args[3]
#writing terms in order of relevance
output = c()
for (i in 1:nTopic){

	#initiate topic terms list
	topicTerms = c()
	for(lambda in seq(from = selected_lambda, to = 0.1, by = -0.1)){

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
  
		for(term in myAPdata$relevance[1:nTerms,1]){
  			if (!term%in%topicTerms){
    				item_count = length(topicTerms)
    				topicTerms[item_count + 1] = term
  			}  
  
		}
	
	}


	#write terms for all topics to a file:
	output = cbind(output,topicTerms)	


}

#note that we have already written the Doc-Topic assignement, topic proportion and topic probability files by now
#write relevance as well here
#write.csv(output,file = 'LDA_FILES/TOPIC_TERMS.csv',row.names = F, quote =  F)
write.csv(output,file = 'outputFiles/TOPIC_TERMS.csv',
          row.names = F, quote =  F)

#warnings()
