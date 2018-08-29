#to take command line arguments
args<-commandArgs(TRUE)

#free up memory from previous runs
gc()

library(tm)
library(dplyr)
library(topicmodels)
library(LDAvis)
library(tsne)
library(servr)
library(Matrix)
library(tidytext)
library(igraph)
library(sqldf)

#inputFile = args[1]
txt_input = read.csv('adhoc_chunked_output.csv',
                     header = TRUE,sep = ',',colClasses = "character")


txt_data = as.character(txt_input[,2])
#writeLines(txt_data[4])

#txt_data = gsub("_"," ",txt_data)
txt_data = gsub("\\|"," ",txt_data)
txt_data = stripWhitespace(txt_data)
writeLines(txt_data[4])

textCorpus = VCorpus(VectorSource(txt_data))
#tokenizer = function(x)unlist(lapply(ngrams(strsplit(as.character(x),split="|",fixed=T), 1),paste, collapse = " "), use.names = FALSE)

removeTrivialTopics = args[3]
if(removeTrivialTopics == 1){
  trivialWords = read.csv('trivialTopicTerms.txt',
                          header = TRUE,sep = ',',colClasses = "character")
  trivialWords = as.character(trivialWords$x)
  textCorpus <- tm_map(textCorpus, removeWords, trivialWords)
}  





tokenizer = function(x)unlist(lapply(ngrams(words(x), 1),paste, collapse = " "), use.names = FALSE)

#dtm with tf, required for the LDA model
ctrl = list(tokenize=tokenizer,bounds = list(global=c(2,Inf)))
dtm = DocumentTermMatrix(textCorpus, control = ctrl)
dtm


tf_idf_filter = args[2]
if(tf_idf_filter == 1){

n_row = dtm$nrow
n_col = dtm$ncol




#dtm with normalized tf_idf, required for more accurate model
ctrl_tfidf = list(tokenize=tokenizer,bounds = list(global=c(2,Inf)),
                  weighting = function(x) weightTfIdf(x, normalize = TRUE)) 

dtm_tfidf = DocumentTermMatrix(textCorpus, control = ctrl_tfidf)

#creating sparse matrix
dtm = sparseMatrix(i=dtm$i, j=dtm$j, x=dtm$v,dimnames = dtm$dimnames,
                    dims=c(dtm$nrow, dtm$ncol))
dtm_tfidf = sparseMatrix(i=dtm_tfidf$i, j=dtm_tfidf$j, x=dtm_tfidf$v,
                         dimnames = dtm_tfidf$dimnames,
                         dims=c(dtm_tfidf$nrow, dtm_tfidf$ncol))

#get index and values of the sparse matrix
sum_dtm_tfidf = summary(dtm_tfidf)

#determine this cutoff using boxplot or the tf-idf score for a common word like account
#cutoff = as.numeric(summary(sum_dtm_tfidf$x)[3])

#other method to get cutoff is to get the mean score for a common word
temp = as.matrix(dtm_tfidf[,"account"])
#temp = temp[temp != 0,]
#summary(temp)
#need words to be at least as important as account, need to make it more precise as per zipf's law
cutoff = max(temp[temp!= 0,])


#below code sets dtm tf to 0 for a word that correspondingly in the dtm_tfidf 
#has a very low tf-idf score
#taking values that are lower than the cutoff
keep_index = sum_dtm_tfidf[sum_dtm_tfidf$x > cutoff && sum_dtm_dtmtfidf$v %not in% trivialWords,1:2 ]


#making replace matrix have same dimension as the dtm
new_row = c(n_row,n_col)
keep_index = rbind(keep_index,new_row)

#adjacency matrix
adjMat = get.adjacency(graph.edgelist(as.matrix(keep_index), directed=TRUE))


#resize adjMat to the size of the dtm, no need to flip bits as it is very inefficient in sparseMatrix
adjMat = adjMat[1:n_row,1:n_col]

#element-wise product
dtm = dtm * adjMat


#dtm is sparse matrix, converting it back to dtm here
#dtm = tidy(dtm)
#dtm = dtm %>%cast_dtm(row, column, value)
#dtm
dtm = as.DocumentTermMatrix(dtm,weighting = function(x) weightTf(x))

}

#removing any doc with no terms
rowTotals = slam::row_sums(dtm, na.rm = T)
colTotals = slam::col_sums(dtm, na.rm = T)
dtm = dtm[rowTotals != 0,]


#because we have removed rows that have zero rowTotals, calculating dimension totals again
rowTotals = slam::row_sums(dtm, na.rm = T)
colTotals = slam::col_sums(dtm, na.rm = T)
dtm


#Topic Modeling: LDA
#Gibbs sampling works better with limited RAM
print('Building Model Now!')
nTopic = args[1]
modelTrain = LDA(dtm,k = nTopic,method="Gibbs",
                 control = list(alpha = 0.1),
                 iter = 2000,
                 beta = 0.001) # "Gibbs" or "VEM"


#save the model in case you want to use it later to find more topic calls
save(modelTrain,file = "adhocOutput/ldaModel.RData")


#LDA Visualization

myAPdata = list()
myAPdata$vocab = modelTrain@terms
myAPdata$term.frequency = as.integer(colTotals)
myAPdata$phi = exp(modelTrain@beta)
myAPdata$theta = modelTrain@gamma
myAPdata$doc.length = as.integer(rowTotals)


json = createJSON(phi=myAPdata$phi,
                  theta=myAPdata$theta,
                  vocab=myAPdata$vocab,
                  doc.length = myAPdata$doc.length,
                  term.frequency = myAPdata$term.frequency,
                  mds.method = function(x) tsne(svd(x)$u),
                  plot.opts = list(xlab="", ylab="")
)

#save(json, file = "C:/Users/a592407/Desktop/LDA_RDATA/LDAVis_361693.RData")
#load(file = "C:/Users/a592407/Desktop/LDA_RDATA/LDAVis_REDEEM_ALL.RData")
#serVis(json,out.dir="C:/Users/A592407/Desktop/LDA_VIS", open.browser = T)


print("Saving model to RData")
save(json, file = "adhocOutput/LDAVis_JSON.RData")
save(myAPdata, file = "adhocOutput/myAPdata.RData")

#load(file = "C:/Users/a592407/Desktop/LDA_RDATA/LDAVis_REDEEM_ALL.RData")
serVis(json,out.dir="adhoc_viz", open.browser = F)


#to make topic order same as LDAVis topic order
modelTrain@gamma = myAPdata$theta


#probabilities associated with each topic assignment
topicProbabilities = as.data.frame(modelTrain@gamma)
write.csv(topicProbabilities,file="adhocOutput/TOPIC_PROB.csv",
           row.names = F)

#alternative logic:
docTopic = as.data.frame(cbind(document = row.names(topicProbabilities),
                                topic = apply(topicProbabilities,1,function(x) names(topicProbabilities)[which(x==max(x))])))

docTopic = data.frame(lapply(docTopic,as.character),stringsAsFactors = T)


#document topic assignment(add logic for threshold!), use above alternate logic!
#docTopic = as.data.frame(topics(modelTrain))
#write.csv(table(docTopic),file = "C:/Users/a592407/Documents/LDA_FILES/REDEEM_DOC_PER_TOPIC_2.csv",
#           row.names = F)


doc_index = rownames(docTopic)

doc_df = data.frame(doc_id = txt_input[doc_index,1],
		     mergedPhrase = txt_input[doc_index,2],
		     chunkedPhrase = txt_input[doc_index,3],
                     topic = docTopic[,2])

write.csv(doc_df,file = "adhocOutput/DOC_TOPIC.csv",row.names = F)


#topic proportions as per docTopic assignment:
#library(sqldf)
groupedData = sqldf('SELECT topic, count(*) AS numDocs FROM doc_df GROUP BY topic')
groupedData$topicProportion = (groupedData$numDocs/sum(groupedData$numDocs))
write.csv(groupedData[order(-groupedData$topicProportion),],
          'adhocOutput/DOC_TOPIC_PROP.csv',quote = F,
          row.names = F)



#write topic call_date to see if there is spike in doc-topic assignment
#output = c()
#for (i in as.character(unique(unlist(doc_df$topic)))){

#doc_df_topic = doc_df[doc_df$topic == i,]
#print(nrow(doc_df_topic))
#group_topic_date = sqldf('SELECT call_date, count(*) AS numDocs FROM doc_df_topic GROUP BY call_date')
#group_topic_date$topic = i
#group_topic_date = group_topic_date[order(as.Date(group_topic_date$call_date))]
#output = rbind(output,group_topic_date)

#}


#write it to a file
#write.table(output,'topic_date_prop.tsv',quote = F,sep = '\t',row.names = F)






#clean up
print('Cleaning objects from RAM and exiting')
rm(list=ls())








