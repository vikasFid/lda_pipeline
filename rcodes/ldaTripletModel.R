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

txt_input = read.csv("../OutputFiles/RANDY_NLP_CONTEXT.csv",
                     header = TRUE,sep = ',',colClasses = "character")

txt_data = as.character(txt_input[,5])
#writeLines(txt_data[4])

#txt_data = gsub("_"," ",txt_data)
txt_data = gsub("\\|"," ",txt_data)
txt_data = stripWhitespace(txt_data)
writeLines(txt_data[4])

textCorpus = VCorpus(VectorSource(txt_data))
#tokenizer = function(x)unlist(lapply(ngrams(strsplit(as.character(x),split="|",fixed=T), 1),paste, collapse = " "), use.names = FALSE)
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
cutoff = as.numeric(summary(sum_dtm_tfidf$x)[3])

#other method to get cutoff is to get the mean score for a common word
#temp = as.matrix(dtm_tfidf[,"account"])
#temp = temp[temp != 0,]
#summary(temp)


#below code sets dtm tf to 0 for a word that correspondingly in the dtm_tfidf 
#has a very low tf-idf score
#taking values that are lower than the cutoff
keep_index = sum_dtm_tfidf[sum_dtm_tfidf$x > cutoff,1:2]

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
save(json, file = "../LDA_RDATA/LDAVis_JSON.RData")
save(myAPdata, file = "../LDA_RDATA/myAPdata.RData")

#load(file = "C:/Users/a592407/Desktop/LDA_RDATA/LDAVis_REDEEM_ALL.RData")
serVis(json,out.dir="../Visualizations/LDA_VIS", open.browser = F)


#to make topic order same as LDAVis topic order
modelTrain@gamma = myAPdata$theta


#probabilities associated with each topic assignment
topicProbabilities = as.data.frame(modelTrain@gamma)
write.csv(topicProbabilities,file="../LDA_FILES/TOPIC_PROB.csv",
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
write.csv(data.frame(interaction_id = txt_input[doc_index,1],
                     nice_interaction_id = as.character(txt_input[doc_index,2]),
                     customer_id = as.character(txt_input[doc_index,3]),
                     call_date = txt_input[doc_index,4],
                     doc = txt_input[doc_index,5],
                     topic = docTopic[,2]),
                     file = "../LDA_FILES/DOC_TOPIC.csv",
                     row.names = F)


#topic proportions as per docTopic assignment:
#library(sqldf)
docTopic = as.data.frame(docTopic)
groupedData = sqldf('SELECT topic, count(*) AS numDocs FROM docTopic GROUP BY topic')
groupedData$topicProportion = (groupedData$numDocs/sum(groupedData$numDocs))*100
write.csv(groupedData[order(-groupedData$topicProportion),],
          '../LDA_FILES/DOC_TOPIC_PROP.csv',quote = F,
          row.names = F)

#clean up
print('Cleaning objects from RAM and exiting')
rm(list=ls())








