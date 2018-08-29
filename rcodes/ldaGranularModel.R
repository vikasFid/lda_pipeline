args<-commandArgs(TRUE)
library(tm)
library(topicmodels)
#library(slam)
library(tokenizers)
library(stringr)
library(Matrix)
library(tidytext)
library(igraph)
library(sqldf)



inputFile = args[1]
txt_input = read.csv(inputFile,header = TRUE,sep = ',',colClasses = "character")

txt_data = as.character(txt_input[,6])
#writeLines(txt_data[4])

txt_data = gsub("_"," ",txt_data)
txt_data = gsub("\\|"," ",txt_data)
txt_data = stripWhitespace(txt_data)
writeLines(txt_data[4])

textCorpus = VCorpus(VectorSource(txt_data))
#tokenizer <- function(x)unlist(lapply(ngrams(strsplit(as.character(x),split="|",fixed=T), 1),paste, collapse = " "), use.names = FALSE)
tokenizer <- function(x)unlist(lapply(ngrams(words(x), 1),paste, collapse = " "), use.names = FALSE)

ctrl <- list(tokenize=tokenizer,bounds = list(global=c(2,Inf))) 
dtm = DocumentTermMatrix(textCorpus, control = ctrl)
dtm


tf_idf_filter = args[3]
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
nTopic = args[2]
modelTrain = LDA(dtm,k = nTopic,method="Gibbs",
                 control = list(alpha = 0.1),
                 iter = 2000,
                 beta = 0.001) # "Gibbs" or "VEM"
#load(file = "C:/Users/a592407/Desktop/LDA_REDEEM_UNIGRAM_20_TOPIC.RData")
#model.train <- CTM(topic.train,nTopic,method="VEM")
#save(modelTrain, file = "C:/Users/a592407/Desktop/LDA_REDEEM_UNIGRAM_20_TOPIC.RData")
#term1 = terms(modelTrain,10)
#term1


#LDA Visualization
library(LDAvis)
library(tsne)
library(servr)

rowTotals = slam::row_sums(dtm, na.rm = T)
colTotals = slam::col_sums(dtm, na.rm = T)


myAPdata <- list()
myAPdata$vocab <- modelTrain@terms
myAPdata$term.frequency <- as.integer(colTotals)
myAPdata$phi <- exp(modelTrain@beta)
myAPdata$theta <- modelTrain@gamma
myAPdata$doc.length <- as.integer(rowTotals)


json <- createJSON(phi=myAPdata$phi,
                   theta=myAPdata$theta,
                   vocab=myAPdata$vocab,
                   doc.length = myAPdata$doc.length,
                   term.frequency = myAPdata$term.frequency,
                   mds.method = function(x) tsne(svd(x)$u),
                   plot.opts = list(xlab="", ylab="")
)

#save(json, file = "C:/Users/a592407/Desktop/LDAVis_TEST.RData")
#load(file = "C:/Users/a592407/Desktop/LDAVis_TEST.RData")
serVis(json,out.dir="LDA_GRANULAR_VIS", open.browser = F)


