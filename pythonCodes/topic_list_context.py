import pandas as pd
import re
import math
import os
import datetime
import raccoon as rc
import nltk
import sys




#this function builds the list of topic words which are specific to the topic, removing words that occur in all topics
def getTopicTerms(topic,numTopics):
    df_topic_words = pd.read_table("outputFiles/TOPIC_TERMS.csv", sep=',', header=(0))
    #hash for word frequency
    word_dict = {}
    for col in df_topic_words.columns:
        colWords = df_topic_words[col]
        for w in colWords:
            if w in word_dict:
                word_dict[w] = word_dict[w] + 1
            else:
                word_dict[w] = 1


    #now hash is built with counts, select all words who are unique to the topic
    words = df_topic_words.ix[:,(topic-1)]
    topic_words = []
    for w in words:
        if word_dict[w] < 3 and w not in topic_words:
            topic_words.append(w)


    return topic_words



def getTopicDistribution(topic_dis,topic):
	
	topic_dis_summary = topic_dis.describe()
	return topic_dis_summary["mean"]




#gets all relevant spans for the topic assignment
def getTopicContext(population,topicNum,numTopics):

    # read LDA output file
    print('Reading LDA classification')
    df = pd.read_table("outputFiles/DOC_TOPIC.csv", sep=',', header=(0),dtype={'nice_interaction_id': 'str'})	
    #df = df.dropna()
    df_topic_prob = pd.read_csv("outputFiles/TOPIC_PROB.csv", sep = ',')

    df_clean_trans = rc.DataFrame(columns=['doc_topic_score','interaction_id','nice_interaction_id','customer_id','call_date','topic_words', 'phrase','origPhrase'])

    topic_words = getTopicTerms(topicNum,numTopics)
    print(topic_words)
	
    topic = "V"+str(topicNum)
    topic_dis = df_topic_prob[topic]
    topic_prob_cutoff = getTopicDistribution(topic_dis,topic)
    topic_dis = topic_dis.to_frame()

    #collect all docs which have the current topic as statistically significant
    for index,row in topic_dis.iterrows():	
		topic_prob = row[0]
		doc_topic_row = df.ix[index]
		interaction_id = doc_topic_row['interaction_id']
        	nice_interaction_id = doc_topic_row['nice_interaction_id']
        	customer_id = doc_topic_row['customer_id']
        	call_date = doc_topic_row['call_date']
       	 	mergedPhrase = doc_topic_row['mergedPhrase']
        	cleanPhrase = doc_topic_row['cleanPhrase']
		doc_topic_score = topic_prob

		#check if any topic word is in the document:		
		topic_presence = 0
		doc_topic_words = []
		for w in topic_words:
			if str(cleanPhrase) != 'nan' and w in cleanPhrase.split():
				topic_presence = 1
				if w not in doc_topic_words:
					doc_topic_words.append(w)
				

		#collect chunks with topic words, note that we may miss chunks here as NLP Chunker may have missed those chunks
        	filteredChunks = []
		#print(mergedPhrase)
		if str(mergedPhrase) != 'nan':
			chunkList = [chunk for chunk in mergedPhrase.split('|')]
			for w in topic_words:
                		for chunk in chunkList:
                        		if w in chunk.split('_'):
                                		if chunk not in filteredChunks:
							filteredChunks.append(chunk)

		if topic_presence != 0:
			filteredChunkPhrase = '|'.join(chunk for chunk in filteredChunks)
			topic_word_string = '|'.join(w for w in doc_topic_words)
			df_clean_trans.append_row(index,{
					 'doc_topic_score' : doc_topic_score,
					 'interaction_id' : interaction_id,
                                         'nice_interaction_id' : nice_interaction_id,
                                         'customer_id' : customer_id,
                                         'call_date' : call_date,
					 'topic_words' : topic_word_string,
                                         'phrase' : filteredChunkPhrase,
					 'origPhrase' : cleanPhrase})



	
    data_dict = df_clean_trans.to_dict(index=False)
    df_clean_trans = pd.DataFrame(data_dict, columns=df_clean_trans.columns, index=df_clean_trans.index)
    df_clean_trans['doc_topic_score'] = df_clean_trans['doc_topic_score'].astype('float')
    df_clean_trans.sort_values(by = 'doc_topic_score',ascending = False)
    df_clean_trans.to_csv('outputFiles/LDA_TOPIC_CONTEXT_PHRASES.csv')


if __name__ == "__main__":
    print('Starting Process')
    '''
    #topic number to be passed by user as command line argument, decide topic number from LDAVis
    if(sys.argv[1] == None):
	print('Please provide the population from which context is to be extracted')
    if(sys.argv[2] == None):
        print('Please pass the topic number from the viz : http://10.240.154.22:8000/LDA_VIS/')
    if(sys.argv[3] == None):
	print('Please provide the number of topics in the vizualization')
    getTopicContext(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
    '''
    getTopicContext([2,8,37])
