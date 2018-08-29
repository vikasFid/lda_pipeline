import pandas as pd
import re
import math
import os
import datetime
import raccoon as rc
import nltk
import sys
import numpy as np



#this function builds the list of topic words which are specific to the topic, removing words that occur in all topics
def getTopicTerms():
    df_topic_words = pd.read_table("outputFiles/related_topic_list.csv", sep=',', header=(0))

    return df_topic_words['x'].tolist()




#gets all relevant spans for the topic assignment
def getTopicContext(population,topic_string):


    topics = []
    for topic in topic_string.split(','):
        topics.append(topic)

    # read LDA output file
    print('Reading LDA classification')
    df = pd.read_table("outputFiles/DOC_TOPIC.csv", sep=',', header=(0),dtype={'nice_interaction_id': 'str'})	
    #df = df.dropna()
    df_topic_prob = pd.read_csv("outputFiles/TOPIC_PROB.csv", sep = ',')

    df_clean_trans = rc.DataFrame(columns=['doc_topic_score','interaction_id','nice_interaction_id','customer_id','call_date','topic_words', 'phrase','origPhrase'])

	


    topic_words = getTopicTerms()
    print(topic_words)


    #collect all docs which have the current topic as statistically significant
    for index,row in df.iterrows():	

		interaction_id = row['interaction_id']
                nice_interaction_id = row['nice_interaction_id']
                customer_id = row['customer_id']
                call_date = row['call_date']
                mergedPhrase = row['mergedPhrase']
                cleanPhrase = row['cleanPhrase']

                #get document topic score : mean of relevant topics scores for each doc
                topic_scores = []
                for topic in topics:
                        topic_scores.append(df_topic_prob.ix[index]["V"+str(topic)]) # gets the topic score for the doc

                doc_topic_score = np.mean(topic_scores)



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
    df_clean_trans.to_csv('outputFiles/related_topics_phrases.csv')


if __name__ == "__main__":
    print('Starting Process')
    #topic number to be passed by user as command line argument, decide topic number from LDAVis
    if(sys.argv[1] == None):
        print('Please provide the population from which context is to be extracted')
    if(sys.argv[2] == None):
        print('Please pass the topic number from the viz : http://10.240.154.22:8000/LDA_VIS/')
    getTopicContext(sys.argv[1],sys.argv[2])
    #getTopicContext([2,6,15])

