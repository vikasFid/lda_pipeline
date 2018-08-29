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
    df_topic_words = pd.read_table("outputFiles/lambda_topic_terms.csv", sep=',', header=(0))
    return df_topic_words




#gets all relevant spans for the topic assignment
def getTopicContext(context,topic_string):
    
    topics = []
    selected_topic_words = getTopicTerms()
	
    if topic_string == 'ALL':
	num_topics = len(selected_topic_words.columns)
	for i in range(1,num_topics):
		topics.append(i)
    else:
	for topic in topic_string.split(','):
        	topics.append(topic)

    # read LDA output file
    print('Reading LDA classification')
    df = pd.read_table("outputFiles/DOC_TOPIC.csv", sep=',', header=(0),dtype={'nice_interaction_id': 'str'})	
    #df = df.dropna()
    df_topic_prob = pd.read_csv("outputFiles/TOPIC_PROB.csv", sep = ',')

    df_clean_trans = rc.DataFrame(columns=['interaction_id','nice_interaction_id','doc_topics','topic_words', 'phrase','origPhrase'])



    #collect all docs which have the current topic as statistically significant
    for index,row in df.iterrows():	

		interaction_id = row['interaction_id']
		nice_interaction_id = row['nice_interaction_id']
                mergedPhrase = row['cleanPhrase']
		chunkedPhrase = row['mergedPhrase']
		doc_topics = []

                #get document topic score : mean of relevant topics scores for each doc


		#check if any topic word is in the document:		
		#need to make it strict for the 
		#print(mergedPhrase)
		#print(chunkedPhrase)
		filteredChunks = []
		doc_topic_word_list = []
		for i,column in enumerate(selected_topic_words.columns):
			cur_topic = topics[i]
			doc_topic_words = []
			topic_presence = 0
                        topic_words = selected_topic_words[column].tolist()

			for w in topic_words:
				if str(mergedPhrase) != 'nan' and w in mergedPhrase.split():
					topic_presence = 1
					if w not in doc_topic_words:
						doc_topic_words.append(w)


			#hack for non phrased terms:
			if topic_presence == 1:
				topic_terms_sen = ' '.join(w for w in doc_topic_words)
				if '-' not in topic_terms_sen or len(doc_topic_words) == 1:
					topic_presence = 0;
				else:
					doc_topics.append(cur_topic)
                			#print(mergedPhrase)
                			if str(chunkedPhrase) != 'nan':
                        			chunkList = [chunk for chunk in chunkedPhrase.split('|')]
                        			for w in topic_words:
                                			for chunk in chunkList:
                                        			if w in chunk.split('_'):
                                                			if chunk not in filteredChunks:
                                                        			filteredChunks.append(chunk)


			for w in doc_topic_words:
				if w not in doc_topic_word_list:
					doc_topic_word_list.append(w)
			
		
		#write only those docs which could be assigned a topic to
		if(len(doc_topics) != 0):
			filteredChunkPhrase = '|'.join(chunk for chunk in filteredChunks)
			topic_word_string = '|'.join(w for w in doc_topic_word_list)
			doc_topic_list = '|'.join(str(t) for t in doc_topics)
                	df_clean_trans.append_row(index,{
                        	'interaction_id' : interaction_id,
				'nice_interaction_id' : nice_interaction_id,
				'doc_topics' : doc_topic_list,
                        	'topic_words' : topic_word_string,
                        	'phrase' : filteredChunkPhrase,
                        	'origPhrase' : mergedPhrase})


	
    data_dict = df_clean_trans.to_dict(index=False)
    df_clean_trans = pd.DataFrame(data_dict, columns=df_clean_trans.columns, index=df_clean_trans.index)
    df_clean_trans.sort_values(by = 'doc_topics',ascending = True)
    df_clean_trans.to_csv('outputFiles/lambda_based_topic.csv',index = False)


if __name__ == "__main__":
    print('Starting Process')
    #topic number to be passed by user as command line argument, decide topic number from LDAVis
    if(sys.argv[1] == None):
        print('Please provide the population from which context is to be extracted')
    if(sys.argv[2] == None):
        print('Please pass the topic number from the viz : http://10.240.154.22:8000/LDA_VIS/')
    getTopicContext(sys.argv[1],sys.argv[2])
    #getTopicContext([2,6,15])

