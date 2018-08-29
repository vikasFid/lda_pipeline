import sys
import spacy
import pandas as pd
import re
import math
import os
import datetime
import raccoon as rc
import nltk
import gensim
from gensim.models import word2vec
from gensim.models import phrases
import string
from nltk import word_tokenize
import random

#restrict chunk size here to get relevant bigrams/trigrams
def extract_useful_terms(interaction_id,nice_interaction_id,psent,customer_id,call_date,context,outputFile):

    #tackling one interaction id at a time, hence all context phrases can be written in one go
    all_imp_phrases = []
    for subtree in psent.subtrees():
        adjective = ''
        noun = ''
        verb = ''
        repPhrase = 0
        if subtree.label() == 'NP':
            # phrase = ' '.join(lmtzr.lemmatize(word) for word, tag in subtree.leaves())
            # print(phrase)
            for word, tag in subtree.leaves():
                # print(word)
                # print(tag)
                if tag in ['NN', 'NNS', 'CD']:
                    _noun = word
                    if (noun != ''):
                        if noun == _noun:
                            repPhrase = 1
                            break
                        else:
                            noun = noun + '_' + _noun
                    else:
                        noun = _noun
                elif tag in ['JJ', 'JJR', 'JJS']:
                    _adjective = word
                    if (adjective != ''):
                        if adjective == _adjective:
                            repPhrase = 1
                            break
                        else:
                            adjective = adjective + '_' + _adjective
                    else:
                        adjective = _adjective
                elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    _verb = word
                    if (verb != ''):
                        if verb == _verb:
                            repPhrase = 1
                            break
                        else:
                            verb = verb + '_' + _verb
                    else:
                        verb = _verb

            if repPhrase == 0 and verb != adjective and verb != noun and adjective != noun:
                if verb != '' and adjective != '':
                    phrase = verb + '_' + adjective + '_' + noun
                elif verb == '':
                    phrase = adjective + '_' + noun
                elif adjective == '':
                    phrase = verb + '_' + noun


                all_imp_phrases.append(phrase)

    with open(outputFile,'a') as f:
        merged_phrase = ' | '.join(phrase for phrase in list(all_imp_phrases))
        fline = str(interaction_id) + ',' + str(nice_interaction_id) + ',' + str(customer_id) + ',' + str(call_date) + ',' + merged_phrase + ',' + context
        f.write(fline+'\n')


#filters POS tags not capturing the most critical information, will need to be modified to capture opinions though
def filter_insignificant(chunk, tag_suffixes=['DT','CC','MD','PRP','PRP$','WDT','WRB','IN','RB','RBR','RBZ']):
    good = []
    stop_words = set(nltk.corpus.stopwords.words('english'))
    for word, tag in chunk:
        ok = True

        for suffix in tag_suffixes:
            if tag.endswith(suffix):
                ok = False
                break

        if ok and word not in stop_words and '\'' not in list(word):
            good.append((word, tag))

    return good


#this function will parse the context using spacy's POS tagger, although I feel NLTK tagger is working better for our use case
def filtered_chunks(doc):
  for w in doc.noun_chunks:
    signature = ''.join(['<%s>' % w.tag_ for w in w])
    pattern = re.compile(r'<V.*><NN.*|CD><JJ.*>')
    if pattern.match(signature) is not None:
      yield w


#this will chunk as per my logic using POS tags, note we can use spacy tags here as well, using the logic written in filtered_chunk function
def chunkProcessedTrans(df_clean_trans,outputFile):

    #time tracking:
    print('Cleaning and Chunking useful terms')
    print(datetime.datetime.now())
    start_time = datetime.datetime.now()

    #define grammar:
    adjGrammar = r'''
                            NP:
        							{<V.*><NN.*|CD><JJ.*>}
                                    {<V.*><JJ.*><NN.*|CD>}
                                    {<V.*><V.*><NN.*|CD>}
                                    {<V.*><NN.*|CD><NN.*|CD>}
                                    {<NN.*|CD><V.*><JJ.*>}
                                    {<JJ.*><V.*><NN.*|CD>}
                                    {<NN.*|CD><V.*><NN.*|CD>}
                                    {<NN.*|CD><JJ.*><V.*>}
                                    {<JJ.*><NN.*|CD><V.*>}
        							{<NN.*|CD><NN.*|CD><V.*>}
        							{<JJ.*><NN.*|CD><NN.*|CD>}
        							{<NN.*|CD><NN.*|CD><JJ.*>}
        							{<NN.*|CD><JJ.*><NN.*|CD>}
        							{<JJ.*><JJ.*><NN.*|CD>}
        							{<NN.*|CD><JJ.*><JJ.*>}
                            '''


    rowNum = 1
    for index,row in df_clean_trans.iterrows():
        print('Processing row :' + str(rowNum))
        rowNum = rowNum + 1
        interaction_id = row['interaction_id']
        nice_interaction_id = row['nice_interaction_id']
        customer_id = row['customer_id']
        call_date = row['call_date']
        context = row['phrase']
        context = context.replace("|"," ")
        context = context.replace("  "," ")
        text = word_tokenize(str(context))
        posTagged = nltk.pos_tag(text)
        cleanTags = filter_insignificant(posTagged)
        cp = nltk.RegexpParser(adjGrammar)
        adjResult = cp.parse(cleanTags)
        extract_useful_terms(interaction_id,nice_interaction_id,adjResult,customer_id,call_date,context,outputFile)


    print(datetime.datetime.now())
    end_time = datetime.datetime.now()
    print('Completed extracting important phrases in ::' + str(end_time - start_time))
    print('Fetched and Chunked useful terms')


#need to load dataframe as I need the interaction_id and not just a list of sentences, which will be a lot more faster using Gensim serialization
def parseTranscripts(inputFile,population,topicNum):
    #throw error if output file already exists:
    outputFile = 'SUMMARY_'+population+'_TOPIC_'+topicNum+'.csv'
    if os.path.isfile(outputFile):
        exit('Output file already exists!')
    else:
        #no previous version of output file exists, write header line
        with open(outputFile, 'a') as f:
            f.write('interaction_id,nice_interaction_id,customer_id,call_date,merged_phrase,context_phrase'+'\n')


    #loading to a dataframe
    df = pd.read_table(inputFile, sep=',', header=(0))

    print(datetime.datetime.now())
    start_time = datetime.datetime.now()
    print('Starting the process')
    #feed tokenized transcript through the POS context extractor logic, which will create final file for LDA input
    chunkProcessedTrans(df,outputFile)
    end_time = datetime.datetime.now()
    print('Completed process in ::' + str(end_time - start_time))


#this function builds LDA context model using gensim
def buildLDAModel():
    print('Will do this later')



if __name__ == "__main__":
    print('Starting Process')
    if(sys.argv[1] == None):
        print('Please provide the lda context file for which to create summary')
	exit(0)
    filename = sys.argv[1]
    if(sys.argv[2] == None):
        print('Please provide the context for the population')
	exit(0)
    population = sys.argv[2]
    if(sys.argv[3] == None):
        print('Please provide the topic number for which context is being created')
	exit(0)
    topicNum = sys.argv[3]
    parseTranscripts(filename,population,topicNum)















