import sys
import spacy
# import en_core_web_md
import en_core_web_sm
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
from nltk.corpus import wordnet
from itertools import chain
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import operator
import pickle
import unicodedata


# restrict chunk size here to get relevant bigrams/trigrams
def extract_useful_terms(doc_id,psent, gensimPhrase, nouns):
    # tackling one interaction id at a time, hence all context phrases can be written in one go
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

    with open('adhoc_chunked_output.csv', 'a') as f:
        merged_phrase = ' | '.join(phrase for phrase in list(all_imp_phrases))
        fline = str(doc_id) + ',' + gensimPhrase + ',' +merged_phrase+ ',' +nouns
        f.write(fline + '\n')


# filters POS tags not capturing the most critical information, will need to be modified to capture opinions though
def filter_insignificant(chunk, tag_suffixes=['DT', 'CC', 'MD', 'PRP', 'PRP$', 'WDT', 'WRB', 'IN', 'RB', 'RBR', 'RBZ']):
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


# this function will parse the context using spacy's POS tagger, although I feel NLTK tagger is working better for our use case
def filtered_chunks(doc):
    for w in doc.noun_chunks:
        signature = ''.join(['<%s>' % w.tag_ for w in w])
        pattern = re.compile(r'<V.*><NN.*|CD><JJ.*>')
        if pattern.match(signature) is not None:
            yield w


# this will chunk as per my logic using POS tags, note we can use spacy tags here as well, using the logic written in filtered_chunk function
def chunkProcessedTrans(df_clean_trans):
    # time tracking:
    print('Cleaning and Chunking useful terms')
    print(datetime.datetime.now())
    start_time = datetime.datetime.now()

    # define grammar:
    adjGrammar = r'''
                            NP:
        			    {<V.*><NN.*|CD>+<JJ.*>}
                                    {<V.*><JJ.*><NN.*|CD>+}
                                    {<V.*><V.*><NN.*|CD>+}
                                    {<V.*><NN.*|CD>+<NN.*|CD>}
                                    {<NN.*|CD>+<V.*><JJ.*>}
                                    {<JJ.*><V.*><NN.*|CD>+}
                                    {<NN.*|CD>+<V.*><NN.*|CD>+}
                                    {<NN.*|CD>+<JJ.*><V.*>}
                                    {<JJ.*><NN.*|CD>+<V.*>}
        		       	    {<NN.*|CD><NN.*|CD>+<V.*>}
        			    {<JJ.*><NN.*|CD>+<NN.*|CD>}
        			    {<NN.*|CD><NN.*|CD>+<JJ.*>}
        			    {<NN.*|CD>+<JJ.*><NN.*|CD>+}
        			    {<JJ.*><JJ.*><NN.*|CD>+}
        			    {<NN.*|CD>+<JJ.*><JJ.*>}
                            '''

    for index, row in df_clean_trans.iterrows():
        cleanTrans = row['phrase']
        nouns = row['nouns']
	doc_id = row['doc_id']
        text = word_tokenize(str(cleanTrans))
        posTagged = nltk.pos_tag(text)
        cleanTags = filter_insignificant(posTagged)
        cp = nltk.RegexpParser(adjGrammar)
        adjResult = cp.parse(cleanTags)
        extract_useful_terms(doc_id,adjResult,cleanTrans, nouns)

    print(datetime.datetime.now())
    end_time = datetime.datetime.now()
    print('Completed extracting important phrases in ::' + str(end_time - start_time))
    print('Fetched and Chunked useful terms')


# cleaning phrase for business specific context, removing rep generic conversations
def cleanTranscript(transcript):
    trans = str.lower(transcript)
    trans = trans.replace('[number redacted]', ' ')
    trans = trans.replace('[name redacted]', ' ')
    trans = trans.replace('[laughter]', ' ')
    trans = trans.replace('[noise]', ' ')
    trans = trans.replace('[vocalized-noise]', ' ')
    trans = trans.replace(' uh- ', ' ')
    trans = trans.replace('mhm', ' ')
    trans = trans.replace('uhm', ' ')
    trans = trans.replace('huh', ' ')
    trans = trans.replace('uh-huh', ' ')
    trans = trans.replace(' ooh ', ' ')
    trans = trans.replace(' mrs ', ' ')
    trans = trans.replace('yeah', ' ')
    trans = trans.replace(' gosh ', ' ')
    trans = trans.replace(' wanna ', ' ')
    trans = trans.replace('okay', ' ')
    trans = trans.replace(' gonna ', ' ')
    # trans = trans.replace(' know ', ' ')
    trans = trans.replace(' does ', ' ')
    trans = trans.replace(' goes ', ' ')
    trans = trans.replace(' how ', ' ')
    trans = trans.replace(' yup ', ' ')
    trans = trans.replace(' yep ', ' ')
    # trans = trans.replace(' like ', ' ')
    trans = trans.replace(' yes ', ' ')
    trans = trans.replace('thanks', ' ')
    trans = trans.replace('thank', ' ')
    trans = trans.replace(' help you today ', ' ')
    trans = trans.replace(' you ', ' ')
    trans = trans.replace(' sorta ', ' ')
    trans = trans.replace(' good bye ', ' ')
    trans = trans.replace(' goodbye ', ' ')
    trans = trans.replace(' bye ', ' ')
    trans = trans.replace(' hey ', ' ')
    trans = trans.replace(' haha ', ' ')
    trans = trans.replace(' maam ', ' ')
    trans = trans.replace('good day', ' ')
    trans = trans.replace('hello', ' ')
    trans = trans.replace('gotcha', ' ')
    trans = trans.replace(' gotta ', ' ')
    trans = trans.replace('<unk>', ' ')
    trans = trans.replace(' good afternoon ', ' ')
    trans = trans.replace(' afternoon ', ' ')
    trans = trans.replace('good morning', '')
    trans = trans.replace(' kinda ', ' ')
    trans = trans.replace(' morning ', ' ')
    trans = trans.replace('connecting support', ' ')
    trans = trans.replace('neck support', ' ')
    trans = trans.replace('necking support', ' ')
    # trans = trans.replace(' great ', ' ')
    trans = trans.replace(' welcome ', ' ')
    trans = trans.replace(' well ', ' ')
    trans = trans.replace(' name\'s ', ' ')
    trans = trans.replace(' place ', ' ')
    # trans = trans.replace(' right ', ' ')
    trans = trans.replace(' day ', ' ')
    trans = trans.replace(' days ', ' ')
    trans = trans.replace(' things ', ' ')
    trans = trans.replace(' thing ', ' ')
    trans = trans.replace(' mean ', ' ')
    # trans = trans.replace(' work ', ' ')
    # trans = trans.replace(' lot ', ' ')
    trans = trans.replace(' kind ', ' ')
    trans = trans.replace(' com ', ' ')
    trans = trans.replace(' dot ', ' ')
    # trans = trans.replace(' percent ', ' ')
    trans = trans.replace(' dollar ', ' ')
    trans = trans.replace(' dollars ', ' ')
    trans = trans.replace(' cent ', ' ')
    trans = trans.replace(' cents ', ' ')
    # trans = trans.replace(' make ', ' ')
    # trans = trans.replace(' bit ', ' ')
    trans = trans.replace(' nah ', ' ')
    trans = trans.replace(' sure ', ' ')
    trans = trans.replace(' sir ', ' ')
    # trans = trans.replace(' look ', ' ')
    # trans = trans.replace(' looking ', ' ')
    trans = trans.replace(' please ', ' ')
    trans = trans.replace(' appreciate ', ' ')
    trans = trans.replace(' phone ', ' ')
    # trans = trans.replace(' call ', ' ')
    trans = trans.replace(' calling ', ' ')
    trans = trans.replace(' pop ', ' ')
    trans = trans.replace(' screen ', ' ')
    # trans = trans.replace(' today ', ' ')
    # trans = trans.replace(' automated ', ' ')
    # trans = trans.replace(' assistance ', ' ')
    # trans = trans.replace(' quick ', ' ')
    # trans = trans.replace(' questions ', ' ')
    # trans = trans.replace(' question ', ' ')
    trans = trans.replace('verified', ' ')
    trans = trans.replace('brief', ' ')
    trans = trans.replace('everything', ' ')
    trans = trans.replace('couple', ' ')
    trans = trans.replace('double check', ' ')
    trans = trans.replace(' rep ', ' ')
    trans = trans.replace(' first name ', ' ')
    trans = trans.replace(' last name ', ' ')
    trans = trans.replace(' first and last name ', ' ')
    # trans = trans.replace(' name ', ' ')
    trans = trans.replace('something', ' ')
    trans = trans.replace('anything', ' ')
    trans = trans.replace(' much ', ' ')
    # trans = trans.replace(' want ', ' ')
    # trans = trans.replace(' wanting ', ' ')
    trans = trans.replace('callback', ' ')
    trans = trans.replace(' say ', ' ')
    trans = trans.replace(' says ', ' ')
    trans = trans.replace(' saying ', ' ')
    # trans = trans.replace(' left ', ' ')
    # trans = trans.replace(' top ', ' ')
    trans = trans.replace(' recorded line ', ' ')
    # trans = trans.replace(' think ', ' ')
    # trans = trans.replace(' care ', ' ')
    trans = trans.replace(' sorry ', ' ')
    # trans = trans.replace(' good ', ' ')
    trans = trans.replace(' number ', ' ')
    trans = trans.replace(' speaking ', ' ')
    trans = trans.replace(' pleasure ', ' ')
    # trans = trans.replace(' happy ', ' ')
    # trans = trans.replace(' guess ', ' ')
    # trans = trans.replace(' hope ', ' ')
    # trans = trans.replace(' feels ', ' ')
    # trans = trans.replace(' feel ', ' ')
    trans = trans.replace(' glad ', ' ')
    trans = trans.replace(' hear ', ' ')
    # trans = trans.replace(' sense ', ' ')
    # trans = trans.replace(' corner ', ' ')
    trans = trans.replace(' called ', ' ')
    trans = trans.replace(' new year ', ' ')
    trans = trans.replace(' feel free ', ' ')
    # trans = trans.replace(' apologize ', ' ')
    trans = trans.replace(' forwarded ', ' ')
    trans = trans.replace(' said ', ' ')
    trans = trans.replace(' talked ', ' ')
    trans = trans.replace(' next available representative ', ' ')
    trans = trans.replace(' electronic channel support ', ' ')
    trans = trans.replace(' high net worth ', ' ')
    trans = trans.replace('recording', ' ')
    trans = trans.replace('message', ' ')
    trans = trans.replace(' presss pound ', ' ')
    trans = trans.replace(' lake city ', ' ')
    trans = trans.replace('representative', ' ')
    trans = trans.replace(' voicemail ', ' ')
    trans = trans.replace(' voice ', ' ')
    trans = trans.replace(' i r a ', ' ira ')
    trans = trans.replace(' 401 k\'s ', ' 401 ')
    trans = trans.replace(' irate ', ' ira ')
    # trans = trans.replace(' year ', ' ')
    # trans = trans.replace(' click ', ' ')
    # trans = trans.replace(' little ', ' ')
    trans = trans.replace(' done ', ' ')
    trans = trans.replace(' ask ', ' ')
    trans = trans.replace(' guy ', ' ')
    # trans = trans.replace(' press ', ' ')
    trans = trans.replace('queue set','cusip')
    trans = trans.replace('kevin','')
    trans = trans.replace('jennifer','')

    '''
    # changing verb form to capture negation:
    trans = trans.replace(' don\'t ', ' dont ')
    trans = trans.replace(' won\'t ', ' wont ')
    trans = trans.replace(' willn\'t ', ' willnot ')
    trans = trans.replace(' doesn\'t ', ' doesnot ')
    trans = trans.replace(' didn\'t ', ' didnot ')
    trans = trans.replace(' can\'t ', ' cannot ')
    trans = trans.replace(' couldn\'t ', ' couldnot ')
    '''

    # keeping some I will lose the 1 and 2 syllable words
    # commenting now, because we will now learn this using Gensim phraser class
    '''
    ##BUSINESS SPECIFIC
    trans = trans.replace('-', '')
    trans = trans.replace('mutual funds', 'mutualfund')
    trans = trans.replace('mutual fund', 'mutualfund')
    trans = trans.replace(' non managed ', ' nonmanaged ')
    '''

    # adding delimiters, makes more sense to do this in end as ... always comes with a space on both sides
    trans = trans.replace('...', '')

    # removing all one and two length words like on, at, of, etc which may add value but in a big corpus add lot of noise too
    trans = re.sub(r'[^\']\b\w{1,2}\b', '', trans)

    trans = trans.replace('  ', ' ')
    trans = trans.strip()
    trans = ' '.join(filter(None, trans.split(' ')))

    return trans


# class with file iterator
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                if len(line > 10):
                    [interaction_id, nice_interaction_id, customer_id, mid, call_data, phrase] = line.split('\t')
                    phrase = cleanTranscript(phrase)
                    yield [interaction_id, phrase]


# need to load dataframe as I need the interaction_id and not just a list of sentences, which will be a lot more faster using Gensim serialization
def parseTranscriptsOptimized():
    # throw error if output file already exists:
    if os.path.isfile('NLP_CONTEXT_TEST.csv'):
        exit('Output file already exists!')

    # initializations
    print(datetime.datetime.now())
    start_time = datetime.datetime.now()
    print('Starting the process')
    nlp = en_core_web_sm.load()

    # interaction list:
    interactions = []

    sentences = MySentences('rawData')  # a memory-friendly iterator
    for sen in sentences:
        print(sen)

    end_time = datetime.datetime.now()
    print('Completed process in ::' + str(end_time - start_time))


# word freq is based on the POS tag by spacy, so as not to convert words used in different contexts
# also if the w2vec doesn't return a synonym but wordnet synsets returns a word that is more in frequency as the current word key then replace the word
def buildWordTagDict(wordTagDict, spacyPhrase):
    for word in spacyPhrase:
        if word.lemma_ not in spacy.en.STOP_WORDS:
            baseWord, tag = wordPosTag(word)
            key = baseWord + '_' + tag
            if key in wordTagDict.keys():
                wordTagDict[key] += 1
            else:
                wordTagDict[key] = 1


# make sure we have converted each word in spacyPhrase in the form w.lemma_ + '_' + w.pos_
def wordListToFreqDict(spacyPhrases):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(spacyPhrases)
    freq = np.ravel(X.sum(axis=0))  # sum each columns to get total counts for each word
    vocab = [v[0] for v in sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))]
    fdist = dict(zip(vocab, freq))  # return same format as nltk
    return fdist


def wordPosTag(spacyObject):
    tag = ''
    if (spacyObject.pos_ == 'NOUN' or spacyObject.pos_ == 'PROPN'):
        tag = 'n'
    elif (spacyObject.pos_ == 'ADJ'):
        tag = 'a'
    elif (spacyObject.pos_ == 'ADV'):
        tag = 'r'
    elif (spacyObject.pos_ == 'VERB'):
        tag = 'v'
    else:
        tag = 'n'

    return tag


# need to load dataframe as I need the interaction_id and not just a list of sentences, which will be a lot more faster using Gensim serialization
def parseTranscripts():
    # loading to a dataframe
    #df = pd.read_table("feedback_ansi.txt", header=(0))
    df = pd.read_excel("clean_reviews_numbered.xlsx", header=(0))
    df = df.dropna()

    # throw error if output file already exists:
    if os.path.isfile('adhoc_chunked_output.csv'):
        exit('Older output file already exists!')
    else:
        with open('adhoc_chunked_output.csv', 'a') as f:
            f.write('doc_id,cleanPhrase,mergedPhrase,parsedNouns' + '\n')

    # cleaning transcripts after collecting lemmas using spacy lemmatizer
    df_clean_trans = rc.DataFrame(
        columns=['phrase','nouns'])

    print(datetime.datetime.now())
    start_time = datetime.datetime.now()
    print('Starting the process')
    rowNum = 0
    nlp = en_core_web_sm.load()
    spacyPhraseDict = {}
    nounDict = {}


    #phrased and cleaned trans dataframe:
    df = phraseBuilder(df)
    replaceDict = {}
    # load word2vec model
    model = word2vec.Word2Vec.load('../VEC_MODELS/fid_w2vec_pos_model')
    for index, row in df.iterrows():
        phrase = row['finalTrans']
	doc_id = row['RowNum']
        rowNum += 1
        print('Processing Row : %s' % rowNum)
        # spacy processing:
        spacyPhrase = nlp(phrase.decode('utf-8'))
        # buildWordTagDict(wordTagDict,spacyPhrase)
        posTaggedWordPhrase = ' '.join(
            string.replace(w.lemma_,'_','-') + '_' + wordPosTag(w) for w in spacyPhrase if w.lemma_ not in spacy.en.STOP_WORDS)
        nounTaggedWords = ' '.join(
            string.replace(w.lemma_, '_', '-') for w in spacyPhrase if (w.pos_ == 'NOUN' or w.pos_ == 'PROPN' or w.pos_ == 'NUM') and w.lemma_ not in spacy.en.STOP_WORDS)


	interaction_key = doc_id
        if interaction_key not in spacyPhraseDict:
            spacyPhraseDict[interaction_key] = posTaggedWordPhrase
            nounDict[interaction_key] = nounTaggedWords


    # use spacy tags to create word-tag freq dictionary for the entire corpus
    wordFreqDict = wordListToFreqDict(list(spacyPhraseDict.values()))

    # now we can use the word-tag dict to start replacing words and reducing our sample space
    print('Reducing word feature space so as to reduce variants of words')
    rowNum = 0
    newWordCounter = 0
    for index, row in df.iterrows():
        #print(spacyPhrase)
        #print(nounPhrase)
        rowNum += 1
	interaction_key = row['RowNum']
        spacyPhrase = spacyPhraseDict[interaction_key]
        nounPhrase = nounDict[interaction_key]
        print('Processing Row : %s' % rowNum)

        # calculate similarity and reduce feature space by reducing words using word2vec and wordnet intersection
        # building sentence one word at a time
        sen = []
        for taggedWord in spacyPhrase.split():
            wordKey = taggedWord  # already in the key format, can be used later
            baseWord, tag = taggedWord.split('_')
            # make sure we see each word only once
            if wordKey in wordFreqDict and baseWord != '' and tag != '':
                wordFreq = wordFreqDict[wordKey]
                # first check the replaceDict if this key is already present, replace if present and skip the loop iteration
                if wordKey in replaceDict:
                    replaceWord = replaceDict[wordKey]
                    w, t = replaceWord.split('_')
                    sen.append(w)
                    continue
                # continue if no replacement word found in the replaceDict
                # word that is not present in the model or is a unique word
                newWordCounter += 1
                if not wordnet.synsets(baseWord, tag) or baseWord not in model.wv.vocab:
                    sen.append(baseWord)
                    replaceDict[wordKey] = wordKey
                # getting best replacement from word2vec context, note that synset only gets the pos tag synonymns
                # hence even though we don't fetch word2vec context we should be finding the correct replacement word
                else:
                    synonyms = wordnet.synsets(baseWord, tag)
                    lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
                    synlist = [str(term) for term in lemmas]
                    model_similar_words = model.most_similar(positive=[baseWord], topn=10)
                    modellist = []
                    for item in model_similar_words:
                        item = str(item)
                        item = item.replace('\'', '')
                        item = item.replace('(', '')
                        item = item.replace(')', '')
                        term, score = item.split(',')
                        modellist.append(term)

                    # common words between synsets and w2vec may be more than one
                    commonWord = [word for word in modellist if word in synlist]
                    if (len(commonWord) != 0):
                        score = -1
                        for word in commonWord:
                            key = word + '_' + tag
                            score = wordFreq
                            if key in wordFreqDict:
                                score_ = wordFreqDict[key]
                            else:
                                score_ = score
                            if score_ > score:
                                score = score_
                                maxScoreWord = word

                        if (score > wordFreq):
                            sen.append(maxScoreWord)
                            # word substitution happening, cache it in the dict
                            replaceDict[wordKey] = maxScoreWord + '_' + tag
                            print('Replacing word :: ' + wordKey + ' with :: ' + maxScoreWord)
                        else:
                            # substitution not happening for the word, add original to the dict
                            sen.append(baseWord)
                            replaceDict[wordKey] = wordKey

                    else:
                        sen.append(baseWord)
                        if wordKey not in replaceDict:
                            replaceDict[wordKey] = wordKey

            else:
                sen.append(baseWord)



        # phrase = ' '.join(wordnet.synsets(w)[0].lemmas()[0].name() for w in phrase.split())
        phrase = ' '.join(w for w in sen)
        phrase = phrase.replace('-PRON-', '')
        phrase = re.sub(r'\b\w{1,2}\b', '', phrase)
        phrase = phrase.replace('\'', '')
        phrase = re.sub(r'(\s)\w+-+\s',' ',phrase)
        phrase = re.sub(r'(\s)-+\w+\s', ' ', phrase)
        phrase = phrase.replace(' - ', '')
        phrase = ' '.join(phrase.split())
        #print(phrase)

        nounPhrase = nounPhrase.replace('-PRON-', '')
        nounPhrase = re.sub(r'\b\w{1,2}\b', '', nounPhrase)
        nounPhrase = nounPhrase.replace('\'', '')
        nounPhrase = re.sub(r'(\s)\w+-+\s', ' ', nounPhrase)
        nounPhrase = re.sub(r'(\s)-+\w+\s', ' ', nounPhrase)
        nounPhrase = nounPhrase.replace(' - ', '')
        nounPhrase = ' '.join(nounPhrase.split())
        #print(nounPhrase)

        # collecting all clean lemmatized phrases by interaction id in a dataframe, this df will be used with gensim phraser
        df_clean_trans.append_row(index, {'doc_id' : interaction_key,
					  'phrase': phrase,
                                          'nouns': nounPhrase})

        # except:
        # print('Encountered issue with Spacy unicode token!!')

    # modify this function to directly read data from disk using LineSentences function of Gensim
    # function to build gensim phraser


    # convert to normal dataframe
    print('New words seen :: ' + str(newWordCounter))
    print('Num of words replaced : ' + str(len(replaceDict.keys())))

    # pickle the dictionary:
    # with open('../REPLACE_DICT_PICKLE', 'wb') as handle:
    #	pickle.dump(replaceDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    data_dict = df_clean_trans.to_dict(index=False)
    df_clean_trans = pd.DataFrame(data_dict, columns=df_clean_trans.columns, index=df_clean_trans.index)

    # feed tokenized transcript through the POS context extractor logic, which will create final file for LDA input
    chunkProcessedTrans(df_clean_trans)

    # write gensim phrase to file
    #nlpOutput(df_clean_trans)

    end_time = datetime.datetime.now()
    print('Completed process in ::' + str(end_time - start_time))
    exit(0)


def nlpOutput(df_phrased):
    for index, row in df_phrased.iterrows():
        gensimPhrase = row['phrase']
	if gensimPhrase != '' and gensimPhrase != None:
        	with open('processed_output.csv', 'a') as f:
            		fline = gensimPhrase
            		f.write(fline + '\n')


# this function merges words to fetch bigrams/trigrams in the corpus using Gensim Phraser module:
def phraseBuilder(df_sentences):
    # creating bigram Gensim Phrases:
    bigram = phrases.Phrases(delimiter='_')

    # convert df to list of sentences:
    sentences = []
    for index, row in df_sentences.iterrows():
	print(row['Feedback'])
	trans = row['Feedback']
	trans = trans.encode('ascii','ignore')
        sentence = [word for word in word_tokenize(cleanTranscript(trans)) if word.isalpha()]
        sentences.append(sentence)
        bigram.add_vocab([sentence])

    finalTrans = []

    # creating trigram Gensim Phrases
    trigram = phrases.Phrases(bigram[sentences], delimiter='_')
    for sen in trigram[bigram[sentences]]:
        trigramSen = ' '.join(w for w in sen)
        finalTrans.append(trigramSen)

    # assign interaction_id to these trigram Phrases that we will chunk to get useful context
    finalTransSeries = pd.Series(finalTrans)
    df_sentences = df_sentences.drop('Feedback', 1)
    df_sentences['finalTrans'] = finalTransSeries.values
    return df_sentences


def punctSpace(token):
    return token.is_punct or token.is_space


# required if using spacy POS tagger
def lineTrans():
    df = pd.read_table("outTrans.tsv", sep='\t', header=(0), nrows=10)
    for index, row in df.iterrows():
        phrase = row['transcript']
        phrase = cleanTranscript(phrase)
        yield unicode(phrase, 'utf-8')


def parseTrans():
    nlp = en_core_web_sm.load()
    for parsed_trans in nlp.pipe(lineTrans(), batch_size=100, n_threads=4):
        for sen in parsed_trans.sents:
            _sen = ' '.join([w.lemma_ for w in sen if w.lemma_ not in spacy.en.STOP_WORDS and not punctSpace(w)])
            _sen = _sen.replace('-PRON-', '')
            _sen = _sen.strip()
            yield (_sen)
            # if _sen != '':
            #    print(_sen)


if __name__ == "__main__":
    print('Starting Process')
    parseTranscripts()















