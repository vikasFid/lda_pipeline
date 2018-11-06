# LDA Pipeline for Speech Processing

This project is a POC I did for building topic modeling pipelines to analyze automated speech recognition system output.
Disclaimer : This is not the production version and is only for reference and learning purposes.
This pipeline includes three components:
* Text pre processing : Implements lemmatization, Gensim phrasing, NP chunking and Synonym substution using word2vec and wordnet.
* LDA Modeling : Gibbs Sampling for creating LDA model. Also creates visualization using LDAVis package(by cpsievert)
* Topic selection and measurement : Command line interface for selecting topic and documents list based on the topic mixture and 
selected lambda value from the LDAVis visualizaton.


### Prerequisites
* Python dependencies: Spacy, NLTK, gensim, raccoon
* R packages : LDAVis, tsne, topicmodels
* Word2Vec model : Use custom or Google's GoogleNews vector

### Execution
* Input : Tab separated text file with a unique identifier.


## Built With

* PyCharm
* RStudio

## Authors

* **Vikash Kumar**

## Acknowledgments
* Gensim package creator : Radim Rehurek
* Latent Ditichlet Allocation : David Blei
