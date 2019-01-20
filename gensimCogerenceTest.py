# Import required packages
import numpy as np
import os
import pandas as pd
import logging
import json
import pickle
import re
import random
random.seed(42)
import warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from numpy import array

# Import dataset
p_df = pd.read_csv('./Reviews.csv')
# Create sample of 10,000 reviews
p_df = p_df.sample(n = 10000)
# Convert to array
docs = array(p_df['Text'])
# Define function for tokenize and lemmatizing
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


def docs_preprocessor(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isdigit()] for doc in docs]

    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 3] for doc in docs]

    # Lemmatize all words in documents.
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    return docs

docs = list()
with open('./plays_data.pickle', 'rb') as f:
    plays_data = pickle.load(f)

stopwords_ru = open('./stopwords_and_others/stop_ru.txt', 'r', encoding='utf-8').read().split('\n')

# Splitting train texts into word-chunks
n = 0
k = 0
chunk_size = 500
min_chunk_size = 100
for play in plays_data:
    doc_text = re.sub('[\.,!\?\(\)\-:;—…́«»–]', '', plays_data[play]['nouns']).split()
    doc_text_wo_stopwords = str()
    for word in doc_text:
        if word not in stopwords_ru:
            doc_text_wo_stopwords += word + ' '
    doc_text_wo_stopwords = doc_text_wo_stopwords.split()
    for i in range(0, len(doc_text_wo_stopwords), chunk_size):
        one_chunk = ' '.join(doc_text_wo_stopwords[i:i + chunk_size])
        if len(one_chunk.split()) > min_chunk_size:
            docs.append(one_chunk)
        if min_chunk_size < len(one_chunk.split()) < chunk_size:
            k += 1
        if len(one_chunk.split()) < min_chunk_size:
            n += 1
print('Taking chunks of length {0} WORDS, length is after excluded stop-words'.format(chunk_size))
print('Chunks with length less than {0} (did not take) after excluding stop-words:'.format(min_chunk_size), n)
print('Chunks with length more than {0} and less than {1} (took) after excluding stop-words:'.format(min_chunk_size, chunk_size), k)
print('\nTopic modeling train text collection size: ', len(docs))


for text in range(len(docs)):
    tokens = docs[text].split()
    docs[text] = tokens

# docs = [[word for word in document if word not in stopwords_ru] for document in docs]

'''
#Create Biagram & Trigram Models
from gensim.models import Phrases
# Add bigrams and trigrams to docs,minimum count 10 means only that appear 10 times or more.
bigram = Phrases(docs, min_count=10)
trigram = Phrases(bigram[docs])

for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)
    for token in trigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)
'''

#Remove rare & common tokens
# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)
dictionary.filter_extremes(no_below=len(docs)/10, no_above=0.7)

dictionary.save("./gensim_lda_dict")
dictionary = Dictionary.load("./gensim_lda_dict")

#Create dictionary and corpus required for Topic Modeling
corpus = [dictionary.doc2bow(doc) for doc in docs]
print(dictionary)
num_unique_tokens = 0
# print('Number of unique tokens: %d' % num_unique_tokens)
print('Number of documents: %d' % len(corpus))
# print(corpus[:1])


'''
# Compute Coherence Score using c_v
coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score (Cv): ', coherence_lda)

# Compute Coherence Score using UMass
coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=dictionary, coherence="u_mass")
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score (UMass): ', coherence_lda)
'''


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        print('Calculating the model with ' + str(num_topics) + ' topics\n')
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# Make a index to word dictionary.
temp = dictionary[0]  # only to "load" the dictionary.
id2word = dictionary.id2token

'''
start = 1
limit = 100
step = 1

model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=docs,
                                                        start=start, limit=limit, step=step)

numOfTopics_Coherence = dict()
for i in range(len(coherence_values)):
    numOfTopics_Coherence[range(start, limit)[i]] = coherence_values[i]

print(numOfTopics_Coherence)

# Show graph
import matplotlib.pyplot as plt

x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

top5_models = sorted([(x, i) for (i, x) in enumerate(coherence_values)], reverse=True)[:5]
top5_models_num_of_topics = [i[1]+1 for i in top5_models]

print('Top num of topics', sorted(top5_models_num_of_topics))
'''

# Set parameters.
#for num_topics in top5_models_num_of_topics:
for num_topics in [100]:
    # chunksize = len(docs)
    passes = 20
    iterations = 400
    eval_every = 1
    lda_model = LdaModel(corpus=corpus, id2word=id2word, # chunksize=chunksize,
                           alpha='auto', eta='auto',
                           iterations=iterations, num_topics=num_topics,
                           passes=passes, eval_every=eval_every, random_state=42)
    # Print the Keyword
    # print(lda_model.print_topics())


    all_corpus_together = str()
    for play in plays_data:
        doc_text = re.sub('[\.,!\?\(\)\-:;—…́«»–]', '', plays_data[play]['nouns']).split()
        doc_text_wo_stopwords = str()
        for word in doc_text:
            if word not in stopwords_ru:
                doc_text_wo_stopwords += word + ' '
        all_corpus_together += doc_text_wo_stopwords
    all_corpus_together = [all_corpus_together.split()]

    dictionary = Dictionary(all_corpus_together)
    all_corpus_together = dictionary.doc2bow(all_corpus_together[0])
    print(lda_model.get_document_topics(all_corpus_together))

