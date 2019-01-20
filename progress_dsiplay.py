import pickle
import re
from gensim.corpora.dictionary import Dictionary

with open('./plays_data.pickle', 'rb') as f:
    plays_data = pickle.load(f)

stopwords_ru = open('./stopwords_and_others/stop_ru.txt', 'r', encoding='utf-8').read().split('\n')

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
print(dictionary)
all_corpus_together = dictionary.doc2bow(all_corpus_together[0])