import os
import re
from nltk import tokenize

# Setting the path to the corpus of txt-files (not lemmatised)
plays_texts_path = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/' \
                   'github desktop/RusDraCor/Ira_Scripts/TopicModelling/' \
                   'speech_corpus/byplay/byplay_not_lemmatised/'
# Setting the output file
all_capitalised_words_file = open('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/'
                             'Programming/github desktop/RusDraCor/Ira_Scripts/'
                             'TopicModelling/all_capitalised_words.txt', 'w', encoding='utf-8')
all_capitalised_words_list = list()

for file in os.listdir(plays_texts_path):
    if file.endswith('.txt'):
        print(file)
        text = open(plays_texts_path + file, 'r', encoding='utf-8').read()
        text = re.sub('\.\.\.', '.', text)
        text = re.sub('\?', '? ', text)
        text = re.sub('!', '! ', text)
        text = re.sub('\.', '. ', text)
        text = re.sub('\n', '. ', text)
        text = re.sub('\.\.\.', '.', text)
        text = re.sub('  ', ' ', text)
        text_by_sentences = tokenize.sent_tokenize(text)
        text_by_sentences = [re.sub('\n', '', sent) for sent in text_by_sentences]
        for sent in text_by_sentences:
            sent = sent.split()
            sent = ' '.join(sent[1:-1])
            capitalised_words_in_sent = re.findall('([А-Я][а-я]+)', sent)
            for word in capitalised_words_in_sent:
                if word not in all_capitalised_words_list:
                    all_capitalised_words_list.append(word)
                    all_capitalised_words_file.write(word + '\n')

all_capitalised_words_file.close()

