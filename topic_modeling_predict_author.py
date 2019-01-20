# -*- coding: utf8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random
import pickle


def display_topics(model, feature_names, no_top_words):
    """Displays all topics' top-words"""
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(", ".join([feature_names[i]
                         for i in topic.argsort()[:-no_top_words - 1:-1]]))


def display_one_topic(model, feature_names, no_top_words, topic_idx_needed):
    """Displays one particular topic's top-words"""
    for topic_idx, topic in enumerate(model.components_):
        if topic_idx == topic_idx_needed:
            print("Topic {}:".format(topic_idx))
            print('Topic top-words: ' + ", ".join([feature_names[i]
                                                   for i in topic.argsort()[:-no_top_words - 1:-1]]))


def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    """Establishes colour range for word-clouds"""
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 30)


def display_wordclouds(model, feature_names, no_top_words, n_topics):
    """Displays word-clouds for n topics' top-words"""
    top_words_weight_dicts = list()
    for topic_idx, topic in enumerate(model.components_):
        top_words_weight_dict = dict()
        for i in topic.argsort()[:-no_top_words - 1:-1]:
            top_words_weight_dict[feature_names[i]] = model.components_[topic_idx][i]
        top_words_weight_dicts.append(top_words_weight_dict)
    for t in range(n_topics):
        plt.figure()
        plt.imshow(WordCloud(background_color='white', color_func=grey_color_func).fit_words(top_words_weight_dicts[t]))
        plt.axis("off")
        plt.title("Topic #" + str(t))
        plt.show()

# Opening a stop-words list for Russian
stopwords_ru = open('./stopwords_and_others/stop_ru.txt', 'r', encoding='utf-8').read().split('\n')

# Determining train texts path (txt-files)
with open('./plays_data.pickle', 'rb') as f:
    plays_data = pickle.load(f)

train_documents_in_nouns = list()
test_documents_in_nouns = list()
train_documents_titles = list()
test_documents_titles = list()


# Splitting train texts into word-chunks
n = 0
k = 0
chunk_size = 500
min_chunk_size = 100
for play in plays_data:
    train_documents_titles.append(play)
    doc_text = re.sub('[\.,!\?\(\)\-:;—…́«»–]', '', plays_data[play]['nouns']).split()
    for i in range(0, len(doc_text), chunk_size):
        one_chunk = ' '.join(doc_text[i:i + chunk_size])
        if len(one_chunk.split()) > min_chunk_size:
            train_documents_in_nouns.append(one_chunk)
        if min_chunk_size < len(one_chunk.split()) < chunk_size:
            k += 1
        if len(one_chunk.split()) < min_chunk_size:
            n += 1
print('Taking chunks of length {0} WORDS'.format(chunk_size))
print('Chunks with length less than {0} (did not take):'.format(min_chunk_size), n)
print('Chunks with length more than {0} and less than {1} (took):'.format(min_chunk_size, chunk_size), k)


# Creating a play-author dictionary
play_author_dict = dict()

for play in plays_data:
    id = play
    author = plays_data[id]['author']
    test_documents_titles.append(id)
    play_author_dict[id] = author
    doc_text = plays_data[id]['nouns']
    test_documents_in_nouns.append(doc_text)


# Reporting statistics on the model
print('\nTopic modeling train text collection size: ', len(train_documents_in_nouns))
print('Median length of train collection\'s documents: ', np.median([len(d.split()) for d in train_documents_in_nouns]))
print('Mean length of train collection\'s documents: ', np.mean([len(d.split()) for d in train_documents_in_nouns]))
print('Minimum length of train collection\'s documents: ', np.min([len(d.split()) for d in train_documents_in_nouns]))
print('Maximum length of train collection\'s documents: ', np.max([len(d.split()) for d in train_documents_in_nouns]))


def create_doc_topic_dict_for_plays(doc_topic_dist):
    """Creates a doc-topic dictionary with topics' probabilities and a dictionary with 3-top topics per document"""
    doc_3toptopic_dict = dict()
    doc_topicsprobs_dict = dict()
    for play in range(len(doc_topic_dist)):
        play_title = test_documents_titles[play]
        play_topic_dist = (doc_topic_dist[play].tolist()[0])
        # creating a list with probs per topic (in 100-notation)
        play_topic_dist = [round(100*float('{:f}'.format(item)), 3) for item in play_topic_dist]
        doc_topicsprobs_dict[play_title] = play_topic_dist
        play_top3_topics = reversed(doc_topic_dist.argsort()[play].tolist()[0][-3::])
        doc_3toptopic_dict[play_title] = play_top3_topics
    return doc_3toptopic_dict, doc_topicsprobs_dict


def print_results(topic_topdocs_dict, lda, tf_feature_names, no_top_words, doc_topic_dict, doc_topicsprobs_dict):
    """Prints the topics (top-words) of a model"""
    print('\nDOCUMENTS PER TOPIC')
    for topic in topic_topdocs_dict:
        display_one_topic(lda, tf_feature_names, no_top_words, int(topic))
        print('Topic plays: ' + ', '.join(topic_topdocs_dict[topic]), '\n')
    print('\n\nTOPICS PER DOCUMENT')
    for play in sorted(list(doc_topic_dict)):
        print(play)
        for topic in doc_topic_dict[play]:
            print(doc_topicsprobs_dict[play])
            display_one_topic(lda, tf_feature_names, no_top_words, int(topic))


def write_topic_author_dist(doc_topic_dict, doc_topicsprobs_dict):
    """Calculates and writes mean topics' probabilities per author into a csv-file (for 13 particular authors)"""
    author_prob_dict = dict()
    for play in sorted(list(doc_topic_dict)):
        author = play_author_dict[play]
        author = re.sub('й', 'й', author)
        if author in ['Сумароков, Александр Петрович', 'Крылов, Иван Андреевич', 'Шаховской, Александр Александрович',
                      'Пушкин, Александр Сергеевич', 'Островский, Александр Николаевич',
                      'Гоголь, Николай Васильевич', 'Сухово-Кобылин, Александр Васильевич',
                      'Тургенев, Иван Сергеевич', 'Чехов, Антон Павлович', 'Булгаков, Михаил Афанасьевич',
                      'Толстой, Лев Николаевич', 'Толстой, Алексей Константинович', 'Фонвизин, Денис Иванович',
                      'Блок, Александр Александрович', 'Гумилёв, Николай Степанович']:
            if author not in author_prob_dict:
                author_prob_dict[author] = list()
                author_prob_dict[author].append(doc_topicsprobs_dict[play])
            else:
                author_prob_dict[author].append(doc_topicsprobs_dict[play])

    print(author_prob_dict)

    authors_mean_probs = dict()
    for author in author_prob_dict:
        num_of_plays_per_year = len(author_prob_dict[author])
        author_sum_probs = [sum(i) for i in zip(*author_prob_dict[author])]
        author_mean_probs = [str(round(i/num_of_plays_per_year, 2)) for i in author_sum_probs]
        authors_mean_probs[author] = author_mean_probs

    author_probs_for_R = open('./graphs_5_topics/by_author/author_probs_for_R.csv', 'w',
                            encoding='utf-8')
    author_probs_for_R.write('Author;Probability;Topic\n')
    for author in authors_mean_probs:
        probs = authors_mean_probs[author]
        for p in range(len(probs)):
            author_probs_for_R.write(author+';'+probs[p]+';Topic'+str(p)+'\n')
    author_probs_for_R.close()


def run_TM(n_topics, doprint, doreturn):
    """Performs Topic Modeling, present topics and return/print/write in a file model's application results"""
    n_topics = n_topics
    no_top_words = 40

    tf_vectorizer = CountVectorizer(max_df=0.7,
                                    min_df=0.2,
                                    stop_words=stopwords_ru,
                                    max_features=500)
    tf = tf_vectorizer.fit_transform(train_documents_in_nouns)
    tf_feature_names = tf_vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100, random_state=42)
    lda_doc_topic = lda.fit_transform(tf)

    tf1 = tf_vectorizer.transform(test_documents_in_nouns)
    doc_topic_dist_unnormalized = np.matrix(lda.transform(tf1))

    # Normalising the distribution
    doc_topic_dist = doc_topic_dist_unnormalized/doc_topic_dist_unnormalized.sum(axis=1)
    topic_topdocs_dict = dict()

    doc_topic_dict = create_doc_topic_dict_for_plays(doc_topic_dist)[0]
    doc_topicsprobs_dict = create_doc_topic_dict_for_plays(doc_topic_dist)[1]


    for play in range(len(doc_topic_dist)):
        top_topic = str(doc_topic_dist.argmax(axis=1)[play].tolist()[0][0])
        if top_topic not in topic_topdocs_dict:
            topic_topdocs_dict[top_topic] = list()
            topic_topdocs_dict[top_topic].append(test_documents_titles[play])
        else:
            topic_topdocs_dict[top_topic].append(test_documents_titles[play])

    # Writing down the author-topic probabilities to a csv-file
    write_topic_author_dist(doc_topic_dict, doc_topicsprobs_dict)

    # Printing topics' 40 top-words, printing topics' distribution in all test documents,
    # displaying word-clouds for 100 topics' top-words if needed
    if doprint:
        print('LDA doc-topic shape:', lda_doc_topic.shape)
        print('\nTOPICS\nLDA top terms:')
        display_topics(lda, tf_feature_names, no_top_words)
        print('\n\n')
        print_results(topic_topdocs_dict, lda, tf_feature_names, no_top_words, doc_topic_dict, doc_topicsprobs_dict)
        # display_wordclouds(lda, tf_feature_names, 100, n_topics)

    # Returning test-documents topics' probabilities for classification task
    if doreturn:
        return doc_topicsprobs_dict

    print('The TM is finished, the model is applied to the authors data, '
          'you can find topics\'s per authors distribution in "author_probs_for_R.csv"')

# Running topic modeling task to build a model with 5 topics
run_TM(5, 1, 0)
