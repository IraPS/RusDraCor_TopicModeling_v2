import os
import re

# Setting the original TEI-files path
infolder = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/' \
           'github desktop/dracor-rusdracor/tei/'
# Setting the path for a new TEI-files' location
outfolder = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/' \
            'Programming/github desktop/RusDraCor/Ira_Scripts/' \
            'TopicModelling/rusdracor_topic_modeling/tei_without_proper_names/'


proper_names_stopwords = open('characters(proper)_names.txt', 'r', encoding='utf-8').read().split('\n')
proper_names_stopwords = sorted(proper_names_stopwords, key=len, reverse=True)

for file in os.listdir(infolder):
    if file.endswith('.xml'):
        filename = file.split('/')[-1]
        print(filename)
        tei_old = open(infolder + file, 'r', encoding='utf-8').read()
        tei_old_meta, tei_old_text = tei_old.split('<text>')[0], tei_old.split('<text>')[1]
        tei_new = tei_old_text
        for w in proper_names_stopwords:
            tei_new = re.sub(w, '', tei_new)
        tei_new = tei_old_meta + '<text>' + tei_new
        tei_new_file = open(outfolder + filename, 'w', encoding='utf-8')
        tei_new_file.write(tei_new)
        tei_new_file.close()