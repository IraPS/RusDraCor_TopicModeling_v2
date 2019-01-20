# -*- coding: utf8 -*-
__author__ = 'IrinaPavlova'
import urllib3
import json
import re
import pickle
from pymystem3 import Mystem

m = Mystem()

urllib3.disable_warnings()
http = urllib3.PoolManager()

r = http.request('GET', 'https://dracor.org/api/corpora/rus')
plays_metadata_table = json.loads(r.data.decode('utf-8'))['dramas']

plays_data = dict()
for play in plays_metadata_table:
    id = play['id']
    author = play['author']['name']
    title = play['title']
    text = http.request('GET', 'https://dracor.org/api/corpora/rus/play/{}/spoken-text'.format(id)).data.decode('utf-8')
    text = re.sub('\n', ' ', text)

    lemmas_with_POS_no_names = list()
    for l in m.analyze(text):
        if 'analysis' in l:
            if len(l['analysis']) > 0:
                if 'имя' not in l['analysis'][0]['gr'] and 'отч' not in l['analysis'][0]['gr']:
                    lemmas_with_POS_no_names.append(l['analysis'][0])
                else:
                    print(l['analysis'])
    nouns = '\n' + ' '.join([l['lex'] for l in lemmas_with_POS_no_names if re.match('^(S)(,|=)', l['gr'])]) + '\n'

    print(id)
    #year = play['year']
    plays_data[id] = dict()
    plays_data[id]['author'] = author
    plays_data[id]['title'] = title
    plays_data[id]['text'] = text
    plays_data[id]['nouns'] = nouns
    #plays_data[id]['year'] = year

with open('./plays_data.pickle', 'wb') as f:
    pickle.dump(plays_data, f, pickle.HIGHEST_PROTOCOL)

