import urllib3
import json
import re
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

    lemmas = list()
    for l in m.analyze(text):
        if 'analysis' in l:
            try:
                lemmas.append(l['analysis'][0]['lex'])
            except:
                lemmas.append(l['text'])

print(len(lemmas))

counted = [[x, lemmas.count(x)] for x in set(lemmas)]
counted_sorted = sorted(counted, key=lambda x: x[1], reverse=True)

print(counted_sorted)
