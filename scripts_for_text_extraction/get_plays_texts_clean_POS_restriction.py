# -*- coding: utf8 -*-
import re
import codecs
import os
import glob
from lxml import etree
from os import walk
from pymystem3 import Mystem

m = Mystem()

# Removing previous results (so to not do this by hand as files are opened in "append" mode)
files = glob.glob('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/'
                  'Programming/github desktop/RusDraCor/Ira_Scripts/'
                  'TopicModelling/rusdracor_topic_modeling/speech_corpus_no_prop_char_names_ONLY_NOUNS/*/*.txt') +\
        glob.glob('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/'
                  'Programming/github desktop/RusDraCor/Ira_Scripts/'
                  'TopicModelling/rusdracor_topic_modeling/speech_corpus_no_prop_char_names_ONLY_NOUNS/*/*/*.txt')

for f in files:
    os.remove(f)

files = glob.glob('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/'
                  'Programming/github desktop/RusDraCor/Ira_Scripts/'
                  'TopicModelling/rusdracor_topic_modeling/speech_corpus_no_prop_char_names_ONLY_NOUNS/*/*/*.txt')

for f in files:
    os.remove(f)

# Setting input (TEI-xml files) and output (folders)
infolder = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/' \
           'Programming/github desktop/RusDraCor/Ira_Scripts/' \
           'TopicModelling/rusdracor_topic_modeling/tei_without_proper_names'
results = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/' \
          'Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/' \
          'rusdracor_topic_modeling/speech_corpus_no_prop_char_names_ONLY_NOUNS'
bycharacter = results + '/bycharacter/'
bysex = results + '/bysex/'

ns = {'tei': 'http://www.tei-c.org/ns/1.0','xml':'http://www.w3.org/XML/1998/namespace'}

genre_by_us = open('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/'
                   'Programming/github desktop/RusDraCor/Ira_Scripts/'
                   'TopicModelling/rusdracor_topic_modeling/Genre_by_us.txt', 'r', encoding='utf-8')
play_genre_dict = dict()
for play_line in genre_by_us:
    play, genre = play_line.split('	')[0], play_line.split('	')[1].split('\n')[0]
    play_genre_dict[play] = genre

print(play_genre_dict)


def get_genre(file):
    """Extracts the genre of the play if there is any in an xml-file.
    It's not used for TM genre (a customised dictionary/csv-file is used"""
    tei_file = open(file, 'r', encoding='utf-8')
    tei = tei_file.read()
    genre = 'None'
    if len(re.findall('<term type="genreTitle" subtype="comedy">', tei)) > 0:
        genre = 'Comedy'
    if len(re.findall('<term type="genreTitle" subtype="drama">', tei)) > 0:
        genre = 'Drama'
    if len(re.findall('<term type="genreTitle" subtype="tragedy">', tei)) > 0:
        genre = 'Tragedy'
    tei_file.close()
    return genre


def get_date(file):
    """Extracts the date-definite of the play"""
    tei_file = open(file, 'r', encoding='utf-8')
    tei = tei_file.read()
    try:
        date_print = int(re.search('<date type="print" when="(.*?)"', tei).group(1))
    except:
        try:
            date_print = int(re.search('<date type="print" .*?notAfter="(.*?)"', tei).group(1))
        except:
            date_print = None
    try:
        date_premiere = int(re.search('<date type="premiere" when="(.*?)"', tei).group(1))
    except:
        try:
            date_premiere = int(re.search('<date type="premiere" .*?notAfter="(.*?)">', tei).group(1))
        except:
            date_premiere = None
    try:
        date_written = int(re.search('<date type="written" when="(.*?)"', tei).group(1))
    except:
        try:
            date_written = int(re.search('<date type="written" .*?notAfter="(.*?)">', tei).group(1))
        except:
            date_written = None

    if date_print and date_premiere:
        date_definite = min(date_print, date_premiere)
    elif date_premiere:
        date_definite = date_premiere
    else:
        date_definite = date_print
    if date_written and date_definite:
        if date_definite - date_written > 10:
            date_definite = date_written
    elif date_written and not date_definite:
        date_definite = date_written

    tei_file.close()

    return date_definite


def getnameandauthor (root):
    """Extracts the author of the play"""
    name = (re.sub('\r|\n', '', root.find('.//tei:titleStmt/tei:title', ns).text))
    author = (re.sub('\r|\n|,', '', root.find('.//tei:titleStmt/tei:author', ns).text))
    return name, author


def checknotmult(speaker):
    """Checks if the <sp> (speech-act) is from several speakers"""
    if re.search(' #', speaker) != None:
        return False
    else:
        return True


def getsex (speaker, root):
    """Extracts a speaker's gender"""
    thisperson = root.find('.//tei:listPerson/tei:person[@xml:id="'+speaker+'"]', ns)
    try:
        if 'sex' in thisperson.attrib:
            return thisperson.attrib['sex']
        else:
            return 'undefined'
    except:
        thisperson = root.find('.//tei:listPerson/tei:personGrp[@xml:id="'+speaker+'"]', ns)
        if 'sex' in thisperson.attrib:
            return thisperson.attrib['sex']
        else:
            return 'undefined'


def getspeakerandsex(somesp, root):
    """Returns speaker name, gender"""
    speaker = 'unknown'
    sex = 'undefined'
    notmult = False
    if 'who' in somesp.attrib:
        speaker = somesp.attrib['who']
        notmult = checknotmult(speaker)
        speaker = re.sub('#', '', speaker)
        if notmult:
            sex = getsex(speaker, root)
    return speaker, sex, notmult


def parse_xml(path, filename):
    """Parses xml-files and writes the output"""
    fullpath = path + '/' + filename
    date_definite = get_date(fullpath)
    dates_ranges = [range(1700, 1750), range(1750, 1800), range(1800, 1850), range(1850, 1900), range(1900, 1950)]
    year_category = 0
    for n in range(len(dates_ranges)):
        if date_definite in dates_ranges[n]:
            year_category = str(dates_ranges[n][0]) + '_' + str(dates_ranges[n][-1])
    root = etree.parse(fullpath)
    name, author = getnameandauthor(root)
    name_for_genre = re.sub('й', 'й', name)
    genre = play_genre_dict[name_for_genre]
    print(name, genre)
    for sp in root.iterfind('.//tei:sp', ns):
        speaker, speakersex, notmult = getspeakerandsex(sp, root)
        if notmult:
            speechtext = ' '.join(sp.xpath('.//tei:p/text()|.//tei:l/text()',
                                           namespaces={'tei': 'http://www.tei-c.org/ns/1.0'}))
            lemmas_with_POS = list()
            for l in m.analyze(speechtext):
                if 'analysis' in l:
                    if len(l['analysis']) > 0:
                        lemmas_with_POS.append(l['analysis'][0])
            # speechtext = ' '.join([l['lex'] for l in lemmas_with_POS if re.match('^(A|ADV|S|V)(,|=)', l['gr'])]) + '\n'
            speechtext = '\n' + ' '.join([l['lex'] for l in lemmas_with_POS if re.match('^(S)(,|=)', l['gr'])]) + '\n'
            speechcode = '_'.join([speakersex, author, name, speaker])
            textbyplay = codecs.open(results+'/byplay/byplay/'+str(date_definite) +'_'+name+'.txt', 'a', 'utf-8')
            author = author.split(' ')[0]
            textbyauthor = codecs.open(results+'/byauthor/'+author+'_'+name+'.txt', 'a', 'utf-8')
            textbyyear_range = codecs.open(results+'/byyear_range/'+str(year_category)+'.txt', 'a', 'utf-8')
            textbygenre = codecs.open(results+'/bygenre/'+genre+'.txt', 'a', 'utf-8')
            textbycharacter = codecs.open(bycharacter+speechcode+'.txt', 'a', 'utf-8')
            textbygender = codecs.open(bysex+speakersex+'.txt', 'a', 'utf-8')
            textbyplay.write(speechtext)
            textbyauthor.write(speechtext)
            textbyyear_range.write(speechtext)
            textbyyear_range.write(speechtext)
            textbygenre.write(speechtext)
            textbycharacter.write(speechtext)
            textbygender.write(speechtext)
            textbyplay.close()
            textbyauthor.close()
            textbyyear_range.close()
            textbygenre.close()
            textbycharacter.close()
            textbygender.close()


for path, dirs, filenames in walk(infolder):
    for filename in filenames:
        if '.xml' in filename:
            print(filename)
            play = parse_xml(path,filename)

print('DONE PARSING')
