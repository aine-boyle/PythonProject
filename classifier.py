import sys, getopt , csv, re

from nltk.corpus import wordnet as wordnet
from nltk.corpus import sentiwordnet as swn

import gensim
import nltk
import math
import pyodbc

from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()

def loadDictionary():

    d= dict()
    wpos = open('lexicon/words_positive','r').readlines()
    for i in range(len(wpos)):
        a = wpos[i].replace("\n","")
        d[a]=[1,0]
    wneg = open('lexicon/words_negative','r').readlines()
    for i in range(len(wneg)):
        a = wneg[i].replace("\n","")
        d[a]=[-1,0]
    wafinn = open('lexicon/AFINN-111.txt','r').readlines()
    for i in range(len(wafinn)):
        ln = wafinn[i].replace("\n","").split("\t")
        sc = float(ln[1])/5.0
        d[ln[0]]= [sc,0]

    return d

d=loadDictionary()

def getListScore(word,diction):
    word = word.lower()
    return [0,0] if (word not in diction) else diction[word]

def wordnet_pos_code(tag):
    if tag.startswith('NN'):
        return wordnet.NOUN
    elif tag.startswith('VB'):
        return wordnet.VERB
    elif tag.startswith('JJ'):
        return wordnet.ADJ
    elif tag.startswith('RB'):
        return wordnet.ADV
    else:
        return ''

def pos_tag(sentence):
    tagged_words = []
    tokens = nltk.word_tokenize(sentence)
    tag_tuples = nltk.tag._pos_tag(tokens, None, tagger)
    for (string, tag) in tag_tuples:
        token = {'word':string, 'pos':tag}
        tagged_words.append(token)
    return tagged_words

def word_sense_cdf(word, context, wn_pos):
    senses = wordnet.synsets(word, wn_pos)
    if len(senses) > 0:
        cfd = nltk.ConditionalFreqDist((sense, def_word)
                       for sense in senses
                       for def_word in sense.definition().split()
                       if def_word in context)
        best_sense = senses[0]
        for sense in senses:
            try:
                if cfd[sense].max() > cfd[best_sense].max():
                    best_sense = sense
            except:
                pass
        return best_sense
    else:
        return None

def word_sense_similarity(word, context, dummy = None):
    wordsynsets = wordnet.synsets(word)
    bestScore = 0.0
    result = None
    for synset in wordsynsets:
        for w in nltk.word_tokenize(context):
            score = 0.0
            for wsynset in wordnet.synsets(w):
                sim = wordnet.path_similarity(wsynset, synset)
                if(sim == None):
                    continue
                else:
                    score += sim
            if (score > bestScore):
                bestScore = score
                result = synset
    return result

def score_ngram(ngram, text, wsd = word_sense_cdf):
    sc=0.0
    imp=0.0
    ngram.append({'word': 'NULL', 'pos': 'NULL'})#in case it is an unigram
    bigram=ngram[0]["word"]+" "+ngram[1]["word"]
    unigram=ngram[0]["word"]
    if (sum(getListScore(bigram,d)) != 0):
        [sc,imp] = getListScore(bigram,d)
    elif (sum(getListScore(unigram,d)) != 0):
        [sc,imp] = getListScore(unigram,d)
    else:
        if 'punct' not in ngram[0] :
            sense = wsd(ngram[0]['word'], text, wordnet_pos_code(ngram[0]['pos']))
            if sense is not None:
                sent = swn.senti_synset(sense.name())
                if sent is not None:
                    pos=float(sent.pos_score() or 0)
                    neg=float(sent.neg_score() or 0)
                    if (pos>neg):
                        sc=pos
                    elif (pos<neg):
                        sc=-neg
    return [sc,imp]

def sentence_score(text, threshold = 0.75, wsd = word_sense_cdf):
    tagged_words = pos_tag(text)
    acumsum = 0.0
    imp_acumsum = 1.0
    for i in range(1,len(tagged_words)+1):
        mngr=tagged_words[i-1:i+1]
        [sc,imp]=score_ngram(mngr,text,wsd)
        acumsum=acumsum+sc
        imp_acumsum=imp_acumsum+imp
    return acumsum*imp_acumsum

def sentTweetWords_final_score(text):
    sentences = nltk.sent_tokenize(text)
    score = 0.0
    for sentence in sentences:
        score = sentence_score(sentence)
    return 1 / (1 + math.exp(-score))

def getMostPos(**t) :
    max = 0.00
    tweet = ""
    for key in t:
        if(float(t[key]) > max) :
            max = float(t[key])
            tweet = key
    return tweet

def getMostNeg(**t):
    min = 1.00
    tweet = ""
    for key in t:
        if (float(t[key]) < min):
            min = float(t[key])
            tweet = key
    return tweet

def getAveScore(**t):
    count = 0
    total = 0.00
    for key in t:
        count = count + 1
        score = float(t[key])
        total = total + score
    average = total / count
    return average

def main():
#Get Tweets from DB

    n = str(100)

    f = open('output.txt', 'w')

    t = {};

    con = pyodbc.connect(Trusted_Connection='yes', driver = '{SQL Server}',server = 'GANESHA\SQLEXPRESS' , database = '4YP')
    print("Connected")

    cur = con.cursor()

    sqlcommand = ("SELECT TOP " + n + " * FROM twitter_data")
    cur.execute(sqlcommand)

    for row in cur.fetchall():
        tweet = row[3]
        stripTweet = tweet.strip()
        tweetFinal = re.sub("[^a-zA-Z ]", "", stripTweet)
        score = str(sentTweetWords_final_score(tweetFinal))
        t[tweetFinal] = score
        output = (tweetFinal + " ||| " + str(score) + "\n")
        f.write(output)

    print("***************************************")

    print("Most Positive Tweet: ", getMostPos(**t))
    print("Most Negative Tweet: ", getMostNeg(**t))
    print("Average Sentiment: ", getAveScore(**t))

    f.close()
    con.close()

if __name__ == '__main__':
    main()