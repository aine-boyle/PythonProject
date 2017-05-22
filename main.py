import argparse
import csv
import gensim
import math
import nltk
import operator
import strcleaner

from flask import Flask, request, render_template, redirect, url_for

from nltk.corpus import wordnet as wordnet
from nltk.corpus import sentiwordnet as swn
from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()

import pattern.en
from pattern.web    import Twitter
from pattern.vector import KNN, count

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

def getMostPos(n, sorted_tweets) :
    pos_list = []
    if len(sorted_tweets) <= n :
        for tweet in sorted_tweets :
            pos_list.append(tweet[0])
    else:
        for tweet in range(0, n) :
            tweettuple = sorted_tweets[tweet]
            pos_list.append(tweettuple[0])

    return pos_list

def getMostNeg(n, sorted_tweets):
    neg_list = []
    if len(sorted_tweets) <= n :
        for tweet in sorted_tweets :
            neg_list.append(tweet[0])
    else:
        for tweet in range(0, n) :
            tweettuple = sorted_tweets[tweet]
            neg_list.append(tweettuple[0])
    return neg_list

def getTweetsAboveThr(a_t, **t) :
    list_at = []
    for key in t:
        if (float(t[key]) > a_t):
            tweet = key
            list_at.append(tweet)
    return list_at

def getTweetsBelowThr(b_t, **t):
    list_bt = []
    for key in t:
        if (float(t[key]) < b_t):
            tweet = key
            list_bt.append(tweet)
    return list_bt

def classify(score) :
    roundedscore = round(score,2)
    if(roundedscore <= 0.30) :
        return "very negative"
    elif(roundedscore < 0.50) :
        return "negative"
    elif(roundedscore >= 0.70) :
        return "very positive"
    elif(roundedscore > 0.55) :
        return "positive"
    else :
        return "neutral"

def getPercentages(list) :
    totalPositive = list[0] + list[1]
    neutral = list[2]
    totalNegative = list[3] + list[4]

    total = totalPositive + neutral + totalNegative

    PosPerc = 100 * (float(totalPositive) / float(total))
    NeutPerc = 100 * (float(neutral) / float(total))
    NegPerc = 100 * (float(totalNegative) / float(total))

    donutList = []
    donutList.insert(0, round(PosPerc, 2))
    donutList.insert(1, round(NeutPerc, 2))
    donutList.insert(2, round(NegPerc, 2))

    return donutList

app = Flask(__name__)

@app.route('/')
def submit():
    return render_template("index.html")

@app.route('/')
@app.route('/<k>/', methods=['POST'])
def main(k=None, n=None):
    k = request.form['search']
    print(k)

    input_word=k

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', help='Print top & bottom n tweets', default = 20, type = int)
    parser.add_argument('--a_t', help='Print tweets above a_t', default = 0.80, type = float)
    parser.add_argument('--b_t', help='Print tweets below b_t', default = 0.20, type = float)

    args = parser.parse_args()

    num = str(500)
    twitter, knn = Twitter(), KNN()

    keyword_list = []

    # Get model & keywords
    model = gensim.models.word2vec.Word2Vec.load('MyModel')

    t = {};

    if k in model.vocab:
        keywords = model.most_similar(k)
        for k in keywords :
            keyword_list.append(k[0])

    query = ''

    for k in keyword_list:
        query = query + ' OR ' + k
    query = query + ' OR ' + input_word

    for i in range(1, 3):
        for tweet in twitter.search(query, start=i, count=100):
            if tweet.language == "en":
                text = tweet.text.lower()
                _tweet = strcleaner.clean(text)
                _tweet = _tweet.replace("\n", " ")
                score = str(sentTweetWords_final_score(_tweet))
                t[_tweet] = score
                with open('tweets.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    if not tweet.location:
                        location = "none"
                    else:
                        location = tweet.location
                    classification = classify(float(score))
                    writer.writerow(
                        [tweet.id, tweet.date, tweet.author, location, keyword_list, _tweet, score, classification])

    return getSentimentDistribution(t, keyword_list, input_word, n)

@app.route('/')
@app.route('/<k>/', methods=['POST'])
def getSentimentDistribution(t, keyword_list, input_word, n = None):

    print(len(t))
    print("sentiment")
    str2num = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7":7, "8":8, "9":9, "10":10,
               "11": 11, "12": 12, "13": 13, "14": 14, "15": 15, "16": 16, "17": 17, "18": 18, "19": 19, "20": 20}

    n = 10

    sentiments = {}

    with open("tweets.csv", "rb") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if any (k in row[4] for k in keyword_list):
                sentiment = row[7]
                print(sentiment)
                if not sentiment in sentiments:
                    sentiments[sentiment] = 1
                else:
                   sentiments[sentiment] += 1

    print(sentiments.get("very positive"))
    print(sentiments.get("positive"))
    print(sentiments.get("neutral"))
    print(sentiments.get("negative"))
    print(sentiments.get("very negative"))

    bar_list = []
    bar_list.insert(0, sentiments.get("very positive"))
    bar_list.insert(1, sentiments.get("positive"))
    bar_list.insert(2, sentiments.get("neutral"))
    bar_list.insert(3, sentiments.get("negative"))
    bar_list.insert(4, sentiments.get("very negative"))

    donut_list = getPercentages(bar_list)

    print(len(t))

    sorted_tweets = sorted(t.items(), key=operator.itemgetter(1))
    neg_list = getMostNeg(n, sorted_tweets)

    reverse_tweets = sorted(t.items(), key=operator.itemgetter(1), reverse=True)
    pos_list = getMostPos(n, reverse_tweets)

    print("returning")
    return render_template("main.html", bar_list=bar_list, donut_list = donut_list, input_word = input_word, neg_list = neg_list, pos_list = pos_list, n=n)

if __name__ == '__main__':
    app.run()