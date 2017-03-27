import argparse
import csv
import gensim
import math
import nltk
import operator
import strcleaner

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
    print("Most Positive ", n, " Tweets: ")
    if len(sorted_tweets) <= n :
        for tweet in sorted_tweets :
            print(tweet)
    else:
        for tweet in range(0, n) :
            print(sorted_tweets[tweet])

def getMostNeg(n, sorted_tweets):
    print("Most Negative ", n, " Tweets: ")
    if len(sorted_tweets) <= n :
        for tweet in sorted_tweets :
            print(tweet)
    else:
        for tweet in range(0, n) :
            print(sorted_tweets[tweet])

def getTweetsAboveThr(a_t, **t) :
    print("Tweets above ", a_t)
    for key in t:
        if (float(t[key]) > a_t):
            tweet = key
            print(tweet, " : ", t[key])
    print("******************************")

def getTweetsBelowThr(b_t, **t):
    print("Tweets below ", b_t)
    for key in t:
        if (float(t[key]) < b_t):
            tweet = key
            print(tweet, " : ", t[key])
    print("******************************")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', help='Keyword', type=str, nargs='*', required = True)
    parser.add_argument('--n', help='Print top & bottom n tweets', default = 20, type = int)
    parser.add_argument('--a_t', help='Print tweets above a_t', default = 0.80, type = float)
    parser.add_argument('--b_t', help='Print tweets below b_t', default = 0.20, type = float)

    args = parser.parse_args()

    num = str(500)
    twitter, knn = Twitter(), KNN()

    t = {};
    keyword_list = []

    # Get model & keywords
    model = gensim.models.word2vec.Word2Vec.load('MyModel')
    for k in args.k :
        keyword_list.append(k)
        if k in model.vocab:
            keywords = model.most_similar(k)
            for k in keywords :
                keyword_list.append(k[0])

    for i in range(1, 3):
        for tweet in twitter.search(k[0], start=i, count=100):
            if tweet.language == "en" :
                text = tweet.text.lower()
                _tweet = strcleaner.clean(text)
                p = k[0] in tweet and 'positive' or 'negative'
                v = pattern.en.tag(_tweet)
                v = [word for word, pos in v if pos == 'JJ']  # JJ = adjective
                v = count(v)  # {'sweet': 1}
                if v:
                    knn.train(v, type=p)
                _tweet = _tweet.replace("\n", " ")
                score = str(sentTweetWords_final_score(_tweet))
                t[_tweet] = score
                with open('tweets.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    if not tweet.location :
                        location = "none"
                    else :
                        location = tweet.location
                    classification = classify(float(score))
                    writer.writerow([tweet.id, tweet.date, tweet.author, location, keyword_list, _tweet, score, classification])

    # Print tweets above a_t
    getTweetsAboveThr(args.a_t, **t)

    # Print tweets below b_t
    getTweetsBelowThr(args.b_t, **t)

    # Print top n & bottom n tweets
    sorted_tweets = sorted(t.items(), key=operator.itemgetter(1))
    getMostNeg(args.n, sorted_tweets)

    reverse_tweets = sorted(t.items(), key=operator.itemgetter(1), reverse=True)
    getMostPos(args.n, reverse_tweets)

if __name__ == '__main__':
    main()