#import numpy as np

import logging
import nltk.data
import pandas as pd
import re

nltk.download()

from bs4 import BeautifulSoup
from gensim.models import word2vec
from nltk.corpus import stopwords

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
                   level=logging.ERROR)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#Set Parameters
num_features = 300      #Word vector dimensionality
min_word_count = 10
num_workers = 4         #Number of threads to run in parallel
context = 10
downsampling = 1e-3     #Downsample setting for frequent words

#Train model w/ Training Dataset
print("Reading CSV...")
train = pd.read_csv("Sentiment Analysis Dataset.csv", names = ['ItemID', 'Sentiment', 'SentimentSource', 'SentimentText'])
print("Read labeled train tweets, ", (train["SentimentText"].size))
#train = pd.read_csv( "labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

#Clean Tweets
def tweet_to_wordlist(tweet, remove_stopwords=False) :
    #remove urls from tweets
    tweet = re.sub(r"http\S+", "", tweet)
    tweet_text = BeautifulSoup(tweet, "lxml").get_text()
    words = tweet_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)

#Split Tweet into Parsed Sentences
def tweet_to_sentences (tweet, tokenizer):
    raw_tweets = tokenizer.tokenize(tweet.strip())
    sentences = []
    for raw_tweet in raw_tweets:
        if len(raw_tweet) > 0:
            sentences.append(tweet_to_wordlist(raw_tweet))
    return sentences

sentences = []

print("Parsing sentences from training set")
for tweet in train["SentimentText"]:
    sentences += tweet_to_sentences(tweet, tokenizer)

print(len(sentences))
for x in range (0, 10):
    print(sentences[x])

print("Training Model... ")
model = word2vec.Word2Vec(sentences, workers = num_workers, \
                          size= num_features, min_count=min_word_count, \
                          window= context, sample= downsampling)

model.init_sims(replace=True)

model_name = "MyModel"
model.save(model_name)
#model.save(model_name, "C:/Users/Aine/PycharmProjects/FinalYearProjectFolder/FinalYearProject/models/")

print(model.doesnt_match("man woman child kitchen".split()))
print(model.most_similar("sad"))