import pyodbc
import re
import numpy as np

from gensim.models import word2vec
import pandas as pd
from bs4 import BeautifulSoup
import nltk.data
nltk.download()
from nltk.corpus import stopwords
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
                    level=logging.INFO)

#Set Parameters
num_features = 300      #Word vector dimensionality
min_word_count = 10
num_workers = 4         #Number of threads to run in parallel
context = 10
downsampling = 1e-3     #Downsample setting for frequent words

#Train model w/ Training Dataset
print("Reading CSV...")
#train = pd.read_csv("traintweets.csv", names = ['id', 'tweet', 'sentiment'])
train = pd.read_csv( "labeledTrainData.tsv", header=0,
 delimiter="\t", quoting=3 )
#train = pd.read_csv("labeledTrainData.tsv", names = ['id', 'sentiment', 'review'])
print("Read labeled train reviews, ", (train["review"].size))

#Clean Tweets
def tweet_to_wordlist(tweet, remove_stopwords=False) :
    tweet_text = BeautifulSoup(tweet, "lxml").get_text()
    #tweet_text = re.sub("[^a-zA-Z]","",tweet_text)
    words = tweet_text.lower().split()

    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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
for tweet in train["review"]:
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

print(model.doesnt_match("man woman child kitchen".split()))
    #Get Tweets from DB
#con = pyodbc.connect(Trusted_Connection='yes', driver = '{SQL Server}',server = 'GANESHA\SQLEXPRESS' , database = '4YP')
#print("Connected")

#cur = con.cursor()

#cur.execute("SELECT * FROM twitter_data")

#tweets = []
#for row in cur.fetchall():
#    tweets.append(row[3])
#    model = gensim.models.Word2Vec(row[3], min_count = 1)
#    print("model: " , model)
#con.close()

#sentences = [['first', 'sentence'], ['second', 'sentence']]
#model = gensim.models.Word2Vec(sentences, min_count=1)
#print("model: ", model)
#print("sentence: ", model['sentence'])