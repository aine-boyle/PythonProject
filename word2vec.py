import gensim
import pyodbc
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk.data
nltk.download()

#Train model w/ Training Dataset
train = pd.read_csv("traintweets.csv", names = ['id', 'tweet', 'sentiment'])
print("Read labeled train reviews, ", (train["tweet"].size))

#Clean Tweets
def tweet_to_wordlist(tweet) :
    tweet_text = BeautifulSoup(tweet).get_text()
    tweet_text = re.sub("[^a-zA-Z]","",tweet_text)
    words = tweet_text.lower().split()

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
for tweet in train["tweet"]:
    sentences += tweet_to_sentences(tweet, tokenizer)

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