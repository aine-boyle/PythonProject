import pyodbc

import pandas as pd
import gensim

train = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
print("Read labeled train reviews, ", (train["review"].size))

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
#model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
#model = gensim.models.Word2Vec.load("C:/Users/Aine/Documents/College/Y4/Final Year Project/annotated_train_tweets.csv")
#model['budget']
#for tweet in tweets:
#    print(tweet)
#sentences = [['first', 'sentence'], ['second', 'sentence']]
#model = gensim.models.Word2Vec(sentences, min_count=1)
#print("model: ", model)
#print("sentence: ", model['sentence'])