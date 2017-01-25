import gensim
import pyodbc
import pandas as pd

#Train model w/ Training Dataset
train = pd.read_csv("traintweets.csv", names = ['id', 'tweet', 'sentiment'])
print("Read labeled train reviews, ", (train["tweet"].size))

#Get Tweets from DB
con = pyodbc.connect(Trusted_Connection='yes', driver = '{SQL Server}',server = 'GANESHA\SQLEXPRESS' , database = '4YP')
print("Connected")

cur = con.cursor()

cur.execute("SELECT * FROM twitter_data")

tweets = []
for row in cur.fetchall():
    tweets.append(row[3])
    model = gensim.models.Word2Vec(row[3], min_count = 1)
    print("model: " , model)
con.close()

#sentences = [['first', 'sentence'], ['second', 'sentence']]
#model = gensim.models.Word2Vec(sentences, min_count=1)
#print("model: ", model)
#print("sentence: ", model['sentence'])