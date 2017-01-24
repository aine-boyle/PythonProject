import pyodbc

import gensim

con = pyodbc.connect(Trusted_Connection='yes', driver = '{SQL Server}',server = 'GANESHA\SQLEXPRESS' , database = '4YP')
print("Connected")

cur = con.cursor()

# Use all the SQL you like
cur.execute("SELECT * FROM twitter_data")

# print all the first cell of all the rows
tweets = []
for row in cur.fetchall():
    tweets.append(row[3])
    model = gensim.models.Word2Vec(row[3], min_count = 1)
    print("model: " , model)
con.close()
#for tweet in tweets:
#    print(tweet)
#sentences = [['first', 'sentence'], ['second', 'sentence']]
#model = gensim.models.Word2Vec(sentences, min_count=1)
#print("model: ", model)
#print("sentence: ", model['sentence'])