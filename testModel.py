from gensim.models import word2vec

model = word2vec.Word2Vec.load('MyModel')

#print(model.doesnt_match("man woman child kitchen".split()))
print("Words Similar to 'Happy'" , model.most_similar("happy"))
print("Words Similar to 'Sad'" , model.most_similar("sad"))