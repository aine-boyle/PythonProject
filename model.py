import csv

import pattern.en
from pattern.web    import Twitter
from pattern.vector import KNN, count

twitter, knn = Twitter(), KNN()

with open("Sentiment Analysis Dataset.csv", "rb") as csvfile:
    reader = csv.reader(csvfile, delimiter='\t')
    for row in reader:
        if row:
            text = row[3]
            if(row[1] == "1"):
                p = "positive"
            elif row[1] == "0":
                p = "negative"
            v = pattern.en.tag(text)
            v = [word for word, pos in v if pos == 'JJ']
            v = count(v)
            if v:
                knn.train(v, type=p)

knn.save("models/knn", final=False)
print("Model Trained & Saved")