from pattern.web    import Twitter
from pattern.vector import KNN, count
import strcleaner
import tweepy

twitter = Twitter()
knn = KNN()

def classifyTweets(tweetList) :
    model = knn.load("C:/Users/Aine/PycharmProjects/FinalYearProjectFolder/FinalYearProject/models/knn")
    for tweet in tweetList:
        classification = model.classify(tweet)
        print(tweet, " : " , classification)
        print(model.classify(tweet, discrete=False))
        t = model.classify(tweet, discrete=False)
        for _class, _probability in t.iteritems():
            if _class == 'positive' and _probability > 0.9:
                print("very positive")
            elif _class == 'positive' and _probability > 0.7:
                print("positive")
            elif _class == 'negative' and _probability > 0.9:
                print("very negative")
            elif _class == "negative" and _probability > 0.7:
                print("negative")


def main():
    tweetList = []
    for i in range(1, 10):
        for tweet in twitter.search('#win OR #fail', start=i, count=10):
            s = tweet.text.lower()
            _tweet = strcleaner.clean(s)
           # print(_tweet)
            tweetList.append(_tweet)

    classifyTweets(tweetList)

if __name__ == '__main__':
    main()