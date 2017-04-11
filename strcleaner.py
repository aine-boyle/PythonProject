import re
import unicodedata

def replaceSmileys(tweet) :
    newtweet =  re.sub('[\U0001F601-\U0001F64F]', lambda m: unicodedata.name(m.group()), tweet)
    newtweet = re.sub(u"(\u2018|\u2019)", " ", newtweet)
    newtweet = newtweet.encode('ascii','ignore')
    newtweet = dontencode(newtweet)
    return newtweet

def clean (tweet) :
    cleantweet = re.sub(r"http\S+", "", tweet).replace("@", "").replace("RT", "")
    cleantweet = replaceSmileys(cleantweet)
    return cleantweet

def dontencode(tweet) :
    tweet = re.sub(r"DIGIT ZERO", "0", tweet)
    tweet = re.sub(r"DIGIT ONE", "1", tweet)
    tweet = re.sub(r"DIGIT TWO", "2", tweet)
    tweet = re.sub(r"DIGIT THREE", "3", tweet)
    tweet = re.sub(r"DIGIT FOUR", "4", tweet)
    tweet = re.sub(r"DIGIT FIVE", "5", tweet)
    tweet = re.sub(r"DIGIT SIX", "6", tweet)
    tweet = re.sub(r"DIGIT SEVEN", "7", tweet)
    tweet = re.sub(r"DIGIT EIGHT", "8", tweet)
    tweet = re.sub(r"DIGIT NINE", "9", tweet)
    tweet = re.sub(r"COLON", ":", tweet)
    tweet = re.sub(r"QUESTION MARK", "?", tweet)
    return tweet