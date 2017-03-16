import re
import unicodedata

def replaceSmileys(tweet) :
    newtweet =  re.sub('[\U0001F602-\U0001F64F]', lambda m: unicodedata.name(m.group()), tweet)
    newtweet = re.sub(u"(\u2018|\u2019)", " ", newtweet)
    newtweet = newtweet.encode('ascii','ignore')
    return newtweet

def clean (tweet) :
    cleantweet = re.sub(r"http\S+", "", tweet).replace("@", "").replace("RT", "")
    cleantweet = replaceSmileys(cleantweet)
    return cleantweet