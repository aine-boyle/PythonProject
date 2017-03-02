from tweepy.streaming import StreamListener

import json
import time
import pyodbc
import re

from html.parser import HTMLParser

access_token = "265437614-dFepW5aHu0imZbBA3mbgGJkzrzTxqg2D8mZARRZY"
access_token_secret = "KKfLtjovox3OJc1jNhBMNj8NEpPWtHee8XOxXcJkN1bHV"
consumer_key = "PM0YukV9OBjEkre1YtOi7JV3S"
consumer_secret = "RmjJnte7kvfS3Y4xIakbiBvqgsqfsZP66w0xwuFRMIx76l7b8M"

class TwitterStreamer(StreamListener):

    con = pyodbc.connect(Trusted_Connection='yes', driver='{SQL Server}', server='GANESHA\SQLEXPRESS', database='4YP')
    print("Connected")

    def __init__(self, time_limit = 20):
        self.start_time = time.time()
        self.limit = time_limit
        super(TwitterStreamer, self).__init__()

    def on_data(self, data):
        cur = TwitterStreamer.con.cursor()
        if(time.time() - self.start_time) < self.limit:
            all_data = json.loads(data)
            if "text" in all_data and "user" in all_data and "id" in all_data and not "RT" in all_data["text"]:
                data = json.loads(HTMLParser().unescape(data))
                tweet = data['text']
                #print(all_data)
                cleaned_tweet = re.sub(r"http\S+", "", tweet)
                cleaned_tweet = cleaned_tweet.replace("@", "").replace("RT", "")
                username = all_data["user"]["screen_name"]
                identifier = all_data["id"]
                if data.get('place'):
                    location = data['place']['full_name']
                    sql = "INSERT INTO python_twitter_data(ID, TWEET, USERNAME, LOCATION) VALUES (?, ?, ?, ?)"
                    cur.execute(sql, (identifier, cleaned_tweet, username, location))
                else:
                    sql = "INSERT INTO python_twitter_data(ID, TWEET, USERNAME, LOCATION) VALUES (?, ?, ?, ?)"
                    cur.execute(sql, (identifier, cleaned_tweet, username, ""))
                    TwitterStreamer.con.commit()
            return True
        else:
            return False

    def on_error(self, status):
        print("error: " , status)