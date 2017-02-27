~ Final Year Project ~

I plan to develop a system that will perform sentiment analysis on Tweets in order to determine if they are positive, negative, or neutral

More Information: https://1drv.ms/w/s!Anpl86rlxlJqmXzjqvViLrDoydit

How to run the code:
	
	Run classifier.py with the following arguments:
		-- k 				// String -> Keyword to seach for, this arg is required!
		-- n 				// Int -> to print top n positive / bottom n negative tweets (not required, default = 20)
		-- a_t				// Float -> get tweets above given threshold (not required, default = 0.8)
		-- b_t				// Float -> get tweets below given threshold (not required, default = 0.2)
		
Printed:
	Tweets above a_t
	****************
	Tweets below b_t
	****************
	Most negative n tweets
	Most positive n tweets