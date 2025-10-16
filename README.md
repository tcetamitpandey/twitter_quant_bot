# 🤖 Twitter/X Market Intelligence Collector

Collects tweets on Indian stock market hashtags, cleans the data, and generates simple trading signals
Features

Real-time tweet collection from Twitter/X


Focused on hashtags: #nifty50, #sensex, #intraday, #banknifty

Extracts username, timestamp, content, likes, retweets, replies, hashtags, mentions

Cleans and deduplicates data

Stores in Parquet format

Generates numerical signal scores using TF-IDF

Provides simple visualization of signals

🛠 How to Run

Create a virtual environment

python -m venv venv


Activate the virtual environment

Windows:

venv\Scripts\activate


Mac / Linux:

source venv/bin/activate


Install required packages

pip install -r requirements.txt


Provide Twitter/X cookies

Export your cookies as a JSON file and place it in the project folder [ you can use Chrome extension like ( edit this cookie V3 ) ]

Or paste the JSON into the COOKIES_JSON variable in the script

Run the script

python twitter_data_collection.py

