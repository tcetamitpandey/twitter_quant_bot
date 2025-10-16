# 🤖 Twitter/X Market Intelligence Collector

> A Python tool for collecting and analyzing real-time tweets related to the **Indian stock market**, generating simple **trading signals** using text analytics.

---

## 🚀 Features

✅ **Real-time Tweet Collection** — Stream tweets from **Twitter/X** in real-time  
📊 **Focused Hashtags** — `#nifty50`, `#sensex`, `#intraday`, `#banknifty`  
🧹 **Smart Data Cleaning** — Removes duplicates and irrelevant content  
📦 **Structured Storage** — Saves clean data in **Parquet** format  
🧠 **Signal Generation** — Uses **TF-IDF** to assign numerical signal scores  
📈 **Visualization** — Generate simple signal plots for quick insights  

---

## ⚙️ Quick Start (Complete Setup)

Follow these steps to get started quickly 👇
# 1️⃣ Create a virtual environment
python -m venv venv

2️⃣ Activate the environment
---
> Windows
venv\Scripts\activate
---
> macOS / Linux
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Provide your Twitter/X cookies
   - Export your cookies as a JSON file using a Chrome extension like:
    👉 EditThisCookie V3
   - Save that JSON file in your project folder, OR
   - Paste the JSON directly into the COOKIES_JSON variable in the script.

# 5️⃣ Run the Script
python twitter_data_collection.py
