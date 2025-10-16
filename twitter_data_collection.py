"""
Twitter bot by Amit to Collect tweets data of specific hastags and later convert them into signals  
"""

import time
import random
import re
import pickle
import os
import sys
import json
import unicodedata
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from sklearn.feature_extraction.text import TfidfVectorizer

# =================== CONFIG ===================
HASHTAGS = ["#nifty50", "#sensex", "#intraday", "#banknifty"]
COOKIES_FILE_PKL = "twitter_cookies.pkl"   # saved cookies (pickle)
COOKIES_JSON_FILE = "twitter_cookies.json" # optional JSON file (user can drop exported cookies here too)
DATA_FILE = "tweets_collected.parquet"
CLEANED_FILE = "tweets_cleaned.parquet"

# If you want youo can paste cookie JSON directly into the script, replace the empty list below
# with the JSON string or Python list object you exported from your browser.
# Example: COOKIES_JSON = '[{"name":"auth_token","value":"...","domain":".twitter.com", ...}, ...]'
COOKIES_JSON = ""  # paste JSON string here or leave empty to load COOKIES_JSON_FILE

MAX_TWEETS_PER_HASHTAG = 20
MIN_TWEETS_PER_HASHTAG = 10
HEADLESS = False
SCROLL_RETRY_LIMIT = 6
ERROR_PAUSE_SECONDS = 600  # 10 min for Something went wrong Error

# =================== SELENIUM INIT ===================
def init_driver(headless=HEADLESS):
    options = Options()
    options.add_argument("--start-maximized")
    # try to reduce detection flags
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    if headless:
        options.add_argument("--headless=new")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(60)
    return driver

# =================== COOKIE UTILITIES ===================
def normalize_cookie_for_selenium(cookie):
    """
    Return a dict suitable for driver.add_cookie.
    Ensures required keys exist and types are correct.
    """
    c = {}
    # required keys for add_cookie: name, value
    c["name"] = cookie.get("name") or cookie.get("Name") or cookie.get("key")
    c["value"] = cookie.get("value") or cookie.get("Value")
    if not c["name"] or c["value"] is None:
        return None
    # domain: ensure leading dot acceptable
    domain = cookie.get("domain") or cookie.get("Domain")
    if domain:
        c["domain"] = domain
    else:
        # default to twitter domain if missing
        c["domain"] = ".twitter.com"

    # path
    path = cookie.get("path") or cookie.get("Path") or "/"
    c["path"] = path

    # secure / httpOnly
    if cookie.get("secure") is not None:
        c["secure"] = bool(cookie.get("secure"))
    if cookie.get("httpOnly") is not None:
        c["httpOnly"] = bool(cookie.get("httpOnly"))

    # expiry -> integer seconds since epoch
    expiry = cookie.get("expiry") or cookie.get("expires") or cookie.get("Expires")
    if expiry:
        try:
            # Some exporters give ISO strings; try parse int first
            if isinstance(expiry, str) and expiry.isdigit():
                c["expiry"] = int(expiry)
            else:
                c["expiry"] = int(float(expiry))
        except Exception:
            # ignore invalid expiry
            pass

    # SameSite (optional)
    if cookie.get("sameSite"):
        c["sameSite"] = cookie.get("sameSite")
    return c

def load_cookies_from_json_string(driver, json_string, save_pickle=True):
    """
    Inject cookies given a JSON string or Python list.
    Returns True if any cookie was added.
    """
    try:
        if isinstance(json_string, str):
            cookies_list = json.loads(json_string)
        elif isinstance(json_string, list):
            cookies_list = json_string
        else:
            cookies_list = list(json_string)
    except Exception as e:
        print(f"[ERROR] Could not parse cookie JSON: {e}")
        return False

    if not isinstance(cookies_list, list):
        print("[ERROR] Cookie JSON must be a list of cookie objects.")
        return False

    # load a base page first (domain needs to match for add_cookie)
    driver.get("https://twitter.com")
    time.sleep(2)

    added = 0
    for cookie in cookies_list:
        normalized = normalize_cookie_for_selenium(cookie)
        if not normalized:
            continue
        try:
            # Selenium doesn't allow setting "sameSite" in some versions; remove if problematic
            if "sameSite" in normalized:
                try:
                    driver.add_cookie(normalized)
                except Exception:
                    # try removing sameSite and retry
                    normalized.pop("sameSite", None)
                    driver.add_cookie(normalized)
            else:
                driver.add_cookie(normalized)
            added += 1
        except Exception as e:
            # some cookies may fail to add; continue
            # print(f"[WARN] Could not add cookie {normalized.get('name')}: {e}")
            continue

    if added > 0:
        # optionally save to pickle for later convenience
        if save_pickle:
            try:
                cookies_for_pickle = driver.get_cookies()
                with open(COOKIES_FILE_PKL, "wb") as f:
                    pickle.dump(cookies_for_pickle, f)
                print(f"[INFO] Injected {added} cookies and saved to {COOKIES_FILE_PKL}")
            except Exception as e:
                print(f"[WARN] Could not save cookies pickle: {e}")
        # refresh and wait
        driver.refresh()
        time.sleep(4)
        return True
    else:
        print("[WARN] No cookies injected (check JSON format and domain).")
        return False

def load_cookies_from_file(driver, json_path=COOKIES_JSON_FILE, save_pickle=True):
    if not os.path.exists(json_path):
        print(f"[WARN] Cookie JSON file {json_path} not found.")
        return False
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read()
            return load_cookies_from_json_string(driver, content, save_pickle=save_pickle)
    except Exception as e:
        print(f"[ERROR] Failed to read cookie JSON file: {e}")
        return False

def try_load_pickle(driver, pickle_path=COOKIES_FILE_PKL):
    if not os.path.exists(pickle_path):
        return False
    try:
        with open(pickle_path, "rb") as f:
            cookies = pickle.load(f)
        driver.get("https://twitter.com")
        time.sleep(1)
        added = 0
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
                added += 1
            except Exception:
                continue
        if added:
            driver.refresh()
            time.sleep(3)
            print(f"[INFO] Loaded {added} cookies from pickle {pickle_path}.")
            return True
    except Exception as e:
        print(f"[WARN] Could not load pickle cookies: {e}")
    return False

# =================== HELPER: click latest tab ===================
def click_latest_tab(driver):
    try:
        possible = driver.find_elements(By.XPATH, "//a | //div[@role='tab'] | //div[@role='link']")
        for el in possible:
            try:
                text = el.text.strip().lower()
                if "latest" in text:
                    driver.execute_script("arguments[0].scrollIntoView({block:'center'})", el)
                    time.sleep(random.uniform(0.3, 0.8))
                    el.click()
                    time.sleep(random.uniform(1.5, 3))
                    return True
            except Exception:
                continue
    except Exception:
        pass
    try:
        latest = driver.find_element(By.LINK_TEXT, "Latest")
        latest.click()
        time.sleep(random.uniform(1.5, 3))
        return True
    except Exception:
        return False

# =================== SEARCH + SCROLL STRATEGIES ===================
def search_hashtag(driver, hashtag):
    driver.get("https://twitter.com/home")
    time.sleep(random.uniform(2.5, 5))
    selectors = [
        '//input[@aria-label="Search query"]',
        '//input[@placeholder="Search Twitter"]',
        '//input[contains(@aria-label,"Search")]',
        '//input[@role="combobox"]'
    ]
    search_input = None
    for sel in selectors:
        try:
            search_input = WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.XPATH, sel)))
            if search_input:
                break
        except Exception:
            continue
    if not search_input:
        raise RuntimeError("Search input not found on page (Twitter layout changed or not logged in).")

    search_input.clear()
    for ch in hashtag:
        search_input.send_keys(ch)
        time.sleep(random.uniform(0.03, 0.12))
    search_input.send_keys(Keys.ENTER)
    time.sleep(random.uniform(3, 5))

    if not click_latest_tab(driver):
        try:
            script = f"window.history.pushState({{}}, '', '?q={hashtag.replace('#','%23')}&f=live');"
            driver.execute_script(script)
            driver.refresh()
            time.sleep(random.uniform(2.5, 4))
        except Exception:
            pass

def small_increment_scroll(driver):
    try:
        driver.execute_script("window.scrollBy(0, 300);")
        return True
    except Exception:
        return False

def page_down_scroll(driver):
    try:
        body = driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.PAGE_DOWN)
        return True
    except Exception:
        return False

def scroll_into_last_article(driver):
    try:
        articles = driver.find_elements(By.XPATH, "//article")
        if not articles:
            return False
        last = articles[-1]
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", last)
        return True
    except Exception:
        return False

def js_scroll_bottom(driver):
    try:
        driver.execute_script("window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });")
        return True
    except Exception:
        return False

SCROLL_STRATEGIES = [small_increment_scroll, scroll_into_last_article, page_down_scroll, js_scroll_bottom]

# =================== TWEET EXTRACTION ===================
def extract_tweet_from_article(article):
    try:
        content = ""
        try:
            content_el = article.find_element(By.XPATH, ".//div[@data-testid='tweetText']")
            content = content_el.text or ""
        except Exception:
            content = article.text or ""

        username = ""
        try:
            username_elem = article.find_element(By.XPATH, ".//div[@dir='auto']//span")
            username = username_elem.text or ""
        except Exception:
            username = ""

        timestamp = None
        try:
            time_el = article.find_element(By.TAG_NAME, "time")
            timestamp = time_el.get_attribute("datetime")
        except Exception:
            timestamp = None

        likes = retweets = replies = 0
        try:
            metrics = article.find_elements(By.XPATH, ".//div[@data-testid='like'] | .//div[@data-testid='retweet'] | .//div[@data-testid='reply']")
            def _num(el_idx):
                try:
                    t = metrics[el_idx].text.strip()
                    return int(t.replace(",", "")) if t and t.isdigit() else 0
                except Exception:
                    return 0
            if len(metrics) > 0: likes = _num(0)
            if len(metrics) > 1: retweets = _num(1)
            if len(metrics) > 2: replies = _num(2)
        except Exception:
            pass

        hashtags_in_tweet = re.findall(r"#\w+", content)
        mentions = re.findall(r"@\w+", content)

        return {
            "username": username,
            "timestamp": timestamp,
            "content": content,
            "likes": likes,
            "retweets": retweets,
            "replies": replies,
            "hashtags": hashtags_in_tweet,
            "mentions": mentions
        }
    except Exception:
        return None

# =================== SCROLL & COLLECT ===================
def scroll_and_collect(driver, hashtag, max_tweets=MAX_TWEETS_PER_HASHTAG, min_tweets=MIN_TWEETS_PER_HASHTAG):
    tweets_data = []
    seen = set()
    retry_page_error = 0

    try:
        search_hashtag(driver, hashtag)
    except Exception as e:
        print(f"[ERROR] Searching {hashtag} failed: {e}")
        return tweets_data

    scroll_retries = 0
    while len(tweets_data) < max_tweets:
        page_source = driver.page_source
        if "Something went wrong" in page_source or "Try reloading" in page_source:
            retry_page_error += 1
            if retry_page_error > 3:
                print(f"[ERROR] Persistent error for {hashtag}, aborting.")
                break
            print(f"[WARN] Twitter error page. Sleeping {ERROR_PAUSE_SECONDS//60} mins.")
            time.sleep(ERROR_PAUSE_SECONDS)
            driver.refresh()
            continue

        articles = driver.find_elements(By.XPATH, "//article")
        new_found = False
        for art in articles:
            try:
                sig = (art.text or "")[:160]
                if not sig or sig in seen:
                    continue
                extracted = extract_tweet_from_article(art)
                if not extracted or not extracted["content"].strip():
                    continue
                seen.add(sig)
                tweets_data.append(extracted)
                new_found = True
                if len(tweets_data) >= max_tweets:
                    break
            except Exception:
                continue

        if len(tweets_data) >= max_tweets:
            break

        if not new_found:
            scroll_retries += 1
            if scroll_retries > SCROLL_RETRY_LIMIT:
                if len(tweets_data) >= min_tweets:
                    print(f"[INFO] Min {min_tweets} tweets reached for {hashtag}.")
                    break
                else:
                    print(f"[WARN] Could not load enough tweets for {hashtag}. Final retry sequence...")
                    time.sleep(30)
                    for strategy in SCROLL_STRATEGIES:
                        try:
                            strategy(driver)
                            time.sleep(random.uniform(1.5, 3.2))
                        except Exception:
                            pass
                    break
            strategy = SCROLL_STRATEGIES[scroll_retries % len(SCROLL_STRATEGIES)]
            try:
                strategy(driver)
            except Exception:
                pass
            time.sleep(random.uniform(2, 4))
        else:
            scroll_retries = 0
            time.sleep(random.uniform(1.2, 3))

    print(f"[INFO] For {hashtag}: collected {len(tweets_data)} tweets.")
    return tweets_data

# =================== MAIN COLLECTOR ===================
def collect_all_hashtags():
    driver = init_driver()

    # 1) Try loading previously-saved pickle cookies for convenience
    loaded = try_load_pickle(driver, COOKIES_FILE_PKL)
    if not loaded:
        # 2) If user provided a JSON string in script, use it
        if COOKIES_JSON:
            print("[INFO] Using COOKIES_JSON provided in script to inject cookies.")
            ok = load_cookies_from_json_string(driver, COOKIES_JSON)
            if not ok:
                print("[WARN] Provided COOKIES_JSON failed to inject. Trying JSON file next.")
        # 3) If there's a JSON file in working dir, use it
        if not os.path.exists(COOKIES_FILE_PKL):
            if os.path.exists(COOKIES_JSON_FILE):
                print(f"[INFO] Loading cookie JSON from {COOKIES_JSON_FILE}.")
                load_cookies_from_file(driver, COOKIES_JSON_FILE)
            else:
                print("[WARN] No cookie pickle or json provided. Please place exported cookie JSON at"
                      f" {COOKIES_JSON_FILE} or set COOKIES_JSON variable, then re-run.")
                driver.quit()
                raise SystemExit("No cookies provided; aborting to prevent login attempts.")

    # After injecting cookies, verify that we're logged in by checking the search input presence
    try:
        driver.get("https://twitter.com/home")
        time.sleep(3)
        # if logged in, search box should be present
        WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.XPATH, '//input[@aria-label="Search query"]')))
        print("[INFO] Login appears successful (search box found). Proceeding.")
    except Exception:
        print("[WARN] Could not verify login. Twitter may have rejected cookies or cookies expired.")
        driver.quit()
        raise SystemExit("Login verification failed. Update cookies and retry.")

    all_tweets = []
    if os.path.exists(DATA_FILE):
        try:
            prev = pd.read_parquet(DATA_FILE)
            all_tweets.extend(prev.to_dict(orient="records"))
            print(f"[INFO] Loaded {len(prev)} previously saved tweets.")
        except Exception:
            print("[WARN] Failed to read existing data file; starting fresh.")

    for tag in HASHTAGS:
        try:
            print(f"[ACTION] Scraping tag: {tag}")
            tweets = scroll_and_collect(driver, tag)
            all_tweets.extend(tweets)
            try:
                df_partial = pd.DataFrame(all_tweets)
                df_partial.to_parquet(DATA_FILE, index=False)
                print(f"[INFO] Saved partial data to {DATA_FILE} (total {len(all_tweets)}).")
            except Exception as e:
                print(f"[ERROR] Failed saving partial data: {e}")
            sleep_time = random.uniform(15, 180)
            print(f"[INFO] Sleeping {int(sleep_time)}s before next tag.")
            time.sleep(sleep_time)
        except Exception as e:
            print(f"[ERROR] Error on {tag}: {e}", file=sys.stderr)
            time.sleep(random.uniform(60, 120))
            continue

    driver.quit()
    return all_tweets

# =================== CLEANING & ANALYSIS ===================
def clean_tweets(tweets):
    if not tweets:
        return pd.DataFrame()
    df = pd.DataFrame(tweets)
    df.drop_duplicates(subset=["content"], inplace=True)
    df["content"] = df["content"].apply(lambda x: unicodedata.normalize("NFKC", x) if isinstance(x, str) else x)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    now_utc = datetime.now(timezone.utc)
    df = df[df["timestamp"].notna()]
    df = df[df["timestamp"] >= (now_utc - timedelta(days=1))]
    return df

def save_cleaned(df, filename=CLEANED_FILE):
    if df.empty:
        print("[WARN] Cleaned dataframe is empty; nothing to save.")
        return
    df.to_parquet(filename, index=False)
    print(f"[INFO] Saved cleaned data to {filename} (rows: {len(df)})")

def generate_signal(df):
    if df.empty:
        return df
    vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
    X = vectorizer.fit_transform(df["content"].astype(str)).toarray()
    df["signal"] = np.mean(X, axis=1)
    return df

def plot_signals(df, sample_size=500):
    if df.empty:
        print("[WARN] Nothing to plot.")
        return
    plot_df = df.sample(min(sample_size, len(df)), random_state=42)
    plt.figure(figsize=(10, 4))
    plt.hist(plot_df["signal"], bins=30)
    plt.title("Distribution of Tweet Signals")
    plt.xlabel("Signal Score")
    plt.ylabel("Count")
    plt.show()

# =================== RUN ===================
if __name__ == "__main__":
    print("[START] Running Twitter/X scraper.")
    collected = collect_all_hashtags()
    print(f"[INFO] Total raw tweets collected: {len(collected)}")

    cleaned = clean_tweets(collected)
    print(f"[INFO] Cleaned tweets (24h): {len(cleaned)}")
    save_cleaned(cleaned)

    if not cleaned.empty:
        cleaned = generate_signal(cleaned)
        print("[INFO] Generated 'signal' scores.")
        plot_signals(cleaned)
    else:
        print("[INFO] No cleaned tweets available for analysis.")

    print("[END] Done.")
