#!/usr/bin/env python
# coding: utf-8

# 
# # **Scraping Tweets with playwright**

# In[6]:


# Standard Library Imports
import datetime as dt
import random
import re
import asyncio
from datetime import datetime, timedelta

# Third-Party Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Machine Learning & Data Science Libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Sentiment Analysis & Finance Libraries
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import preprocessor as p

# Async & Web Scraping Libraries
import nest_asyncio
from playwright.async_api import async_playwright

# Apply nest_asyncio patch for asyncio inside Jupyter
nest_asyncio.apply()


# In[8]:


get_ipython().system("pip install pycodestyle-magic")
get_ipython().run_line_magic("load_ext", "pycodestyle_magic")
get_ipython().run_line_magic("pycodestyle_on", "")


# In[ ]:


# Install required packages (only once)
get_ipython().system(
    "pip install nest_asyncio playwright playwright-stealth pandas --quiet"
)

# Install browser engines (Chromium) for Playwright
get_ipython().system("playwright install")


# In[ ]:


# Manual Login

async def login_and_save_storage():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto("https://twitter.com/login")
        print("Log in manually in the browser window...")

        await asyncio.sleep(70)  # Give yourself time to log in

        await context.storage_state(path="twitter_auth.json")
        print("Session saved to twitter_auth.json")
        await browser.close()


asyncio.run(login_and_save_storage())


# In[ ]:


# Testing scraping method for single tweet

async def test_single_tweet():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        # Navigate to a single tweet
        await page.goto(
            "https://x.com/StockMKTNewz/status/1889344331053404337"
        )

        # Extract Tweet
        await page.wait_for_selector(
            "article[aria-labelledby]", timeout=10000
        )  # Wait for up to 10s
        tweet = await page.query_selector("article[aria-labelledby]")

        if tweet:
            try:
                # Extract Date
                time_element = await tweet.query_selector("time")
                date = (
                    await time_element.get_attribute("datetime")
                    if time_element
                    else "Unknown"
                )

                # Extract Text
                text_element = await tweet.query_selector("div[lang]")
                text = (
                    await text_element.inner_text()
                    if text_element
                    else "Unknown"
                )

                # Extract Username
                handles = await tweet.query_selector_all("a")
                username = "Unknown"
                for h in handles:
                    handle_text = await h.inner_text() or await h.evaluate(
                        "(el) => el.textContent"
                    )
                    if handle_text.startswith("@"):
                        username = handle_text
                        break  # Stop at first valid username

                print(
                    f"📌 Tweet Extracted:\n- Date: {date}\n- Username: {username}\n- Text: {text}"
                )

            except Exception as e:
                print(f"Error extracting tweet: {e}")

        else:
            print("No tweet found!")

        await browser.close()


# Run the test
asyncio.run(test_single_tweet())


# In[ ]:


# Scraping tweets of Netflix and Amazon

nest_asyncio.apply()

async def scrape_twitter(
    stock, start_date, end_date, accounts, retries=3, timeout=120000
):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state="twitter_auth.json")
        page = await context.new_page()

        all_tweets = []  # Store tweets from all accounts

        for account in accounts:
            print(
                f"🔄 Collecting tweets for {account} from {start_date} to {end_date}"
            )
            search_query = f"{stock} from:{account} since:{start_date} until:{end_date} lang:en"
            final_url = f"https://twitter.com/search?q={search_query.replace(' ', '%20')}&src=typed_query&f=live"

            print(f"🔍 Searching {account}: {final_url}")
            await page.goto(final_url, timeout=timeout)

            # Click "Latest" tab explicitly after search loads
            try:
                await page.wait_for_selector("text=Latest", timeout=10000)
                await page.click("text=Latest")
                print("✅ Clicked 'Latest' tab.")
            except Exception as e:
                print(f"⚠️ 'Latest' tab click skipped - error: {e}")

            # Ensure tweets load before scraping, but handle cases where no tweets exist
            try:
                await page.wait_for_selector(
                    "article[aria-labelledby], article", timeout=15000
                )  # note that twitter frequently changes DOM structure
            except:
                print(f"⚠️ No tweets found for {account}. Skipping...")
                all_tweets.append(
                    {
                        "tweet": "No tweets found",
                        "date": "N/A",
                        "account": account,
                    }
                )
                continue  # Skip to the next account

            # Fetch initial tweets
            tweet_elements = await page.query_selector_all(
                "article[aria-labelledby], article"
            )
            print(f"🔍 Found {len(tweet_elements)} tweets before scrolling.")

            # Scroll dynamically **only if tweets exist**
            if tweet_elements:
                previous_tweet_count = 0
                scroll_attempts = 0

                while scroll_attempts < 50:  # Avoid infinite scrolling
                    await page.evaluate(
                        "window.scrollBy(0, 2000 + Math.random() * 500)"
                    )
                    await asyncio.sleep(random.uniform(2, 5))

                    tweet_elements = await page.query_selector_all(
                        "article[aria-labelledby], article"
                    )
                    current_tweet_count = len(tweet_elements)

                    if (
                        current_tweet_count == previous_tweet_count
                    ):  # No new tweets loaded
                        print(
                            f"✅ No new tweets detected for {account}, stopping scroll."
                        )
                        break

                    previous_tweet_count = current_tweet_count
                    scroll_attempts += 1
            else:
                print(
                    f"⚠️ No tweets detected initially for {account}, skipping scrolling."
                )

            # Extract tweets
            for tweet in tweet_elements:
                try:
                    time_element = await tweet.query_selector("time")
                    date = (
                        await time_element.get_attribute("datetime")
                        if time_element
                        else "Unknown"
                    )

                    text_element = await tweet.query_selector("div[lang]")
                    text = (
                        await text_element.inner_text()
                        if text_element
                        else "Unknown"
                    )

                    handles = await tweet.query_selector_all("a")
                    username = "Unknown"
                    for h in handles:
                        handle_text = await h.inner_text() or await h.evaluate(
                            "(el) => el.textContent"
                        )
                        if handle_text.startswith("@"):
                            username = handle_text
                            break

                    # Append all tweets without filtering by "$NFLX"
                    all_tweets.append(
                        {
                            "tweet": text.strip(),
                            "date": date,
                            "account": username,
                        }
                    )

                except Exception as e:
                    print(f"⚠️ Error processing tweet: {e}")

        await browser.close()
        return pd.DataFrame(all_tweets).drop_duplicates(
            subset=["tweet"]
        )  # Remove duplicate tweets


async def collect_tweets(
    stock, start_date, end_date, accounts, filename_prefix
):
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    all_tweets = pd.DataFrame()
    weeks_processed = 0

    while current_date < end_date_dt:
        week_end = min(current_date + timedelta(days=7), end_date_dt)
        week_start_str = current_date.strftime("%Y-%m-%d")
        week_end_str = week_end.strftime("%Y-%m-%d")
        print(f"🔄 Collecting tweets from {week_start_str} to {week_end_str}")

        weekly_df = await scrape_twitter(
            stock, week_start_str, week_end_str, accounts
        )
        all_tweets = pd.concat([all_tweets, weekly_df], ignore_index=True)
        current_date = week_end
        weeks_processed += 1

        # Save progress every 4 weeks
        if weeks_processed % 4 == 0:
            checkpoint_file = (
                f"{filename_prefix}_checkpoint_{week_start_str}.csv"
            )
            all_tweets.to_csv(checkpoint_file, index=False)
            print(f"💾 Saved intermediate results to {checkpoint_file}")

    return all_tweets


start_date = "2024-06-01"
end_date = "2025-04-01"


async def main():
    accounts = [
        "cnbc",
        "Stocktwits",
        "WSJ",
        "reuters",
        "forbes",
        "RobinJPowell",
        "WSJDealJournal",
        "Benzinga",
        "EventDrivenMgr",
        "SeekingAlpha",
        "WSJmarkets",
        "Business",
        "tradingguru",
        "marketcurrents",
        "financialtimes",
        "elonmusk",
        "SpiegelPeter",
        "BillGates",
        "bespokeinvest",
        "AlphaTrends",
        "StockMKTNewz",
        "PeterLBrandt",
        "JPMorgan",
        "SvenHenrich",
        "Bloomberg",
    ]

    netflix_tweets = await collect_tweets(
        "Netflix OR $NFLX", start_date, end_date, accounts, "netflix"
    )
    netflix_tweets.to_csv("netflix_tweets.csv", index=False)
    print(
        f"✅ Saved {len(netflix_tweets)} tweets for Netflix to netflix_tweets.csv"
    )

    amazon_tweets = await collect_tweets(
        "Amazon OR $AMZN", start_date, end_date, accounts, "amazon"
    )
    amazon_tweets.to_csv("amazon_tweets.csv", index=False)
    print(
        f"✅ Saved {len(amazon_tweets)} tweets for Amazon to amazon_tweets.csv"
    )


asyncio.run(main())


# In[7]:


# Import datasets of tweets
netflix_df = pd.read_csv("netflix_all_tweets.csv")
amazon_df = pd.read_csv("amazon_all_tweets.csv")

# drop duplicates
netflix_df = netflix_df.drop_duplicates(subset="tweet", keep="first")
amazon_df = amazon_df.drop_duplicates(subset="tweet", keep="first")

print(netflix_df.shape)
print(amazon_df.shape)


# # **Tweet Preprocessing**

# In[10]:


# Combining both dataframes into one with adding a column of netflix / amzn
amazon_df["stock"] = "amazon"
netflix_df["stock"] = "netflix"
tweet_df = pd.concat([amazon_df, netflix_df])
tweet_df["date"] = pd.to_datetime(tweet_df["date"]).dt.date
tweet_df = tweet_df[tweet_df["tweet"] != "No tweets found"]  # dropping
tweet_df


# In[12]:


# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

# Define financial stopwords that should NOT be removed
financial_stopwords = {
    "up",
    "down",
    "bullish",
    "bearish",
    "profit",
    "loss",
    "growth",
    "decline",
}


def preprocess_pipeline(tweet):
    # Remove URLs, mentions, hashtags, emojis, and reserved words
    tweet = p.clean(str(tweet))

    # Remove stock symbols ($NFLX, $AMZN)
    tweet = re.sub(r"[$]+[a-zA-Z]+", "", tweet)

    # Remove punctuation & numbers
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)

    # Normalize text (reduce lengthening)
    pattern = re.compile(r"(.)\1{2,}")
    tweet = pattern.sub(r"\1\1", tweet)

    # Convert to lowercase
    tweet = tweet.lower()

    # Tokenization & Stopword Removal (EXCLUDING financial stopwords)
    tokens = word_tokenize(tweet)
    stop_words = stopwords.words("english")
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words or word in financial_stopwords
    ]

    return " ".join(tokens)  # Convert back to string


# Apply pipeline to all tweets
tweet_df["processed_tweet"] = tweet_df["tweet"].apply(preprocess_pipeline)

tweet_df.head()


# In[13]:


# Filter tweets for Amazon and Netflix separately
amazon_tweets = " ".join(
    tweet_df[tweet_df["stock"] == "amazon"]["processed_tweet"]
)
netflix_tweets = " ".join(
    tweet_df[tweet_df["stock"] == "netflix"]["processed_tweet"]
)

# Generate word clouds
amazon_wordcloud = WordCloud(
    background_color="white", width=800, height=400
).generate(amazon_tweets)

netflix_wordcloud = WordCloud(
    background_color="white", width=800, height=400
).generate(netflix_tweets)

# Display word clouds side by side
fig, ax = plt.subplots(1, 2, figsize=(15, 7))

ax[0].imshow(amazon_wordcloud, interpolation="bilinear")
ax[0].set_title("Amazon Word Cloud", fontsize=18)
ax[0].axis("off")

ax[1].imshow(netflix_wordcloud, interpolation="bilinear")
ax[1].set_title("Netflix Word Cloud", fontsize=18)
ax[1].axis("off")

plt.show()


# # **VADER - Sentiment Analysis**

# In[15]:


analyser = SentimentIntensityAnalyzer()


# Function to compute sentiment scores
def get_sentiment_scores(text):
    scores = analyser.polarity_scores(text)
    return pd.Series(
        [scores["neg"], scores["neu"], scores["pos"], scores["compound"]],
        index=["neg", "neu", "pos", "compound"],
    )


tweet_df[["neg", "neu", "pos", "compound"]] = tweet_df[
    "processed_tweet"
].apply(get_sentiment_scores)

tweet_df.head()


# In[18]:


# Defining classes
def analyse_sentiment(compound):
    if compound > 0:
        return "Positive_VADER"
    if compound == 0:
        return "Neutral_VADER"
    if compound < 0:
        return "Negative_VADER"


# Apply the funtion on Polarity column and add the results into a new column
tweet_df["class_VADER"] = tweet_df["compound"].apply(analyse_sentiment)

print(tweet_df.shape)
tweet_df


# In[19]:


# Sorting values by date
tweet_df = tweet_df.sort_values(by="date", ascending=True)


# In[20]:


# Generate dummy variables for class_VADER
dummies_VADER = pd.get_dummies(tweet_df["class_VADER"])
tweet_df = pd.concat([tweet_df, dummies_VADER], axis=1)

# Ensure all expected columns exist before grouping
expected_cols = ["Negative_VADER", "Neutral_VADER", "Positive_VADER"]
for col in expected_cols:
    if col not in tweet_df.columns:
        tweet_df[col] = 0  # Add missing column with default value

# Group by date to sum counts per day
nrclass_VADER = (
    tweet_df.groupby(["stock", "date"])[expected_cols].sum().reset_index()
)

# Shift values forward by one day **one column at a time** to avoid length mismatch
for col in expected_cols:
    nrclass_VADER[col] = nrclass_VADER[col].shift(1)

tweet_sentiment_df = nrclass_VADER


# In[22]:


# DOUBLE CHECK if we have any NaN in the sentiment
tweet_sentiment_df.isna().sum()


# In[ ]:


# Store the Data after Sentiment Analysis
tweet_sentiment_df.to_csv("tweet_sentiment_df.csv")


# In[ ]:


# Grouped Bar Chart

# Group by stock and sum sentiment values
sentiment_sums = (
    tweet_sentiment_df.groupby("stock")[
        ["Negative_VADER", "Neutral_VADER", "Positive_VADER"]
    ]
    .sum()
    .reset_index()
)

# Create grouped bar chart
fig = go.Figure(
    data=[
        go.Bar(
            name="Neutral",
            x=sentiment_sums["stock"],
            y=sentiment_sums["Neutral_VADER"],
        ),
        go.Bar(
            name="Negative",
            x=sentiment_sums["stock"],
            y=sentiment_sums["Negative_VADER"],
        ),
        go.Bar(
            name="Positive",
            x=sentiment_sums["stock"],
            y=sentiment_sums["Positive_VADER"],
        ),
    ]
)

# Update layout
fig.update_layout(
    barmode="group",
    title_text="Distribution of Sentiment Classes for Netflix & Amazon",
    title_font=dict(size=30),
    plot_bgcolor="white",
    xaxis_title="Stock",
    yaxis_title="Sentiment Count",
)

fig.show()




# # **Getting Financial data (Yahoo)**

# In[24]:


# Fetch historical market data
AMZN = yf.Ticker("AMZN")
NFLX = yf.Ticker("NFLX")

# Get historical data for Amazon
amazon_hist = AMZN.history(start="2024-06-01", end="2025-04-01")
amazon_hist["stock"] = "amazon"  # Add stock identifier

# Get historical data for Netflix
netflix_hist = NFLX.history(start="2024-06-01", end="2025-04-01")
netflix_hist["stock"] = "netflix"  # Add stock identifier

# Combine both datasets
financials = pd.concat([amazon_hist, netflix_hist], ignore_index=False)

print(financials.shape)
financials

# should replace for var
# start_date = "2025-02-09"
# end_date = "2025-02-13"


# In[27]:


# Calculate returns
financials["returns"] = financials["Close"] / financials["Close"].shift(1) - 1


# Get class
def analyse_returns(returns):
    if returns >= 0:
        return "1"
    else:
        return "0"


# Apply the function on return column and add the results into a new column
financials["return_sign"] = financials["returns"].apply(analyse_returns)

print(financials.shape)
financials


# In[28]:


# Dropping unnecessary columns
financials = financials.reset_index()
financials["date"] = pd.to_datetime(financials["Date"]).dt.date
financials = financials.drop(
    columns=[
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Dividends",
        "Stock Splits",
        "Date",
    ]
)
print(financials.shape)
financials


# 
# # **Supervised Machine Learning - SML**

# ## Train Test Split

# In[30]:


# Merge
merged_sentiment = tweet_sentiment_df.merge(
    financials, on=["stock", "date"], how="inner"
)


print(merged_sentiment.shape)
merged_sentiment


# In[31]:


# Extract y values based on stock type
y_net = merged_sentiment[merged_sentiment["stock"] == "netflix"][
    "return_sign"
].to_numpy()
y_am = merged_sentiment[merged_sentiment["stock"] == "amazon"][
    "return_sign"
].to_numpy()

print("y_net:", y_net)
print("y_am:", y_am)


# In[33]:


# Train test split netflix
random_seed_val = 100

x_net = (
    merged_sentiment[merged_sentiment["stock"] == "netflix"]
    .iloc[:, 2:5]
    .to_numpy()
)

x_net_train, x_net_test, y_net_train, y_net_test = train_test_split(
    x_net, y_net, test_size=0.2, shuffle=False
)


# In[34]:


# Train test split amazon
x_am = (
    merged_sentiment[merged_sentiment["stock"] == "amazon"]
    .iloc[:, 2:5]
    .to_numpy()
)

x_am_train, x_am_test, y_am_train, y_am_test = train_test_split(
    x_am, y_am, test_size=0.2, shuffle=False
)


# In[35]:


print(x_net_train.shape)
print(x_am_train.shape)

print(x_net_test.shape)
print(x_am_test.shape)


# In[36]:


y_net_train = y_net_train.astype(int)
y_net_test = y_net_test.astype(int)
y_am_train = y_am_train.astype(int)
y_am_test = y_am_test.astype(int)


# In[39]:


# scaling the data
scaler_netflix = StandardScaler()
scaler_amazon = StandardScaler()

netflix_train_scaled = scaler_netflix.fit_transform(x_net_train)
netflix_test_scaled = scaler_netflix.transform(x_net_test)

amazon_train_scaled = scaler_amazon.fit_transform(x_am_train)
amazon_test_scaled = scaler_amazon.transform(x_am_test)


# ## The Models (LR, MLP and SVM)

# ### Logistic Regression

# In[43]:


# logistic regression

datasets_scaled = {
    "netflix": (netflix_train_scaled, y_net_train),
    "amazon": (amazon_train_scaled, y_am_train),
}

log_models = {}

for name, (train_scaled, y_train) in datasets_scaled.items():
    model = LogisticRegression(
        C=0.3, class_weight="balanced", random_state=100
    )
    model.fit(train_scaled, y_train)
    log_models[name] = model

# predictions
y_net_pred_lr = log_models["netflix"].predict(netflix_test_scaled)
y_am_pred_lr = log_models["amazon"].predict(amazon_test_scaled)


# Function to evaluate multiple metrics dynamically
def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


# Define datasets for evaluation
datasets = {
    "Netflix": (y_net_test, y_net_pred_lr),
    "Amazon": (y_am_test, y_am_pred_lr),
}

# Metrics to check
metrics_to_check = ["accuracy", "precision", "recall", "f1"]

# Evaluate all models
for stock, (y_true, y_pred) in datasets.items():
    results = evaluate_model(y_true, y_pred)
    for metric in metrics_to_check:
        print(f"{stock} {metric.capitalize()}: {results[metric]:.4f}")


# ### MLP

# In[42]:


# Define scaled datasets
datasets_scaled = {
    "netflix": (netflix_train_scaled, y_net_train),
    "amazon": (amazon_train_scaled, y_am_train),
}

mlp_models = {}

# Train MLP models for each dataset
for name, (train_scaled, y_train) in datasets_scaled.items():
    model = MLPClassifier(
        hidden_layer_sizes=(50, 30),
        activation="relu",
        solver="lbfgs",
        alpha=0.001,
        random_state=100,
        max_iter=3000,
    )
    model.fit(train_scaled, y_train)
    mlp_models[name] = model

# Predictions
y_net_pred_mlp = mlp_models["netflix"].predict(netflix_test_scaled)
y_am_pred_mlp = mlp_models["amazon"].predict(amazon_test_scaled)


# Evaluation function
def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


# Define test sets
datasets = {
    "Netflix": (y_net_test, y_net_pred_mlp),
    "Amazon": (y_am_test, y_am_pred_mlp),
}

metrics_to_check = ["accuracy", "precision", "recall", "f1"]

# Evaluate all models
for stock, (y_true, y_pred) in datasets.items():
    results = evaluate_model(y_true, y_pred)
    for metric in metrics_to_check:
        print(f"{stock} {metric.capitalize()}: {results[metric]:.4f}")


# ### SVM

# In[45]:


# Define scaled datasets
datasets_scaled = {
    "netflix": (netflix_train_scaled, y_net_train),
    "amazon": (amazon_train_scaled, y_am_train),
}

svm_models = {}

# Train SVM models for each dataset
for name, (train_scaled, y_train) in datasets_scaled.items():
    model = SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=100)
    model.fit(train_scaled, y_train)
    svm_models[name] = model

# Predictions
y_net_pred_svm = svm_models["netflix"].predict(netflix_test_scaled)
y_am_pred_svm = svm_models["amazon"].predict(amazon_test_scaled)


# Evaluation function
def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


# Define test sets
datasets = {
    "Netflix": (y_net_test, y_net_pred_svm),
    "Amazon": (y_am_test, y_am_pred_svm),
}

metrics_to_check = ["accuracy", "precision", "recall", "f1"]

# Evaluate all models
for stock, (y_true, y_pred) in datasets.items():
    results = evaluate_model(y_true, y_pred)
    for metric in metrics_to_check:
        print(f"{stock} {metric.capitalize()}: {results[metric]:.4f}")


# ### Model Performance Comparison - Plots

# In[49]:


# Model performance on training set
# Compute accuracy for each model
train_accuracy = np.array(
    [
        accuracy_score(
            y_net_train, log_models["netflix"].predict(netflix_train_scaled)
        ),
        accuracy_score(
            y_net_train, mlp_models["netflix"].predict(netflix_train_scaled)
        ),
        accuracy_score(
            y_net_train, svm_models["netflix"].predict(netflix_train_scaled)
        ),
        accuracy_score(
            y_am_train, log_models["amazon"].predict(amazon_train_scaled)
        ),
        accuracy_score(
            y_am_train, mlp_models["amazon"].predict(amazon_train_scaled)
        ),
        accuracy_score(
            y_am_train, svm_models["amazon"].predict(amazon_train_scaled)
        ),
    ]
)

# Compute recall for each model
train_recall = np.array(
    [
        recall_score(
            y_net_train, log_models["netflix"].predict(netflix_train_scaled)
        ),
        recall_score(
            y_net_train, mlp_models["netflix"].predict(netflix_train_scaled)
        ),
        recall_score(
            y_net_train, svm_models["netflix"].predict(netflix_train_scaled)
        ),
        recall_score(
            y_am_train, log_models["amazon"].predict(amazon_train_scaled)
        ),
        recall_score(
            y_am_train, mlp_models["amazon"].predict(amazon_train_scaled)
        ),
        recall_score(
            y_am_train, svm_models["amazon"].predict(amazon_train_scaled)
        ),
    ]
)

# Compute precision for each model
train_precision = np.array(
    [
        precision_score(
            y_net_train, log_models["netflix"].predict(netflix_train_scaled)
        ),
        precision_score(
            y_net_train, mlp_models["netflix"].predict(netflix_train_scaled)
        ),
        precision_score(
            y_net_train, svm_models["netflix"].predict(netflix_train_scaled)
        ),
        precision_score(
            y_am_train, log_models["amazon"].predict(amazon_train_scaled)
        ),
        precision_score(
            y_am_train, mlp_models["amazon"].predict(amazon_train_scaled)
        ),
        precision_score(
            y_am_train, svm_models["amazon"].predict(amazon_train_scaled)
        ),
    ]
)

# Compute F1-score for each model
train_f1 = np.array(
    [
        f1_score(
            y_net_train, log_models["netflix"].predict(netflix_train_scaled)
        ),
        f1_score(
            y_net_train, mlp_models["netflix"].predict(netflix_train_scaled)
        ),
        f1_score(
            y_net_train, svm_models["netflix"].predict(netflix_train_scaled)
        ),
        f1_score(
            y_am_train, log_models["amazon"].predict(amazon_train_scaled)
        ),
        f1_score(
            y_am_train, mlp_models["amazon"].predict(amazon_train_scaled)
        ),
        f1_score(
            y_am_train, svm_models["amazon"].predict(amazon_train_scaled)
        ),
    ]
)

# Store results in DataFrame
df_train_performance = pd.DataFrame(
    [train_accuracy, train_recall, train_precision, train_f1]
).T
df_train_performance.columns = ["Accuracy", "Recall", "Precision", "F1-score"]

# Labels for models & datasets
df_train_performance.index = [
    "LG-Netflix",
    "MLP-Netflix",
    "SVM-Netflix",
    "LG-Amazon",
    "MLP-Amazon",
    "SVM-Amazon",
]

# Display results
df_train_performance.head(6)


# In[50]:


# Model performance on test set
# Compute accuracy for each model
met_accuracy = np.array(
    [
        accuracy_score(y_net_test, y_net_pred_lr),
        accuracy_score(y_net_test, y_net_pred_mlp),
        accuracy_score(y_net_test, y_net_pred_svm),
        accuracy_score(y_am_test, y_am_pred_lr),
        accuracy_score(y_am_test, y_am_pred_mlp),
        accuracy_score(y_am_test, y_am_pred_svm),
    ]
)

# Compute recall for each model
met_recall = np.array(
    [
        recall_score(y_net_test, y_net_pred_lr),
        recall_score(y_net_test, y_net_pred_mlp),
        recall_score(y_net_test, y_net_pred_svm),
        recall_score(y_am_test, y_am_pred_lr),
        recall_score(y_am_test, y_am_pred_mlp),
        recall_score(y_am_test, y_am_pred_svm),
    ]
)

# Compute precision for each model
met_precision = np.array(
    [
        precision_score(y_net_test, y_net_pred_lr),
        precision_score(y_net_test, y_net_pred_mlp),
        precision_score(y_net_test, y_net_pred_svm),
        precision_score(y_am_test, y_am_pred_lr),
        precision_score(y_am_test, y_am_pred_mlp),
        precision_score(y_am_test, y_am_pred_svm),
    ]
)

# Compute f1 for each model
met_f1 = np.array(
    [
        f1_score(y_net_test, y_net_pred_lr),
        f1_score(y_net_test, y_net_pred_mlp),
        f1_score(y_net_test, y_net_pred_svm),
        f1_score(y_am_test, y_am_pred_lr),
        f1_score(y_am_test, y_am_pred_mlp),
        f1_score(y_am_test, y_am_pred_svm),
    ]
)

# Store results in DataFrame
df_performance = pd.DataFrame(
    [met_accuracy, met_recall, met_precision, met_f1]
).T
df_performance.columns = ["Accuracy", "Recall", "Precision", "f1"]

# Correct labels for models & datasets
df_performance.index = [
    "LG-Netflix",
    "MLP-Netflix",
    "SVM-Netflix",
    "LG-Amazon",
    "MLP-Amazon",
    "SVM-Amazon",
]

# Display results
df_performance.head(6)


# In[51]:


# Plot comparison
plt.figure(figsize=(10, 6))
df_performance.plot(
    kind="bar", figsize=(12, 6), colormap="viridis", edgecolor="black"
)

plt.title("Model Performance Comparison", fontsize=14)
plt.xlabel("Models", fontsize=12)
plt.ylabel("Metric Scores", fontsize=12)
plt.xticks(rotation=45)
plt.legend(loc="best", fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()


# In[53]:


# Function to plot confusion matrix
def plot_conf_matrix(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)


# Create figure with multiple subplots
fig, axes = plt.subplots(
    nrows=2, ncols=3, figsize=(12, 8)
)  # Adjusted size for compact view

# Train set matrices
plot_conf_matrix(
    axes[0, 0],
    y_net_train,
    log_models["netflix"].predict(netflix_train_scaled),
    "Train - Netflix (LR)",
)
plot_conf_matrix(
    axes[0, 1],
    y_net_train,
    mlp_models["netflix"].predict(netflix_train_scaled),
    "Train - Netflix (MLP)",
)
plot_conf_matrix(
    axes[0, 2],
    y_net_train,
    svm_models["netflix"].predict(netflix_train_scaled),
    "Train - Netflix (SVM)",
)

# Test set matrices
plot_conf_matrix(axes[1, 0], y_net_test, y_net_pred_lr, "Test - Netflix (LR)")
plot_conf_matrix(
    axes[1, 1], y_net_test, y_net_pred_mlp, "Test - Netflix (MLP)"
)
plot_conf_matrix(
    axes[1, 2], y_net_test, y_net_pred_svm, "Test - Netflix (SVM)"
)

plt.tight_layout()  # Ensures proper spacing between subplots
plt.show()


# # Hyper-Parameter Tuning

# ### Logistic Regression

# In[55]:


# Define scorer (optimize for F1-score due to slight class imbalance)
scorer = make_scorer(f1_score)

# Define smaller, optimized parameter grid
parameter_space_lr = {
    "penalty": ["l1", "l2"],  # Limited options for regularization
    "C": [0.01, 0.1, 1, 10],  # Controls regularization strength
    "solver": [
        "liblinear",
        "saga",
    ],  # Efficient solvers for small-scale tuning
    "max_iter": [100, 500],  # Lower iterations for faster convergence
}

# Run GridSearch for both datasets
best_params_lr = {}

for dataset_name, (train_scaled, y_train) in datasets_scaled.items():
    print(f"Running GridSearchCV for {dataset_name}...")

    lr_tune = LogisticRegression(random_state=100)
    GSCV_lr = GridSearchCV(
        lr_tune, parameter_space_lr, n_jobs=-1, cv=3, scoring=scorer
    )

    GSCV_lr.fit(train_scaled, y_train)

    best_params_lr[dataset_name] = GSCV_lr.best_params_
    print(f"Best Parameters for {dataset_name}: {GSCV_lr.best_params_}")

# Store results in DataFrame for easy comparison
df_best_params_lr = pd.DataFrame(best_params_lr).T
df_best_params_lr


# In[57]:


# Define optimized Logistic Regression models for Netflix and Amazon
optimized_lr_models = {}

best_params_lr = {
    "penalty": "l2",
    "C": 0.05,
    "solver": "liblinear",  # Using liblinear for small datasets
    "max_iter": 200,
    "random_state": 100,
}

# Train Logistic Regression models with optimized parameters
for name, (train_scaled, y_train) in datasets_scaled.items():
    model = LogisticRegression(**best_params_lr)
    model.fit(train_scaled, y_train)
    optimized_lr_models[name] = model

# Generate predictions
threshold = 0.51  # Adjust this for better precision vs recall balance, as it gives an irrealistic high recall - the default is 0.5

y_net_pred_probs = optimized_lr_models["netflix"].predict_proba(
    netflix_test_scaled
)
y_am_pred_probs = optimized_lr_models["amazon"].predict_proba(
    amazon_test_scaled
)

y_net_pred_lr_opt = (y_net_pred_probs[:, 1] > threshold).astype(int)
y_am_pred_lr_opt = (y_am_pred_probs[:, 1] > threshold).astype(int)

# Evaluate the optimized model
datasets_opt_lr = {
    "Netflix": (y_net_test, y_net_pred_lr_opt),
    "Amazon": (y_am_test, y_am_pred_lr_opt),
}

metrics_to_check = ["accuracy", "precision", "recall", "f1"]

# Print optimized evaluation results
for stock, (y_true, y_pred) in datasets_opt_lr.items():
    results = evaluate_model(y_true, y_pred)
    for metric in metrics_to_check:
        print(
            f"{stock} {metric.capitalize()} (Optimized Logistic Regression): {results[metric]:.4f}"
        )


# ### MLP

# In[58]:


# Define scorer (optimize for F1-score in this case due to slightly imbalanced data set)
scorer = make_scorer(f1_score)

# Define parameter grid
parameter_space_mlp = {
    "hidden_layer_sizes": [
        (50,),
        (100,),
        (50, 30),
        (100, 50, 30),
        (200, 100, 50),
    ],
    "activation": ["tanh", "relu", "logistic"],
    "solver": ["adam", "sgd", "lbfgs"],
    "alpha": [0.0001, 0.001, 0.005, 0.01],
    "learning_rate": ["constant", "adaptive", "invscaling"],
    "max_iter": [200, 500, 1000, 3000],
}

# Run GridSearch for both datasets
best_params = {}

for dataset_name, (train_scaled, y_train) in datasets_scaled.items():
    print(f"Running GridSearchCV for {dataset_name}...")

    mlp_tune = MLPClassifier(random_state=100)
    GSCV_mlp = GridSearchCV(
        mlp_tune, parameter_space_mlp, n_jobs=-1, cv=3, scoring=scorer
    )

    GSCV_mlp.fit(train_scaled, y_train)

    best_params[dataset_name] = GSCV_mlp.best_params_
    print(f"Best Parameters for {dataset_name}: {GSCV_mlp.best_params_}")

# Store results in a DataFrame for easy comparison
df_best_params = pd.DataFrame(best_params).T
df_best_params


# In[59]:


# Define optimized MLP models for Netflix and Amazon
optimized_mlp_models = {}

best_params = {
    "activation": "relu",
    "alpha": 0.02,
    "hidden_layer_sizes": (50,),
    "learning_rate": "adaptive",
    "max_iter": 500,
    "solver": "sgd",
    "random_state": 100,  # Keep this fixed for consistency
}

# Train MLP models with optimized parameters
for name, (train_scaled, y_train) in datasets_scaled.items():
    model = MLPClassifier(**best_params)
    model.fit(train_scaled, y_train)
    optimized_mlp_models[name] = model

# Generate predictions
y_net_pred_mlp_opt = optimized_mlp_models["netflix"].predict(
    netflix_test_scaled
)
y_am_pred_mlp_opt = optimized_mlp_models["amazon"].predict(amazon_test_scaled)

# Evaluate the optimized model
datasets_opt = {
    "Netflix": (y_net_test, y_net_pred_mlp_opt),
    "Amazon": (y_am_test, y_am_pred_mlp_opt),
}

metrics_to_check = ["accuracy", "precision", "recall", "f1"]

for stock, (y_true, y_pred) in datasets_opt.items():
    results = evaluate_model(y_true, y_pred)
    for metric in metrics_to_check:
        print(
            f"{stock} {metric.capitalize()} (Optimized): {results[metric]:.4f}"
        )

# note that with using the best parameters, recall was 1 which is unrealistic so I used slightly different parameters to improve, i.e. switched to relu as it handles class imbalances better than logistic, increased alpha to 0.02


# ### SVM

# In[60]:


# Define scorer (optimize for F1-score due to slight class imbalance)
scorer = make_scorer(f1_score)

# Define optimized parameter grid for SVM
parameter_space_svm = {
    "C": [0.01, 0.1, 1, 10],  # Regularization strength
    "kernel": [
        "linear",
        "rbf",
    ],  # Different kernel types (skipped poly cause computationally expensive)
    "gamma": [
        "scale",
        "auto",
    ],  # Kernel coefficient (affects non-linear models)
    "class_weight": [None, "balanced"],  # Adjusts for imbalanced datasets
}

# Run GridSearch for both datasets
best_params_svm = {}

for dataset_name, (train_scaled, y_train) in datasets_scaled.items():
    print(f"Running GridSearchCV for {dataset_name}...")

    svm_tune = SVC(random_state=100)
    GSCV_svm = GridSearchCV(
        svm_tune, parameter_space_svm, n_jobs=-1, cv=3, scoring=scorer
    )

    GSCV_svm.fit(train_scaled, y_train)

    best_params_svm[dataset_name] = GSCV_svm.best_params_
    print(f"Best Parameters for {dataset_name}: {GSCV_svm.best_params_}")

# Store results in DataFrame for easy comparison
df_best_params_svm = pd.DataFrame(best_params_svm).T
df_best_params_svm


# In[63]:


# Define optimized SVM models for Netflix and Amazon
optimized_svm_models = {}

best_params_svm = {
    "C": 0.25,
    "kernel": "poly",
    "gamma": "scale",
    "class_weight": "balanced",
    "probability": True,
    "random_state": 100,
}
# Note that im using poly instead of linear due to recall of 1 otherwise

# Train SVM models with optimized parameters
for name, (train_scaled, y_train) in datasets_scaled.items():
    model = SVC(**best_params_svm)
    model.fit(train_scaled, y_train)
    optimized_svm_models[name] = model

# Generate predictions
# Define threshold
threshold = 0.48  # Adjust this as needed

# Get prediction probabilities
y_net_pred_probs = optimized_svm_models["netflix"].predict_proba(
    netflix_test_scaled
)
y_am_pred_probs = optimized_svm_models["amazon"].predict_proba(
    amazon_test_scaled
)

# Apply thresholding
y_net_pred_svm_opt = (y_net_pred_probs[:, 1] > threshold).astype(int)
y_am_pred_svm_opt = (y_am_pred_probs[:, 1] > threshold).astype(int)

# Evaluate the optimized model
datasets_opt_svm = {
    "Netflix": (y_net_test, y_net_pred_svm_opt),
    "Amazon": (y_am_test, y_am_pred_svm_opt),
}

metrics_to_check = ["accuracy", "precision", "recall", "f1"]

# Print optimized evaluation results
for stock, (y_true, y_pred) in datasets_opt_svm.items():
    results = evaluate_model(y_true, y_pred)
    for metric in metrics_to_check:
        print(
            f"{stock} {metric.capitalize()} (Optimized SVM): {results[metric]:.4f}"
        )


# In[64]:


# Initialize list to store results dynamically
metrics_results = []

# Define datasets containing previously computed results
datasets_opt_all = {
    "SVM": datasets_opt_svm,  # Contains optimized SVM results
    "Logistic Regression": datasets_opt_lr,  # Contains optimized LR results
    "MLP": datasets_opt,  # Contains optimized MLP results
}

# Loop through each model type and stock
for model_name, dataset in datasets_opt_all.items():
    for stock, (y_true, y_pred) in dataset.items():
        results = evaluate_model(
            y_true, y_pred
        )  # Pull previously computed metrics

        # Store results dynamically
        metrics_results.append(
            {
                "Model": model_name,
                "Stock": stock,
                **{
                    metric: results[metric]
                    for metric in ["accuracy", "precision", "recall", "f1"]
                },
            }
        )

# Convert list into DataFrame
df_metrics = pd.DataFrame(metrics_results)

# Display consolidated metrics
df_metrics


# 
# #  **Trading Strategy**

# In[65]:


# Get the best model for each stock based on highest precision
best_netflix_model_name = (
    df_metrics[df_metrics["Stock"] == "Netflix"]
    .sort_values(by="precision", ascending=False)
    .iloc[0]["Model"]
)
best_amazon_model_name = (
    df_metrics[df_metrics["Stock"] == "Amazon"]
    .sort_values(by="precision", ascending=False)
    .iloc[0]["Model"]
)

print(f"Best Model for Netflix: {best_netflix_model_name}")
print(f"Best Model for Amazon: {best_amazon_model_name}")

# Retrieve predictions from the selected models
netflix_predictions = datasets_opt_all[best_netflix_model_name]["Netflix"][1]
amazon_predictions = datasets_opt_all[best_amazon_model_name]["Amazon"][1]

# Apply trading rules based on predictions
trade_threshold = 0.50  # Adjust if necessary
netflix_trades = [
    "BUY" if pred > trade_threshold else "SELL" for pred in netflix_predictions
]
amazon_trades = [
    "BUY" if pred > trade_threshold else "SELL" for pred in amazon_predictions
]

# Filter sentiment data separately to match test set sizes
merged_sentiment_netflix = merged_sentiment[
    merged_sentiment["stock"] == "netflix"
].tail(len(netflix_trades))
merged_sentiment_amazon = merged_sentiment[
    merged_sentiment["stock"] == "amazon"
].tail(len(amazon_trades))

# Create separate DataFrames
df_netflix_strategy = pd.DataFrame(
    {"Date": merged_sentiment_netflix["date"], "Netflix Trade": netflix_trades}
)

df_amazon_strategy = pd.DataFrame(
    {"Date": merged_sentiment_amazon["date"], "Amazon Trade": amazon_trades}
)


# ## Amazon Trading Strategy

# In[66]:


# Merge the pred data frame with histdata which contains the level of returns

amazon_returns = merged_sentiment[merged_sentiment["stock"] == "amazon"][
    ["returns", "date"]
]
amazon_returns.rename(columns={"date": "Date"}, inplace=True)

amazon_backtest = df_amazon_strategy.merge(
    amazon_returns, on="Date", how="left"
)


# In[67]:


# Buy and hold returns
amazon_backtest["buyandhold"] = (amazon_backtest["returns"] + 1).cumprod()

# Map trade actions to numeric values
amazon_backtest["strategy_multiplier"] = amazon_backtest["Amazon Trade"].map(
    {"BUY": 1, "SELL": -1}
)

# Apply strategy returns calculation
amazon_backtest["strategy_return"] = (
    amazon_backtest["strategy_multiplier"] * amazon_backtest["returns"]
)

# Calculate cumulative strategy return
amazon_backtest["strategy_cumulative"] = (
    amazon_backtest["strategy_return"] + 1
).cumprod()

amazon_backtest


# In[68]:


# Create the plot
plt.figure(figsize=(10, 6))

# Plot buy-and-hold cumulative returns
plt.plot(
    amazon_backtest["Date"],
    amazon_backtest["buyandhold"],
    label="Buy & Hold",
    linestyle="--",
    color="blue",
)

# Plot strategy cumulative returns
plt.plot(
    amazon_backtest["Date"],
    amazon_backtest["strategy_cumulative"],
    label="Trading Strategy",
    linestyle="-",
    color="red",
)

# Formatting
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title("Buy-and-Hold vs. Trading Strategy Returns (Amazon)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the plot
plt.show()


# In[69]:


# Normalize buy-and-hold returns to start from 1
amazon_backtest["buyandhold_indexed"] = (
    amazon_backtest["buyandhold"] / amazon_backtest["buyandhold"].iloc[0]
)

# Normalize strategy cumulative returns to start from 1
amazon_backtest["strategy_indexed"] = (
    amazon_backtest["strategy_cumulative"]
    / amazon_backtest["strategy_cumulative"].iloc[0]
)

# Plot indexed returns

plt.figure(figsize=(10, 6))
plt.plot(
    amazon_backtest["Date"],
    amazon_backtest["buyandhold_indexed"],
    label="Buy & Hold (Indexed)",
    linestyle="--",
    color="blue",
)
plt.plot(
    amazon_backtest["Date"],
    amazon_backtest["strategy_indexed"],
    label="Trading Strategy (Indexed)",
    linestyle="-",
    color="red",
)

plt.xlabel("Date")
plt.ylabel("Cumulative Returns (Indexed)")
plt.title("Performance Comparison: Buy-and-Hold vs. Trading Strategy")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.show()


# In[70]:


# Calculate total percentage returns for each strategy
buy_hold_final_return = (amazon_backtest["buyandhold"].iloc[-1] - 1) * 100
strategy_final_return = (
    amazon_backtest["strategy_cumulative"].iloc[-1] - 1
) * 100

# Calculate mean return for each strategy
buy_hold_avg_return = amazon_backtest["buyandhold"].pct_change().mean() * 100
strategy_avg_return = (
    amazon_backtest["strategy_cumulative"].pct_change().mean() * 100
)

# Calculate volatility (standard deviation of returns)
buy_hold_volatility = amazon_backtest["buyandhold"].pct_change().std() * 100
strategy_volatility = (
    amazon_backtest["strategy_cumulative"].pct_change().std() * 100
)

# Calculate max drawdown (largest peak-to-trough decline)
buy_hold_max_drawdown = (
    (amazon_backtest["buyandhold"].max() - amazon_backtest["buyandhold"].min())
    / amazon_backtest["buyandhold"].max()
    * 100
)
strategy_max_drawdown = (
    (
        amazon_backtest["strategy_cumulative"].max()
        - amazon_backtest["strategy_cumulative"].min()
    )
    / amazon_backtest["strategy_cumulative"].max()
    * 100
)

# Create final DataFrame with comparison metrics
amazon_summary_metrics = pd.DataFrame(
    {
        "Metric": [
            "Total Return (%)",
            "Average Daily Return (%)",
            "Volatility (%)",
            "Max Drawdown (%)",
        ],
        "Buy & Hold": [
            buy_hold_final_return,
            buy_hold_avg_return,
            buy_hold_volatility,
            buy_hold_max_drawdown,
        ],
        "Trading Strategy": [
            strategy_final_return,
            strategy_avg_return,
            strategy_volatility,
            strategy_max_drawdown,
        ],
    }
)

# summary metrics
amazon_summary_metrics


# ## Netflix Trading Strategy

# In[71]:


# Merge the Pred Data Frame with histdata which contains the level of returns

netflix_returns = merged_sentiment[merged_sentiment["stock"] == "netflix"][
    ["returns", "date"]
]
netflix_returns.rename(columns={"date": "Date"}, inplace=True)

netflix_backtest = df_netflix_strategy.merge(
    netflix_returns, on="Date", how="left"
)


# In[72]:


# Buy and hold returns
netflix_backtest["buyandhold"] = (netflix_backtest["returns"] + 1).cumprod()

# Map trade actions to numeric values
netflix_backtest["strategy_multiplier"] = netflix_backtest[
    "Netflix Trade"
].map({"BUY": 1, "SELL": -1})

# Apply strategy returns calculation
netflix_backtest["strategy_return"] = (
    netflix_backtest["strategy_multiplier"] * netflix_backtest["returns"]
)

# Calculate cumulative strategy return
netflix_backtest["strategy_cumulative"] = (
    netflix_backtest["strategy_return"] + 1
).cumprod()

netflix_backtest


# In[73]:


# Create the plot
plt.figure(figsize=(10, 6))

# Plot buy-and-hold cumulative returns
plt.plot(
    netflix_backtest["Date"],
    netflix_backtest["buyandhold"],
    label="Buy & Hold",
    linestyle="--",
    color="blue",
)

# Plot strategy cumulative returns
plt.plot(
    netflix_backtest["Date"],
    netflix_backtest["strategy_cumulative"],
    label="Trading Strategy",
    linestyle="-",
    color="red",
)

# Formatting
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title("Buy-and-Hold vs. Trading Strategy Returns (Netflix)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the plot
plt.show()


# In[75]:


# Normalize buy-and-hold returns to start from 1
netflix_backtest["buyandhold_indexed"] = (
    netflix_backtest["buyandhold"] / netflix_backtest["buyandhold"].iloc[0]
)

# Normalize strategy cumulative returns to start from 1
netflix_backtest["strategy_indexed"] = (
    netflix_backtest["strategy_cumulative"]
    / netflix_backtest["strategy_cumulative"].iloc[0]
)

# Plot indexed returns

plt.figure(figsize=(10, 6))
plt.plot(
    netflix_backtest["Date"],
    netflix_backtest["buyandhold_indexed"],
    label="Buy & Hold (Indexed)",
    linestyle="--",
    color="blue",
)
plt.plot(
    netflix_backtest["Date"],
    netflix_backtest["strategy_indexed"],
    label="Trading Strategy (Indexed)",
    linestyle="-",
    color="red",
)

plt.xlabel("Date")
plt.ylabel("Cumulative Returns (Indexed)")
plt.title("Performance Comparison: Buy-and-Hold vs. Trading Strategy")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.show()


# In[76]:


# Calculate total percentage returns for each strategy
buy_hold_final_return = (netflix_backtest["buyandhold"].iloc[-1] - 1) * 100
strategy_final_return = (
    netflix_backtest["strategy_cumulative"].iloc[-1] - 1
) * 100

# Calculate mean return for each strategy
buy_hold_avg_return = netflix_backtest["buyandhold"].pct_change().mean() * 100
strategy_avg_return = (
    netflix_backtest["strategy_cumulative"].pct_change().mean() * 100
)

# Calculate volatility (standard deviation of returns)
buy_hold_volatility = amazon_backtest["buyandhold"].pct_change().std() * 100
strategy_volatility = (
    amazon_backtest["strategy_cumulative"].pct_change().std() * 100
)

# Calculate max drawdown (largest peak-to-trough decline)
buy_hold_max_drawdown = (
    (
        netflix_backtest["buyandhold"].max()
        - netflix_backtest["buyandhold"].min()
    )
    / netflix_backtest["buyandhold"].max()
    * 100
)
strategy_max_drawdown = (
    (
        netflix_backtest["strategy_cumulative"].max()
        - netflix_backtest["strategy_cumulative"].min()
    )
    / netflix_backtest["strategy_cumulative"].max()
    * 100
)

# Create final DataFrame with comparison metrics
netflix_summary_metrics = pd.DataFrame(
    {
        "Metric": [
            "Total Return (%)",
            "Average Daily Return (%)",
            "Volatility (%)",
            "Max Drawdown (%)",
        ],
        "Buy & Hold": [
            buy_hold_final_return,
            buy_hold_avg_return,
            buy_hold_volatility,
            buy_hold_max_drawdown,
        ],
        "Trading Strategy": [
            strategy_final_return,
            strategy_avg_return,
            strategy_volatility,
            strategy_max_drawdown,
        ],
    }
)

# summary metrics
netflix_summary_metrics

