import os
from dotenv import load_dotenv
import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import time
import gensim.downloader as api
from task2_tfidf import preprocess

load_dotenv()
API_KEY = os.getenv("NEWSAPI_KEY")
if not API_KEY:
    raise ValueError("Please set your NEWSAPI_KEY in the .env file.")

print("Loading Word2Vec model (this may take a while)...")
w2v_model = api.load("word2vec-google-news-300")  

keywords = ["nature", "resilience", "sustainability", "climate change"]

def get_time_periods(months=1):
    """Return list of (start_date, end_date) strings for each month period."""
    end_date = datetime.now()
    periods = []
    for i in range(months):
        start = (end_date - timedelta(days=30*(i+1))).strftime("%Y-%m-%d")
        end = (end_date - timedelta(days=30*i)).strftime("%Y-%m-%d")
        periods.append((start, end))
    return periods[::-1]  # earliest to latest

# ------------------- Fetch News -------------------
def fetch_news(keywords, from_to_period):
    """Fetch news for given keywords and period. Returns DataFrame."""
    all_articles = []
    from_param, to_param = from_to_period

    for kw in keywords:
        url = (
            f"https://newsapi.org/v2/everything?"
            f"q={kw}&from={from_param}&to={to_param}&language=en&"
            f"sortBy=relevancy&pageSize=100&apiKey={API_KEY}"
        )
        r = requests.get(url)
        data = r.json()
        if data.get("status") == "ok":
            articles = data.get("articles", [])
            for article in articles:
                all_articles.append({
                    "keyword": kw,
                    "from_date": from_param,
                    "to_date": to_param,
                    "date": article.get("publishedAt", ""),
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "content": article.get("content", "")
                })
        else:
            print(f"Error fetching '{kw}' ({from_param} to {to_param}): {data.get('message')}")
        time.sleep(1)

    df = pd.DataFrame(all_articles)
    return df

# ------------------- Generate WordCloud -------------------
def generate_wordcloud(df, period_label):
    """Generate WordClouds for each keyword in a DataFrame."""
    if df.empty:
        print(f"No news found for period {period_label}, skipping WordCloud.")
        return

    for kw in df["keyword"].unique():
        subset = df[df["keyword"] == kw]
        if subset.empty:
            continue
        text = " ".join(subset["content"].dropna().tolist())
        if not text.strip():
            continue
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"WordCloud for '{kw}' ({period_label})")
        plt.show()

# ------------------- Compute TF-IDF Similarity -------------------
def compute_tfidf_similarity(df):
    """Compute similarity between keywords using TF-IDF and cosine similarity."""

    # Step 1: Check if the DataFrame is empty
    if df.empty:
        print("No data available!")
        return pd.DataFrame()

    # Step 2: Group all articles by keyword
    # For each keyword, combine all the articles' content into one big string
    grouped_contents = {}
    for keyword in df['keyword'].unique():
        # Select articles for this keyword
        articles = df[df['keyword'] == keyword]['content']
        # Remove any empty content and combine into a single string
        combined_text = " ".join(articles.dropna())
        grouped_contents[keyword] = combined_text
        print(f"Keyword '{keyword}': {len(combined_text.split())} words combined")

    # Step 3: Preprocess the combined text (cleaning, lowercasing, etc.)
    preprocessed_contents = {}
    for keyword, text in grouped_contents.items():
        preprocessed_contents[keyword] = preprocess(text)

    # Step 4: Convert the preprocessed text into TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=2000)
    tfidf_matrix = vectorizer.fit_transform(preprocessed_contents.values())

    # Step 5: Compute cosine similarity between each keyword
    cos_sim = cosine_similarity(tfidf_matrix)

    # Step 6: Create a nice table (DataFrame) to show similarity scores
    sim_df = pd.DataFrame(cos_sim, index=preprocessed_contents.keys(), columns=preprocessed_contents.keys())

    # Step 7: Return the similarity table
    return sim_df

# ------------------- Compute Word2Vec Similarity -------------------
def compute_w2v_similarity(df):
    """Compute similarity between keywords using Word2Vec embeddings."""

    # Step 1: Check if the DataFrame is empty
    if df.empty:
        print("No data available!")
        return pd.DataFrame()

    # Step 2: Group all articles by keyword
    # For each keyword, combine all articles' content into a single string
    grouped_contents = {}
    for keyword in df['keyword'].unique():
        articles = df[df['keyword'] == keyword]['content']
        combined_text = " ".join(articles.dropna())
        grouped_contents[keyword] = combined_text
        print(f"Keyword '{keyword}': {len(combined_text.split())} words combined")

    # Step 3: Preprocess the text
    preprocessed_contents = {}
    for keyword, text in grouped_contents.items():
        preprocessed_contents[keyword] = preprocess(text)

    # Step 4: Prepare a similarity matrix
    keywords_list = list(preprocessed_contents.keys())
    sim_matrix = pd.DataFrame(index=keywords_list, columns=keywords_list, dtype=float)

    # Step 5: Compute Word2Vec similarity
    for i, kw1 in enumerate(keywords_list):
        # Get words for the first keyword that exist in the Word2Vec model
        words1 = [w for w in preprocessed_contents[kw1].split() if w in w2v_model.key_to_index]

        for j, kw2 in enumerate(keywords_list):
            # Get words for the second keyword
            words2 = [w for w in preprocessed_contents[kw2].split() if w in w2v_model.key_to_index]

            # If both have valid words in Word2Vec, compute similarity
            if words1 and words2:
                sim = w2v_model.n_similarity(words1, words2)
            else:
                sim = 0  # If no words found, similarity is 0

            # Store similarity in the matrix
            sim_matrix.loc[kw1, kw2] = sim

    # Step 6: Return the similarity matrix
    return sim_matrix

# ------------------- Main -------------------
if __name__ == "__main__":
    periods = get_time_periods(months=1)  # free plan limit

    all_data = []
    for start, end in periods:
        period_label = f"{start}_to_{end}"
        print(f"Fetching news for period: {period_label}")

        df_period = fetch_news(keywords, (start, end))
        if df_period.empty:
            print(f"No news found for period {period_label}, skipping.")
            continue

        df_period.to_csv(f"news_dataset_{period_label}.csv", index=False)
        generate_wordcloud(df_period, period_label)

        tfidf_sim = compute_tfidf_similarity(df_period)
        tfidf_sim.to_csv(f"tfidf_similarity_{period_label}.csv")

        w2v_sim = compute_w2v_similarity(df_period)
        w2v_sim.to_csv(f"w2v_similarity_{period_label}.csv")

        all_data.append(df_period)

    # Combine all periods into one dataset
    if all_data:
        df_all = pd.concat(all_data, ignore_index=True)
        df_all.to_csv("news_dataset_all_periods.csv", index=False)
