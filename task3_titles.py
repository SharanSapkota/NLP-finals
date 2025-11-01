# task3_titles.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from task2_tfidf import preprocess

def compute_tfidf_titles_similarity(wiki_data):
    preprocessed_titles = {k: preprocess(" ".join(v["sections"])) for k, v in wiki_data.items()}
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_titles.values())
    cos_sim = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(cos_sim, index=wiki_data.keys(), columns=wiki_data.keys())
