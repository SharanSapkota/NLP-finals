# task4_entities.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from task2_tfidf import preprocess

def compute_tfidf_entities_similarity(wiki_data):
    preprocessed_entities = {k: preprocess(" ".join(v["entities"])) for k, v in wiki_data.items()}
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_entities.values())
    cos_sim = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(cos_sim, index=wiki_data.keys(), columns=wiki_data.keys())
