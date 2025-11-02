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

if __name__ == "__main__":
    import task1_wiki as t1
    wiki_data = {k: t1.get_wiki_content(k) for k in t1.keywords}

    tfidf_entities_sim = compute_tfidf_entities_similarity(wiki_data)
    print("TF-IDF Cosine Similarity (Entities):")
    print(tfidf_entities_sim)

