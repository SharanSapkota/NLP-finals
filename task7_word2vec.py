# task7_word2vec.py
from gensim.models import KeyedVectors
import pandas as pd
import gensim.downloader as api

def compute_word2vec_similarity(words):
    """
    Use pretrained Word2Vec embeddings to compute pairwise similarity.
    """
    model = api.load("word2vec-google-news-300")  # Pretrained model
    sim_matrix = []
    for w1 in words:
        row = []
        for w2 in words:
            try:
                sim = model.similarity(w1, w2)
            except KeyError:
                sim = 0  # Word not in vocabulary
            row.append(sim)
        sim_matrix.append(row)
    df = pd.DataFrame(sim_matrix, index=words, columns=words)
    return df

# Test run
if __name__ == "__main__":
    keywords = ["nature", "resilience", "sustainability", "climate change"]
    df = compute_word2vec_similarity(keywords)
    print("Word2Vec similarity:")
    print(df)
