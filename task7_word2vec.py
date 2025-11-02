# task7_word_similarity.py

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import gensim.downloader as api
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

def get_keyword_words(keywords):
    words = []
    for kw in keywords:
        words.extend(preprocess(kw))
    return words

def compute_tfidf_word_similarity(words):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(words)
    sim_matrix = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(sim_matrix, index=words, columns=words)

def compute_w2v_word_similarity(words):
    w2v_model = api.load("word2vec-google-news-300")
    sim_matrix = []
    for w1 in words:
        row = []
        for w2 in words:
            try:
                sim = w2v_model.similarity(w1, w2)
            except KeyError:
                sim = 0
            row.append(sim)
        sim_matrix.append(row)
    return pd.DataFrame(sim_matrix, index=words, columns=words)


if __name__ == "__main__":
    keywords = ["nature", "resilience", "sustainability", "climate change"]

    keyword_words = get_keyword_words(keywords)
    print("Preprocessed words:", keyword_words)

    tfidf_df = compute_tfidf_word_similarity(keyword_words)
    print("\nTF-IDF Cosine Similarity (words):")
    print(tfidf_df)

    w2v_df = compute_w2v_word_similarity(keyword_words)
    print("\nWord2Vec Similarity (words):")
    print(w2v_df)
