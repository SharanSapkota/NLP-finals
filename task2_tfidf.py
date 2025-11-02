# task2_tfidf.py

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Download required NLTK data (only first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------
# Preprocessing function
# ---------------------------
def preprocess(text):
    """
    Tokenize, remove stopwords, lemmatize and return cleaned text.
    """
    # Tokenize
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return " ".join(tokens)

# ---------------------------
# TF-IDF similarity function
# ---------------------------
def compute_tfidf_similarity(wiki_data):
    """
    Given a dictionary of Wikipedia data (keyword -> {text, sections, entities}),
    preprocess the text, compute TF-IDF vectors, and return cosine similarity dataframe.
    """
    # Preprocess full text of each document
    preprocessed_texts = {k: preprocess(v["text"]) for k, v in wiki_data.items()}
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit-transform all documents
    tfidf_matrix = vectorizer.fit_transform(preprocessed_texts.values())
    
    # Compute cosine similarity
    cos_sim = cosine_similarity(tfidf_matrix)
    
    # Return as pandas DataFrame
    df = pd.DataFrame(cos_sim, index=wiki_data.keys(), columns=wiki_data.keys())
    return df

# Run on the file here 
if __name__ == "__main__":
    import task1_wiki as t1
    from IPython.display import display
    
    # Get Wikipedia data for keywords
    wiki_data = {k: t1.get_wiki_content(k) for k in t1.keywords}
    
    # Compute TF-IDF similarity
    tfidf_sim_full = compute_tfidf_similarity(wiki_data)
    
    print("TF-IDF Cosine Similarity (Full Text):")
    display(tfidf_sim_full)


# comment on this
#  The TF-IDF cosine similarity matrix indicates that documents X and Y share the highest similarity, suggesting overlapping topics or vocabulary, whereas A and B are the least similar, indicating distinct subject areas