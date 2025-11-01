# task6_entity_scraper.py
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from task2_tfidf import preprocess
import time

def scrape_entity_content(wiki_data):
    """
    For each entity in wiki_data, fetch its Wikipedia page content (first pass).
    """
    entity_content = {}
    for keyword, data in wiki_data.items():
        entity_texts = []
        for entity in data["entities"][:50]:  # Limit to first 50 entities for demo
            url = f"https://en.wikipedia.org/wiki/{entity.replace(' ', '_')}"
            try:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, "html.parser")
                # Get main content paragraph
                p_text = " ".join([p.get_text() for p in soup.find_all("p")])
                entity_texts.append(p_text)
                time.sleep(0.5)  # polite delay
            except Exception:
                continue
        entity_content[keyword] = " ".join(entity_texts)
    return entity_content

def compute_tfidf_entity_content(entity_content):
    preprocessed = {k: preprocess(v) for k, v in entity_content.items()}
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed.values())
    cos_sim = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(cos_sim, index=entity_content.keys(), columns=entity_content.keys())

# Test run
if __name__ == "__main__":
    import task1_wiki as t1
    wiki_data = {k: t1.get_wiki_content(k) for k in t1.keywords}
    entity_content = scrape_entity_content(wiki_data)
    df = compute_tfidf_entity_content(entity_content)
    print("TF-IDF similarity of entity content:")
    print(df)
