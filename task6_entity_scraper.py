import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from task2_tfidf import preprocess
import time

def scrape_entity_content(wiki_data, max_entities=5):
    """
    For each keyword in wiki_data, fetch content of its entities (first pass).
    Returns a dict: {keyword: combined text of entities}.
    """
    entity_content = {}
    headers = {
        "User-Agent": "MyWikiScraperBot/1.0 (your_email@example.com)"  
    }

    for keyword, data in wiki_data.items():
        entity_texts = []
        for entity in data.get("entities", [])[:max_entities]:
            url = f"https://en.wikipedia.org/wiki/{entity.replace(' ', '_')}" 
            try:
                r = requests.get(url, headers=headers, timeout=10)
                print(r)
                if r.status_code != 200:
                    continue
                soup = BeautifulSoup(r.text, "html.parser")
                p_text = " ".join([p.get_text() for p in soup.find_all("p")])
                if p_text.strip():
                    entity_texts.append(p_text)
                time.sleep(0.5)  # polite delay
            except requests.exceptions.RequestException:
                continue
        entity_content[keyword] = " ".join(entity_texts)
    return entity_content

def scrape_entity_categories(wiki_data, max_entities=10):
    """
    For each keyword, fetch categories of its entities (first pass).
    Returns a dict: {keyword: list of categories}.
    """
    entity_categories = {}
    for keyword, data in wiki_data.items():
        categories = []
        for entity in data.get("entities", [])[:max_entities]:
            url = f"https://en.wikipedia.org/wiki/{entity.replace(' ', '_')}"
            try:
                r = requests.get(url)
                soup = BeautifulSoup(r.text, "html.parser")
                cats = [cat.get_text() for cat in soup.select("#mw-normal-catlinks ul li")]
                categories.extend(cats)
                time.sleep(0.5)
            except Exception:
                continue
        entity_categories[keyword] = categories
    return entity_categories


def compute_tfidf_entity_content(entity_content):
    preprocessed = {k: " ".join(preprocess(v)) for k, v in entity_content.items() if preprocess(v)}
    if not preprocessed:
        raise ValueError("No valid entity content after preprocessing.")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed.values())
    cos_sim = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(cos_sim, index=preprocessed.keys(), columns=preprocessed.keys())

def compute_tfidf_entity_categories(entity_categories):
    preprocessed = {k: " ".join(preprocess(" ".join(v))) for k, v in entity_categories.items() if v}
    if not preprocessed:
        raise ValueError("No valid entity categories found after preprocessing.")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed.values())
    cos_sim = cosine_similarity(tfidf_matrix)
    return pd.DataFrame(cos_sim, index=preprocessed.keys(), columns=preprocessed.keys())

def scrape_entity_content_and_categories(wiki_data, max_entities=50):
    """
    Returns both entity content and entity categories dicts.
    """
    content = scrape_entity_content(wiki_data, max_entities)
    categories = scrape_entity_categories(wiki_data, max_entities)
    return content, categories

if __name__ == "__main__":
    import task1_wiki as t1
    wiki_data = {k: t1.get_wiki_content(k) for k in t1.keywords}

    entity_content, entity_categories = scrape_entity_content_and_categories(wiki_data)

    content_sim_df = compute_tfidf_entity_content(entity_content)
    category_sim_df = compute_tfidf_entity_categories(entity_categories)

    print(content_sim_df)
    print(category_sim_df)
