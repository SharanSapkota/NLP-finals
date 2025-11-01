# task1.py
import wikipediaapi
from bs4 import BeautifulSoup
import requests

keywords = ["nature", "resilience", "sustainability", "climate change"]

# Correct module-level Wikipedia object
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='EnvironmentKeywordMapping/1.0 (University Project)'
)

def get_wiki_content(keyword):
    page = wiki_wiki.page(keyword)
    if not page.exists():
        return None
    sections = [s.title for s in page.sections]
    entities = list(page.links.keys())
    text = page.text
    fullurl = page.fullurl
    print(page.fullurl)
    return {
        "text": text,
        "sections": sections,
        "entities": entities,
        'url': fullurl,

    }
wiki_data = {}
for k in keywords:
    wiki_data[k] = get_wiki_content(k)  # reuse the same API object
    # time.sleep(1)