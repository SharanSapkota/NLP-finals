# task5_wordnet.py

import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np

# Download WordNet data (only needed once)
nltk.download('wordnet')
nltk.download('omw-1.4')

# wu & Palmer semantic similarity
def wu_palmer_similarity(word_list):
    """
    Compute pairwise Wu & Palmer similarity for a list of words using WordNet.
    Returns a pandas DataFrame (similarity matrix).
    """
    sim_matrix = []

    for w1 in word_list:
        row = []
        syn1 = wn.synsets(w1)
        for w2 in word_list:
            syn2 = wn.synsets(w2)
            # Take max similarity between all synset pairs
            max_sim = 0
            for s1 in syn1:
                for s2 in syn2:
                    sim = s1.wup_similarity(s2)
                    if sim is not None and sim > max_sim:
                        max_sim = sim
            row.append(max_sim if max_sim > 0 else 0)
        sim_matrix.append(row)

    df = pd.DataFrame(sim_matrix, index=word_list, columns=word_list)
    return df

# Step 2: Helper function to flatten similarity matrices
def flatten_upper_triangle(df):
    """
    Convert the upper triangle of a similarity DataFrame into a 1D vector
    (excluding diagonal elements).
    """
    vals = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            vals.append(df.iloc[i, j])
    return np.array(vals)

# --------------------------------------------
# Step 3: Compute correlation with Wikipedia-based similarities
# --------------------------------------------
def compute_correlation_with_wikipedia_similarities(keywords):
    import task1_wiki as t1
    import task2_tfidf as t2
    import task3_titles as t3
    import task4_entities as t4

    # Step 3.1 - Load Wikipedia data
    wiki_data = {k: t1.get_wiki_content(k) for k in keywords}

    # Step 3.2 - Compute Wikipedia-based similarities
    sim_full = t2.compute_tfidf_similarity(wiki_data)
    sim_titles = t3.compute_tfidf_titles_similarity(wiki_data)
    sim_entities = t4.compute_tfidf_entities_similarity(wiki_data)

    # Step 3.3 - Compute WordNet Wu–Palmer similarity
    sim_wordnet = wu_palmer_similarity(keywords)

    # Step 3.4 - Flatten all matrices
    vec_wordnet = flatten_upper_triangle(sim_wordnet)
    vec_full = flatten_upper_triangle(sim_full)
    vec_titles = flatten_upper_triangle(sim_titles)
    vec_entities = flatten_upper_triangle(sim_entities)

    # Step 3.5 - Compute Pearson correlations
    correlations = {
        "Text vs WordNet": np.corrcoef(vec_full, vec_wordnet)[0, 1],
        "Titles vs WordNet": np.corrcoef(vec_titles, vec_wordnet)[0, 1],
        "Entities vs WordNet": np.corrcoef(vec_entities, vec_wordnet)[0, 1],
    }

    # Display results
    print("\n=== Wu & Palmer WordNet Semantic Similarity Matrix ===")
    print(sim_wordnet)
    print("\n=== Correlation between WordNet and Wikipedia-based similarities ===")
    for k, v in correlations.items():
        print(f"{k}: {v:.3f}")

    # Optionally return all data
    return sim_wordnet, correlations

# --------------------------------------------
# Step 4: Run as standalone script
# --------------------------------------------
if __name__ == "__main__":
    keywords = ["nature", "resilience", "sustainability", "climate change"]

    sim_wordnet, correlations = compute_correlation_with_wikipedia_similarities(keywords)
