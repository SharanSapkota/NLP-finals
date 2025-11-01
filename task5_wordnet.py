# task5_wordnet.py
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd

nltk.download('wordnet')
nltk.download('omw-1.4')

def wu_palmer_similarity(word_list):
    """
    Compute pairwise Wu & Palmer similarity for a list of words using WordNet.
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
            row.append(max_sim)
        sim_matrix.append(row)
    
    df = pd.DataFrame(sim_matrix, index=word_list, columns=word_list)
    return df

def compute_wordnet_similarity(keywords):
    return wu_palmer_similarity(keywords)

# Test run
if __name__ == "__main__":
    keywords = ["nature", "resilience", "sustainability", "climate change"]
    df = compute_wordnet_similarity(keywords)
    print("Wu & Palmer WordNet similarity:")
    print(df)
