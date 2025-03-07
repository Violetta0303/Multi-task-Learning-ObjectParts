import pandas as pd
import requests
import json
import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet')

# Load the term lists from the CSLB and THINGS datasets
cslb_df = pd.read_csv('..\\CSLB\\updated_feature_matrix.csv')
things_df = pd.read_csv('THINGS.csv')

# Extract the list of terms
cslb_terms = cslb_df.iloc[:, 0].tolist()
things_terms = things_df.iloc[:, 2].tolist()

def get_synonyms_from_conceptnet(term):
    url = f"http://api.conceptnet.io/query?node=/c/en/{term}&rel=/r/Synonym"
    response = requests.get(url)
    data = response.json()
    synonyms = set()
    for edge in data['edges']:
        synonym = edge['end']['label']
        if synonym != term and synonym in things_terms:
            synonyms.add(synonym)
    return list(synonyms)

def get_synonyms_from_wordnet(term):
    synonyms = set()
    for syn in wn.synsets(term):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ').lower()
            if synonym != term and synonym in things_terms:
                synonyms.add(synonym)
    return list(synonyms)

def get_synonyms_from_datamuse(term):
    synonyms = set()
    url = f"https://api.datamuse.com/words?rel_syn={term}"
    response = requests.get(url)
    data = response.json()
    for item in data:
        synonym = item['word'].replace('_', ' ').lower()
        if synonym != term and synonym in things_terms:
            synonyms.add(synonym)
    return list(synonyms)

def find_mappings(cslb_list):
    mappings = {}
    for index, term in enumerate(cslb_list):
        print(f"Processing {index + 1}/{len(cslb_list)}: {term}")
        synonyms = set(get_synonyms_from_conceptnet(term) + get_synonyms_from_wordnet(term) + get_synonyms_from_datamuse(term))
        if synonyms:
            mappings[term] = list(synonyms)[0]
    return mappings

def find_exact_matches(cslb_list, things_list):
    exact_matches = {}
    for term in cslb_list:
        if term in things_list:
            exact_matches[term] = term
    return exact_matches

# Find mappings using synonyms
auto_mappings = find_mappings(cslb_terms)

# Find exact matches
exact_mappings = find_exact_matches(cslb_terms, things_terms)

# Manual mappings section (add as needed)
manual_mappings = {
    # Add more manual mappings if necessary
}

# Combine automatic, exact, and manual mappings into namemap
namemap = {**auto_mappings, **exact_mappings, **manual_mappings}

# Save namemap as JSON file
with open('namemap.json', 'w') as file:
    json.dump(namemap, file, indent=4)

print("Namemap saved to namemap.json")
