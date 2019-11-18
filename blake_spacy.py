import itertools
import re
import pandas as pd

# NLTK imports
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Spacy imports
import spacy

nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser"])
from spacy import displacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from spacy.pipeline import EntityRuler

# Graph imports
import matplotlib.pyplot as plt
import networkx as nx

# NLTK stopwords
stop_words = stopwords.words('english')
stop_words.extend(['thy', 'thou', 'thee', 'till', 'every', 'shall', 'like', 'every'])

blake_poems = gutenberg.sents('blake-poems.txt')


# Prepare text
# Segment by poem (originally, the whole txt doc is divided by line)  # TODO + remove the book of thel
def is_current_line_a_title(line):
    list_of_numbers = ['I', 'II', 'III']
    return all(word.isupper() for word in line if word not in list_of_numbers)


def chunk_poems(docu):
    list_of_poems = []
    current_poem = []

    for listed_line in docu:
        if is_current_line_a_title(listed_line):
            list_of_poems.append(current_poem.copy())
            current_poem.clear()

        current_poem.extend(listed_line)

    return list_of_poems


all_poems = chunk_poems(blake_poems)


# Tokenize, lemmatize, and clean text
def clean_text(poem):
    tokenized_text = []
    for word in poem:
        if word not in tokenized_text:
            word = lemmatizer.lemmatize(word)
            tokenized_text.extend(word_tokenize(word.lower()))

    return [
        elem for elem in tokenized_text
        if elem not in stop_words and
        re.match(
            '[a-zA-Z\-][a-zA-Z\-]{2,}',
            elem
        )
    ]


# Tokenize data
def tokenize_data(text):
    tokenized_data = []
    for poem in text:
        tokenized_data.append(clean_text(poem))
    return tokenized_data


tokenized_data = tokenize_data(all_poems)
poems_of_innocence = tokenized_data[:17]
poems_of_experience = tokenized_data[17:]

# Spacy entity recognition
innocence_text = " ".join([val for sublist in tokenized_data[:17] for val in sublist if val.isalnum()])
experience_text = " ".join([val for sublist in tokenized_data[17:] for val in sublist if val.isalnum()])

matcher = Matcher(nlp.vocab)

# Innocence matcher # TODO add more and group by symbolism
ENTITY_MAPPING = {
    "animals": ["lamb", "tiger", "sheep", "dove", "grasshopper"],
    "bucolic_symbolism": ["shepherd"],
    "religious_figures": ["angel", ],
    "emotions": ["joy", "merry", "cheer", "happy", "wept", "smile"],
    "colors": ["green"],
    "times_of_the_day": ["morning", "night", "day"]
}

ruler = EntityRuler(nlp)

for label_name, entity_list in ENTITY_MAPPING.items():
    for entity in entity_list:
        ruler.add_patterns([{"label": label_name, "pattern": [{"LOWER": entity}]}])
nlp.add_pipe(ruler)

doc_innocence = nlp(innocence_text)
for ent in doc_innocence.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

with doc_innocence.retokenize() as retokenizer:
    for ent in doc_innocence.ents:
        retokenizer.merge(doc_innocence[ent.start:ent.end])

matches = matcher(doc_innocence)

for match_id, start, end in matches:
    span = doc_innocence[start:end]

# displacy.serve(doc_innocence, style="ent")
# TODO # Experience matcher

# The Network (Just Innocence for now) ->
# TODO: need a weighted, bidirectional graph
# https://networkx.github.io/documentation/stable/auto_examples/index.html
# Author: Aric Hagberg (hagberg@lanl.gov)

# Nodes: entities, symbols
# Edges: how often entities appear together in poems

G = nx.star_graph(20)
G.add_nodes_from([ent for ent in doc_innocence.ents])

innocence_entities = [ent.text for ent in doc_innocence.ents]
co_occurences = []
for poem in poems_of_innocence:
    # Find common names & extract only the unique names
    co_occurences.append(list(set(set(innocence_entities) & (set(poem)))))

# Get all co-occurrences and their frequency and pass it as edges
combinations = [list(itertools.combinations(combo, 2)) for combo in co_occurences]
flattened_combinations = [x[0] for x in combinations if x]
occurrence_dict = {tup: flattened_combinations.count(tup) for tup in flattened_combinations}
for tup in flattened_combinations:
    G.add_edge(tup, flattened_combinations.count(tup))

pos = nx.spring_layout(G)
colors = range(2) # edge_colors todo
nx.draw(G, pos, node_color='#A0CBE2',
        width=4, edge_cmap=plt.cm.Blues, with_labels=False)
plt.show()