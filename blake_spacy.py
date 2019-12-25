import itertools
import random
import re

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
import matplotlib.pyplot as plt
from constants import ENTITY_MAPPING
from constants import TRAIN_DATA

# Graph imports
import matplotlib.pyplot as plt
import networkx as nx

# NLTK stopwords
stop_words = stopwords.words('english')
stop_words.extend(['thy', 'thou', 'thee', 'till', 'every', 'shall', 'like', 'every'])

blake_poems = gutenberg.sents('blake-poems.txt')


# Prepare text
# Segment by poem (originally, the whole txt doc is divided by line)
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
        if word.text not in tokenized_text and word.text.isalnum():
            tokenized_text.append(word.lemma_)

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
        tokenized_data.append(clean_text(nlp(" ".join([word.lower() for word in poem]))
                                         ))
    return tokenized_data


tokenized_data = tokenize_data(all_poems)
poems_of_innocence = tokenized_data[3:17]
poems_of_experience = tokenized_data[17:-3]  # TODO double check

# Spacy entity recognition
innocence_text = " ".join([val for sublist in poems_of_innocence for val in sublist if val.isalnum()])
experience_text = " ".join([val for sublist in poems_of_experience for val in sublist if val.isalnum()])

matcher = Matcher(nlp.vocab)

# Train inaccurate data
nlp = spacy.blank("en")
optimizer = nlp.begin_training()
for i in range(20):
    random.shuffle(TRAIN_DATA)
    for text, annotations in TRAIN_DATA:
        nlp.update([text], [annotations], sgd=optimizer)
nlp.to_disk("./model")

# Entity Matcher
ruler = EntityRuler(nlp)

for label_name, entity_list in ENTITY_MAPPING.items():
    for entity in entity_list:
        ruler.add_patterns([{"label": label_name, "pattern": [{"LOWER": entity}]}])
nlp.add_pipe(ruler)


def matching(text):
    doc = nlp(text)

    with doc.retokenize() as retokenizer:
        for ent in doc.ents:
            retokenizer.merge(doc[ent.start:ent.end])

    return doc


doc_innocence = matching(innocence_text)
doc_experience = matching(experience_text)

# To visualize the entity tagging, uncomment
# displacy.serve(doc_innocence, style="ent")
# displacy.serve(doc_experience, style="ent")


# The Network - a weighted, bidirectional graph
# from: https://networkx.github.io/documentation/stable/auto_examples/index.html
# Author: Aric Hagberg (hagberg@lanl.gov)

# Nodes: entities, symbols
# Edges: how often entities appear together in poems


def graph_building(doc, poems):
    G = nx.Graph()
    G.add_nodes_from([ent for ent in doc.ents])

    entities = [ent.text for ent in doc.ents]
    co_occurences = []
    for poem in poems:
        # Find common names & extract only the unique names
        co_occurences.append(list(set(set(entities) & (set(poem)))))

    # Get all co-occurrences and their frequency and pass as edges
    combinations = [list(itertools.combinations(combo, 2)) for combo in co_occurences]
    flattened_combinations = [tup for sublist in combinations for tup in sublist]
    dict_combinations = {tup: flattened_combinations.count(tup) for tup in flattened_combinations}
    for tup, frequency in dict_combinations.items():
        if frequency > 3:  # the groups of entities frequently together
            G.add_edge(tup[0], tup[1], weight=frequency)

    pos = nx.spring_layout(G)
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 3]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < 3]

    nx.draw_networkx_nodes(G, pos, node_color="#aab3e3", node_size=200)
    nx.draw_networkx_edges(G, pos, edgelist=elarge, edge_color="#A0CBE2",
                           width=2)
    nx.draw_networkx_edges(G, pos, edgelist=esmall,
                           width=2, alpha=0.5, edge_color='#6776c7', style='dashed')
    # labels
    nx.draw_networkx_labels(G, pos, font_size=15, font_family='sans-serif')

    plt.axis('off')
    plt.show()


graph_building(doc_innocence, poems_of_innocence)
graph_building(doc_experience, poems_of_experience)

