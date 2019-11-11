import re
import pandas as pd

# NLTK imports
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk import word_tokenize

# Spacy imports
import spacy
nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser"])
from spacy import displacy
from spacy.matcher import Matcher
from spacy.tokens import Span

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

# Tokenize and clean text, make it into data for Gensim
# TODO lemmatize


def clean_text(poem):
    tokenized_text = []
    for word in poem:
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

# Spacy entity recognition
innocence_text = " ".join([val for sublist in all_poems[:17] for val in sublist if val.isalnum()])
experience_text = " ".join([val for sublist in all_poems[17:] for val in sublist if val.isalnum()])

doc_innocence = nlp(innocence_text)
doc_experience = nlp(experience_text)

for ent in doc_innocence.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

matcher = Matcher(nlp.vocab)

# Innocence matcher
matcher.add("Animals", None, [{"LOWER": "lamb"}, {"LOWER": "tyger"}, {"LOWER": "dove"}])
matches = matcher(doc_innocence)

for match_id, start, end in matches:
    # create a new Span for each match and use the match_id (ANIMAL) as the label
    span = Span(doc_innocence, start, end, label=match_id)
    doc_innocence.ents = list(doc_innocence.ents) + [span]  # add span to doc.ents

displacy.serve(doc_innocence, style="ent")

# TODO GRaph library for py