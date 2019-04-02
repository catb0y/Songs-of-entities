import re
import pandas as pd

# Gensim
import gensim
import gensim.corpora as corpora

# Plotting tools
import pyLDAvis.gensim

# NLTK imports
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
from nltk import word_tokenize

# Enable logging for gensim - optional
import logging
import warnings

# Import graph material
import matplotlib.pyplot as plt
import numpy as np


warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


# NLTK stopwords
stop_words = stopwords.words('english')
stop_words.extend(['thy', 'thou', 'thee', 'till'])

blake_poems = gutenberg.sents('blake-poems.txt')


## Prepare text
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


# Build Dictionary (Construct word<->id mappings)
dictionary = corpora.Dictionary(tokenized_data)  # Initialize a Dictionary

# Texts to Bag Of Words
corpus = [dictionary.doc2bow(text) for text in tokenized_data]


# Note: format of piece [(word_id, count), ...]
# e.g. corpus[20]
# [(12, 3), (14, 1), (21, 1), (25, 5), (30, 2), (31, 5), (33, 1), (42, 1), (43, 2),  ...

# Build LDA (topic) model
lda_model = gensim.models.LdaModel(corpus=corpus,
                                   num_topics=10,
                                   id2word=dictionary)
lda_model.save(fname="blake.lda")


# Visualize topics
# vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
# pyLDAvis.show(vis)


# Determine which topics belong to which documents

def get_song_collection(all_poems_index):
    song = []
    song_poems = all_poems_index
    for poem in range(len(song_poems)):
        song.extend(lda_model[corpus[poem]])
    return song


innocence = get_song_collection(all_poems[:17])
experience = get_song_collection(all_poems[17:])

# See which id corresponds to which word
# print(innocence)
# for x in innocence:
#     x[0] = lda_model.id2word[x[0]]
#     print(innocence)


# For each topic, we will explore the words occuring in that topic and its relative weight
# see: https://github.com/priya-dwivedi/Deep-Learning/blob/master/topic_modeling/LDA_Newsgroup.ipynb

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")


# To export the top words for each topic to a csv file:

top_words_per_topic = []
for t in range(lda_model.num_topics):
    top_words_per_topic.extend([(t, ) + i for i in lda_model.show_topic(t, topn=20)])


# For visualization, see: https://stackoverflow.com/questions/32945012/how-to-understand-this-heat-map-of-topic-shares-in-documents

# Notes  topic strengths in poem = lda_model[dictionary.doc2bow(poem)]
