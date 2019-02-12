import re

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

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)


# NLTK stopwords
stop_words = stopwords.words('english')
stop_words.extend(['thy', 'thou', 'thee', 'till'])

blake_poems = gutenberg.sents('blake-poems.txt')

## Prepare text
# Segment by poem (originally, the whole txt doc is divided by line)
# TODO + remove the book of thel


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

# Segment by song collection


def chunk_song_collection(text):
    innocence = []
    experience = []
    list_of_song_collections = []
    splitter = text.index(['SONGS', 'OF', 'EXPERIENCE'])
    start = 2
    end = text.index(['APPENDIX'])

    innocence.append(text[start:splitter])
    print(innocence)
    experience.append(text[splitter:end])
    print(experience)

    list_of_song_collections.append(innocence)
    list_of_song_collections.append(experience)

    return innocence, experience
    return list_of_song_collections


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
# Keep 1 model, find out relative poems to documents, analyze and compare
# Use method lda_model[corpus[n]

def explore_song_collections(text):
    innocence, experience = chunk_song_collection(text)
    pass

explore_song_collections(all_poems)