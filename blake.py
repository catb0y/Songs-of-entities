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


all_songs = chunk_poems(blake_poems)


# Segment by song collection


def chunk_song_collection(per_song):
    list_of_song_collections = []
    innocence = []
    experience = []
    start = 1
    end = per_song.index(['APPENDIX'])

    splitter = per_song.index(['SONGS', 'OF', 'EXPERIENCE'])
    innocence.extend(per_song[start:splitter])
    experience.extend(per_song[splitter:end])

    list_of_song_collections.append(innocence)
    list_of_song_collections.append(experience)

    return list_of_song_collections


all_songs_split_by_collection = chunk_song_collection(all_songs)
# Tokenize and clean text, make it into data for Gensim
# TODO lemmatize


def clean_text(collection):
    tokenized_text = []
    for text in collection:
        for word in text:
            tokenized_text.extend(word_tokenize(word.lower()))

    return [
        elem for elem in tokenized_text
        if elem not in stop_words and
        re.match(
            '[a-zA-Z\-][a-zA-Z\-]{2,}',
            elem
        )
    ]

# Tokenize data by poem
#tokenized_data = []
#for poem in all_songs:
 #   tokenized_data.append(clean_text(poem))


# Tokenize data by collection
tokenized_data = []
for collection in all_songs_split_by_collection:
    tokenized_data.append(clean_text(collection))

print("tokenized")

# Build Dictionary (Construct word<->id mappings)
dictionary = corpora.Dictionary(tokenized_data)  # Initialize a Dictionary

# Texts to Bag Of Words
corpus = [dictionary.doc2bow(text) for text in tokenized_data]

print("corpus")

# Note: format of piece [(word_id, count), ...]
# e.g. corpus[20]
# [(12, 3), (14, 1), (21, 1), (25, 5), (30, 2), (31, 5), (33, 1), (42, 1), (43, 2),  ...

# Build LDA (topic) model
# Build the LDA model
lda_model = gensim.models.LdaModel(corpus=corpus,
                                   num_topics=3,
                                   id2word=dictionary)
lda_model.save(fname="blake.lda")

print("lda")


# Visualize topics
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.show(vis)


# Determine which topics belong to which documents
# Option 1: create 2 models, one per collection
# Option 2: keep 1 model but find out how to proceed


# Compare topics