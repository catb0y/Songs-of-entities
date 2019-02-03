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

# Segment by poem (currently, the whole txt doc is divided by line)
# TODO + remove the book of thel


def chunk_poems(docu):
    list_of_numbers = ['I', 'II', 'III']
    list_of_poems = []
    current_poem = []

    for listed_line in docu:
        is_current_line_a_title = all(word.isupper() for word in listed_line if word not in list_of_numbers)

        if is_current_line_a_title:
            list_of_poems.append(current_poem.copy())
            current_poem.clear()

        current_poem.extend(listed_line)

    return list_of_poems


all_songs = chunk_poems(blake_poems)

print("poems")

# Tokenize and clean text, make it into data for Gensim
# TODO does it lemmatize, too? no


def clean_text(text):
    tokenized_text = []
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


tokenized_data = []
for poem in all_songs:
    tokenized_data.append(clean_text(poem))

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
                                   num_topics=10,
                                   id2word=dictionary)
lda_model.save(fname="blake.lda")

print("lda")


# Visualize topics
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.show(vis)
