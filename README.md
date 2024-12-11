# Network Analysis of Symbolism in William Blake's Poetry

> Tiger, tiger, burning bright  
> In the forests of the night,  
> What immortal hand or eye  
> Could frame thy fearful symmetry?


### Project Overview

This project applies **network analysis** to explore the symbolism in **William Blake's *Songs of Innocence and of Experience***. By performing **Named Entity Recognition (NER)** and **network graph analysis**, it uncovers how major symbols and imagery related to the dualities of Innocence and Experience interact and contribute to the overarching themes in Blake's poetry.


You can find the article that delves into poetry analysis here: [A Brief Network Analysis of Symbolism in Blake's Poetry with Python](https://marta-p.com/2020/01/05/a-brief-network-analysis-of-symbolism-in-blakes-poetry-with-python/)




***


## **Data Collection & Preprocessing**

### **Loading and Segmenting Blake’s Poems**

The text is sourced from the **Gutenberg** corpus, which contains the complete collection of Blake's poems (and it can be found in `blake-poems.txt` within the repo). The `gutenberg.sents` function is used to load the poems, splitting the text into sentences. We then define a function `chunk_poems()` that identifies poem titles based on lines written in uppercase, excluding Roman numerals ('I', 'II', and 'III'), which are used to number sections in Blake’s work. This way, only actual poem titles are selected, while lines marking sections are disregarded.

```python

blake_poems = gutenberg.sents('blake-poems.txt')

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

```


### **Cleaning the Text**

In the `clean_text()` function, we tokenize the poem and remove stopwords, non-alphanumeric characters, and words that are less than three characters long. We also perform lemmatization using **WordNetLemmatizer** to reduce words to their base form.

```python

def clean_text(poem):

    tokenized_text = []

    for word in poem:

        if word.text not in tokenized_text and word.text.isalnum():

            tokenized_text.append(word.lemma_)

    return [

        elem for elem in tokenized_text

        if elem not in stop_words and

           re.match(

               '[a-zA-Z-][a-zA-Z-]{2,}',

               elem

           )

    ]

```


### **Tokenizing the Data**

This function tokenizes the entire collection of poems using the `nlp` pipeline and cleans each tokenized poem using the `clean_text()` function.

```python

def tokenize_data(text):

    tokenized_data = []

    for poem in text:

        tokenized_data.append(clean_text(nlp(" ".join([word.lower() for word in poem]))

                                         ))

    return tokenized_data

tokenized_data = tokenize_data(all_poems)

poems_of_innocence = tokenized_data[3:17]

poems_of_experience = tokenized_data[17:-3]

```

***


## **Named Entity Recognition (NER)**

### **Preparing Text for Entity Recognition**

The next step is to prepare the text by creating strings that contain only alphanumeric words from both the _Innocence_ and _Experience_ poems.

```python

innocence_text = " ".join([val for sublist in poems_of_innocence for val in sublist if val.isalnum()])

experience_text = " ".join([val for sublist in poems_of_experience for val in sublist if val.isalnum()])

```


### **Training the SpaCy Model**

A custom **blank spaCy model** is created to train on the provided data. The `TRAIN_DATA` variable contains the training annotations, which is used in a loop to train the model over multiple iterations.

```python

nlp = spacy.blank("en")

optimizer = nlp.begin_training()

for i in range(20):

    random.shuffle(TRAIN_DATA)

    for text, annotations in TRAIN_DATA:

        nlp.update([text], [annotations], sgd=optimizer)

nlp.to_disk("./model")

```


### **Matching Entities Using the Entity Ruler**

We use the `EntityRuler` component in spaCy to match predefined symbolic entities from `ENTITY_MAPPING` and add them to the model.

```python

ruler = EntityRuler(nlp)

for label_name, entity_list in ENTITY_MAPPING.items():

    for entity in entity_list:

        ruler.add_patterns([{"label": label_name, "pattern": [{"LOWER": entity}]}])

nlp.add_pipe(ruler)

```

***


## **Entity Recognition**

### **Processing the Text**

We apply the trained spaCy model to the _Innocence_ and _Experience_ text data using the `matching()` function. This function retokenizes the recognized entities so that they are grouped together as a single token.

```python

def matching(text):

    doc = nlp(text)

    with doc.retokenize() as retokenizer:

        for ent in doc.ents:

            retokenizer.merge(doc[ent.start:ent.end])

    return doc

doc_innocence = matching(innocence_text)

doc_experience = matching(experience_text)

```


### **Debugging by Visualizing Entities**

While this is not part of the final code output, **spaCy's displaCy** visualizer can be used for debugging and inspecting entity recognition, making sure everything is running as expected. Uncommenting the following lines will display the entities in both _Innocence_ and _Experience_ texts.

```python

# displacy.serve(doc_innocence, style="ent")

# displacy.serve(doc_experience, style="ent")

```

***


## **Network Construction and Visualization**

### **Building the Graph**

The `graph_building()` function constructs a **network graph** using **NetworkX** to represent the relationships between entities. Here:

- **Nodes** are entities detected by the spaCy model.

- **Edges** are formed between entities that co-occur in the same poem. The edges are weighted based on the frequency of co-occurrence.

```python

def graph_building(doc, poems):

    G = nx.Graph()

    G.add_nodes_from([ent for ent in doc.ents])

    entities = [ent.text for ent in doc.ents]

    co_occurences = []

    for poem in poems:

        co_occurences.append(list(set(set(entities) & (set(poem)))))

    combinations = [list(itertools.combinations(combo, 2)) for combo in co_occurences]

    flattened_combinations = [tup for sublist in combinations for tup in sublist]

    dict_combinations = {tup: flattened_combinations.count(tup) for tup in flattened_combinations}

    for tup, frequency in dict_combinations.items():

        if frequency > 3:

            G.add_edge(tup[0], tup[1], weight=frequency)

    pos = nx.spring_layout(G)

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 3]

    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < 3]

    nx.draw_networkx_nodes(G, pos, node_color="#aab3e3", node_size=200)

    nx.draw_networkx_edges(G, pos, edgelist=elarge, edge_color="#A0CBE2", width=2)

    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=2, alpha=0.5, edge_color='#6776c7', style='dashed')

    nx.draw_networkx_labels(G, pos, font_size=15, font_family='sans-serif')

    plt.axis('off')

    plt.show()

```


### **Displaying the Graph**

Finally, the function is called to create the graph for both the _Innocence_ and _Experience_ poems:

```python

graph_building(doc_innocence, poems_of_innocence)

graph_building(doc_experience, poems_of_experience)

```

***


## **Conclusion**

This code shows how to apply **Named Entity Recognition (NER)** using **spaCy** to extract symbolic entities from _William Blake's Songs of Innocence and Experience_. The entities are then used to construct a **network graph** that visualizes how different symbols interact within the poems. 

- **Poem segmentation** is done by identifying titles and dividing the text into distinct poems.

- **Text cleaning and tokenization** prepares the text for further analysis.

- **Custom NER training** is used to better recognize symbolic entities, since the symbols in Blake’s poems have meanings that depend on the context of the work. This customization helps the model identify key symbols like "Lamb," "Tyger," "Innocence," and "Experience," all of which carry deeper, layered meanings specific to Blake’s themes.

- The **network graph** provides a visualization of the relationships between these entities based on their co-occurrence in the poems, showing how the symbolic elements interact and evolve within Blake’s thematic structure.

This approach can be extended to analyze the symbolism in other literary works or explore more advanced NLP techniques for symbolic recognition. Just make sure to customize your NER model accordingly, depending on the literary context and desired research approach.
