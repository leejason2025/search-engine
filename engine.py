import string
import nltk
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#First step: Utilize NLTK to tokenize and stemming for each word in a query

#A mapping table that allows use to remove all punctuation
REMOVE_PUNCTUATION_TABLE = str.maketrans({x: None for x in string.punctuation})

#Tokenizes all words in a query
TOKENIZER = TreebankWordTokenizer()

#Picks out only stem words such as cat but not cats
STEMMER = PorterStemmer()

#Function to transform all queries into a list of terms
def tokenize_and_stem(s):
    return [STEMMER.stem(t) for t 
            in TOKENIZER.tokenize(s.translate(REMOVE_PUNCTUATION_TABLE))]

#Second step: Use scikit-learn to assign a score for words in all documents

#Create a vector of all words from the documents
vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english')

docs = [
    '''About us. We deliver Artificial Intelligence & Machine Learning
       solutions to solve business challenges.''',
    '''Contact information. Email [martin davtyan at filament dot ai]
       if you have any questions''',
    '''Filament Chat. A framework for building and maintaining a scalable
       chatbot capability''',
]

example_doc = docs[1]
example_doc_tokenized = TOKENIZER.tokenize(
        example_doc.translate(REMOVE_PUNCTUATION_TABLE)
        )

print(example_doc_tokenized)

example_doc_tokenized_and_stemmed = [STEMMER.stem(token) for token
                                     in example_doc_tokenized]

print(example_doc_tokenized_and_stemmed)

print(vectorizer.fit(docs))

print(vectorizer.vocabulary_)

query = 'contact email to chat to martin'
query_vector = vectorizer.transform([query]).todense()

print(query_vector)

doc_vectors = vectorizer.transform(docs)
similarity = cosine_similarity(np.asarray(query_vector), np.asarray(doc_vectors))

print(similarity)
