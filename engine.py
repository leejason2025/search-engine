import string
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

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
query_vector = np.asarray(vectorizer.transform([query]).todense())  # Convert to numpy array
query_vector = np.reshape(query_vector, (1, -1))  # Reshape to 2D array with a single row

print(query_vector)

doc_vectors = vectorizer.transform(docs).toarray()  # Convert document vectors to an array

similarity = cosine_similarity(query_vector, doc_vectors)

print(similarity)

ranks = (-similarity).argsort(axis=None)
print(ranks)

most_relevant_doc = docs[ranks[0]]
print(most_relevant_doc)

feedback = {
        'who makes chatbots': [(2, 0.), (0, 1.), (1, 1.), (0, 1.)],
        'about page': [(0, 1.)]
}

similarity = cosine_similarity(vectorizer.transform(
                           ['who makes chatbots']), doc_vectors)
ranks = (-similarity).argsort(axis=None)
print(docs[ranks[0]])

query = 'who is making chatbots information'
feedback_queries = list(feedback.keys())

similarity = cosine_similarity(vectorizer.transform([query]), 
                               vectorizer.transform(feedback_queries))
print(similarity)

max_idx = np.argmax(similarity)
print(feedback_queries[max_idx])

pos_feedback_doc_idx = [idx for idx, feedback_value 
                        in feedback[feedback_queries[max_idx]] 
                        if feedback_value == 1.]
print(pos_feedback_doc_idx)

counts = Counter(pos_feedback_doc_idx)
print(counts)

pos_feedback_proportions = {
        doc_idx: count / sum(counts.values()) for doc_idx, count in counts.items()
}
print(pos_feedback_proportions)

nn_similarity = np.max(similarity)
pos_feedback_feature = [nn_similarity * pos_feedback_proportions.get(idx, 0.) 
                        for idx, _ in enumerate(docs)]
print(pos_feedback_feature)

class Scorer():
    """ Scores documents for a search query based on tf-idf
        similarity and relevance feedback
        
    """
    def __init__(self, docs):
        """ Initialize a scorer with a collection of documents, fit a 
            vectorizer and list feature functions
        
        """
        self.docs = docs
        
        self.vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, 
                                          stop_words='english')
        self.doc_tfidf = self.vectorizer.fit_transform(docs)
        
        self.features = [
            self._feature_tfidf,
            self._feature_positive_feedback,
        ]
        self.feature_weights = [
            1.,
            2.,
        ]
        
        self.feedback = {}
        
    def score(self, query):
        """ Generic scoring function: for a query output a numpy array
            of scores aligned with a document list we initialized the
            scorer with
        
        """
        feature_vectors = [feature(query) for feature 
                           in self.features]
        
        feature_vectors_weighted = [feature * weight for feature, weight
                                    in zip(feature_vectors, self.feature_weights)]
        return np.sum(feature_vectors_weighted, axis=0)
    
    def learn_feedback(self, feedback_dict):
        """ Learn feedback in a form of `query` -> (doc index, feedback value).
            In real life it would be an incremental procedure updating the
            feedback object.
        
        """
        self.feedback = feedback_dict
        
    def _feature_tfidf(self, query):
        """ TF-IDF feature. Return a numpy array of cosine similarities
            between TF-IDF vectors of documents and the query
        
        """
        query_vector = vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, self.doc_tfidf)
        return similarity.ravel()
    
    def _feature_positive_feedback(self, query):
        """ Positive feedback feature. Search the feedback dict for a query
            similar to the given one, then assign documents positive values
            if there is positive feedback about them.
        
        """
        if not self.feedback:
            return np.zeros(len(self.docs))
        
        feedback_queries = list(self.feedback.keys())
        similarity = cosine_similarity(self.vectorizer.transform([query]),
                                       self.vectorizer.transform(feedback_queries))
        nn_similarity = np.max(similarity)
        
        nn_idx = np.argmax(similarity)
        pos_feedback_doc_idx = [idx for idx, feedback_value in
                                self.feedback[feedback_queries[nn_idx]]
                                if feedback_value == 1.]
        
        feature_values = {
                doc_idx: nn_similarity * count / sum(counts.values()) 
                for doc_idx, count in Counter(pos_feedback_doc_idx).items()
        }
        return np.array([feature_values.get(doc_idx, 0.) 
                         for doc_idx, _ in enumerate(self.docs)])


scorer = Scorer(docs)
print(scorer.score(query))

print(docs[scorer.score(query).argmax()])

scorer.learn_feedback(feedback)
print(scorer.score(query))


print(docs[scorer.score(query).argmax()])


scorer.feature_weights = [0.6, 0.4]
print(scorer.score(query))

print(docs[scorer.score(query).argmax()])
