import string
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

#Example documents
docs = [
    '''About us. We deliver Artificial Intelligence & Machine Learning
       solutions to solve business challenges.''',
    '''Contact information. Email [martin davtyan at filament dot ai]
       if you have any questions''',
    '''Filament Chat. A framework for building and maintaining a scalable
       chatbot capability''',
]

#A mapping table that allows use to remove all punctuation
REMOVE_PUNCTUATION_TABLE = str.maketrans({x: None for x in string.punctuation})

#Tokenizes all words in a query
TOKENIZER = TreebankWordTokenizer()

#Picks out only stem words such as cat but not cats
STEMMER = PorterStemmer()

#Take the first document and print the list of tokenized words
example_doc = docs[1]
example_doc_tokenized = TOKENIZER.tokenize(
        example_doc.translate(REMOVE_PUNCTUATION_TABLE)
        )
print("Tokenized Doc: ", example_doc_tokenized, "\n")

#Use the tokenized words to generate stems of unique words
example_doc_tokenized_and_stemmed = [STEMMER.stem(token) for token
                                     in example_doc_tokenized]
print("Stemmed Doc: ", example_doc_tokenized_and_stemmed, "\n")

#Function to transform all queries into a list of terms
def tokenize_and_stem(s):
    return [STEMMER.stem(t) for t 
            in TOKENIZER.tokenize(s.translate(REMOVE_PUNCTUATION_TABLE))]

#Create a vector of all words from the documents
vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english')
vectorizer.fit(docs)
print("Document Vector: ", vectorizer.vocabulary_,"\n")

#Transform the example query provided into a vector
query = 'contact email to chat to martin'
query_vector = np.asarray(vectorizer.transform([query]).todense())  
query_vector = np.reshape(query_vector, (1, -1)) 
doc_vectors = vectorizer.transform(docs).toarray()
similarity = cosine_similarity(query_vector, doc_vectors)
print("Similarity: ", similarity,"\n")

#Rank the documents to find most relevant doc
ranks = (-similarity).argsort(axis=None)
most_relevant_doc = docs[ranks[0]]
print("Most Relevant Doc: ", most_relevant_doc, "\n")

#Now we try to improve ranking with feedback
feedback = {
        'who makes chatbots': [(2, 0.), (0, 1.), (1, 1.), (0, 1.)],
        'about page': [(0, 1.)]
}  

#Current search system response
similarity = cosine_similarity(vectorizer.transform(
                           ['who makes chatbots']), doc_vectors)
ranks = (-similarity).argsort(axis=None)
print("Most Relevant doc with feedback: ", docs[ranks[0]], "\n")

#Implementing Positive Feedback feature for search engine
query = 'who is making chatbots information'
feedback_queries = list(feedback.keys())
similarity = cosine_similarity(vectorizer.transform([query]), 
                               vectorizer.transform(feedback_queries))
max_idx = np.argmax(similarity)
pos_feedback_doc_idx = [idx for idx, feedback_value 
                        in feedback[feedback_queries[max_idx]] 
                        if feedback_value == 1.]
counts = Counter(pos_feedback_doc_idx)
pos_feedback_proportions = {
        doc_idx: count / sum(counts.values()) 
        for doc_idx, count in counts.items()
}
nn_similarity = np.max(similarity)
pos_feedback_feature = [nn_similarity * pos_feedback_proportions.get(idx, 0.) 
                        for idx, _ in enumerate(docs)]
print("Similarity with feed back feature: ", pos_feedback_feature, "\n")

#Class Scorer scores documents for a search query based on tf-idf 
#similarity and relevance feedback
class Scorer():
        #Initialize a scorer with a collection of documents, fit a 
        #vectorizer and list feature functions
    def __init__(self, docs):
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
        
    #Generic scoring function: for a query output a numpy array
    #of scores aligned with a document list we initialized the scorer with
    def score(self, query):
        feature_vectors = [feature(query) for feature 
                           in self.features]
        
        feature_vectors_weighted = [feature * weight for feature, weight
                                    in zip(feature_vectors, self.feature_weights)]
        return np.sum(feature_vectors_weighted, axis=0)
    
    #Learn feedback in a form of `query` -> (doc index, feedback value). In real life 
    # it would be an incremental procedure updating the feedback object.
    def learn_feedback(self, feedback_dict):
        self.feedback = feedback_dict
        
    #TF-IDF feature. Return a numpy array of cosine similarities
    #between TF-IDF vectors of documents and the query
    def _feature_tfidf(self, query):
        query_vector = vectorizer.transform([query])
        similarity = cosine_similarity(query_vector, self.doc_tfidf)
        return similarity.ravel()
    
    #Positive feedback feature. Search the feedback dict for a query
    #similar to the given one, then assign documents positive values
    #if there is positive feedback about them.
    def _feature_positive_feedback(self, query):
        
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

#Test before any feed back is given
scorer = Scorer(docs)
print("No Feedback Test: ", docs[scorer.score(query).argmax()], "\n")

#Test after feedback is given
scorer.learn_feedback(feedback)
print("With Feedback Test: ", docs[scorer.score(query).argmax()], "\n")

#Test with weights and feedback
scorer.feature_weights = [0.6, 0.4]
print("With Feedback and Weight Test: ", docs[scorer.score(query).argmax()], "\n")
