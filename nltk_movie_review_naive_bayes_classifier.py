# NLTK Movie Reviews - Naive Bayes Classifier

# * I'll train a classifier using nltk corpus 'movie_reviews' and the Statistic Classifier 'Naive Bayes'

# Imports
import nltk
import random
import pickle
from nltk.corpus import movie_reviews


# Getting the documents and their category from corpus
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle
random.shuffle(documents)


# List of all words of corpus
all_words = [word.lower() for word in movie_reviews.words()]

# Count the frequence of each word in corpus
all_words = nltk.FreqDist(all_words)
all_words


# Preparing the features to train the model
word_features = all_words.keys()

# Fuction to get features from a document/review
def find_features(document):
    '''
    This fuction get a document/review and then compare what words from list of features is on document.
    '''
    words_doc = set(document)
    features = {}
    for w_f in word_features:
        features[w_f] = (w_f in words_doc)

    return features


# Creating the features set for all documents
featuresets = [(find_features(rev), category) for (rev, category) in documents]


# Train Data. 95% for train
training_set = featuresets[:(len(featuresets)*95)//100]

# Test Data. 5% for test
testing_set = featuresets[(len(featuresets)*95)//100:]


# Creating and train the classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)


# Getting the Accuracy
print("\nModel Accuracy: ")
print("Classifier Accuracy:",(nltk.classify.accuracy(classifier, testing_set))*100,"%")



# Most infromative features
classifier.show_most_informative_features(15)


# Saving the classifier
'''
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
'''

# Load the classifier
'''
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
'''

