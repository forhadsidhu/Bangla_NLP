import nltk

nltk.download('movie_reviews')
import random
import pickle
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import  SVC,LinearSVC,NuSVC


# for votting
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers=classifiers
    def classify(self, features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)    #classifier gula psitive or negative return
            votes.append(v)           #krte thakbe but statistics r mode most occurance ta output dibe
        return mode(votes)

    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes=votes.count(mode(votes)) #how many most popular votes
        conf=choice_votes/(len(votes))
        return conf








'''
    pos-|-> .txt    neg-|-> .txt
        |-> .txt        |-> .txt
        |-> .txt        |-> .txt

    list = allwords in movie review
    Req dictionary={words,count}

    final words=unique_word

                       final words
                          |
                        /  \
                      /      \
            training set    testing set





'''
print(movie_reviews)

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
print(documents[0])

# print(documents[1])

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

# Frequency count
all_words = nltk.FreqDist(all_words)
# print(all_words.most_common(15))

# print(all_words["stupid"])

# train with top3k words
word_features = list(all_words.keys())[:3000]


def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)  # if word is in document it is true otherwise false
    return features


print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

# documents theke word niye dekhbo
featuresets = [(find_features(rev), category) for (rev, category) in documents]

print(featuresets[:5])

training_sets = featuresets[:1900]
testing_sets = featuresets[1900:]

# Now make a classifier.
# classifier =nltk.NaiveBayesClassifier.train(training_sets)

classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Original Naive Bayes Algo accuracy:", (nltk.classify.accuracy(classifier, testing_sets)) * 100)
classifier.show_most_informative_features(15)  # want to show 15 most informative features

'''
How to save your train algorithm?


'''
# save_classifier=open("naivebayes.pickle","wb")
# pickle.dump(classifier,save_classifier)
# save_classifier.close()


#create a classifier using sklearn
MNB_classifier=SklearnClassifier(MultinomialNB( ))
MNB_classifier.train(training_sets)
print("MNB classifier Algo accuracy:", (nltk.classify.accuracy(MNB_classifier, testing_sets)) * 100)

# Gaus_classifier=SklearnClassifier(GaussianNB( ))
# Gaus_classifier.train(training_sets)
#
#
# print("Gaussian learn classifier Algo accuracy:", (nltk.classify.accuracy(Gaus_classifier, testing_sets)) * 100)

bernou_classifier=SklearnClassifier(BernoulliNB( ))
bernou_classifier.train(training_sets)


print("Bernouli classifier Algo accuracy:", (nltk.classify.accuracy(bernou_classifier, testing_sets)) * 100)


voted_classifier=VoteClassifier(MNB_classifier,bernou_classifier,classifier)
print("voted classifier",(nltk.classify.accuracy(voted_classifier,testing_sets))*100)
print("Classification:",voted_classifier.classify(testing_sets[0][0]),"Confidence %:",voted_classifier.confidence(testing_sets[0][0]))


def sentiment(text):
    feats=find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)