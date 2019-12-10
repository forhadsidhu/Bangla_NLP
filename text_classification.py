import nltk
nltk.download('movie_reviews')
import random
import pickle
from nltk.corpus import movie_reviews

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

documents=[(list(movie_reviews.words(fileid)),category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category) ]


random.shuffle(documents)
print(documents[0])

#print(documents[1])

all_words=[]

for w in movie_reviews.words():
    all_words.append(w.lower())

#Frequency count
all_words=nltk.FreqDist(all_words)
#print(all_words.most_common(15))

# print(all_words["stupid"])

#train with top3k words
word_features=list(all_words.keys())[:3000]


def find_features(document):
    words=set(document)
    features={}
    for w in word_features:
        features[w]=(w in words) #if word is in document it is true otherwise false
    return features


print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

#documents theke word niye dekhbo
featuresets=[(find_features(rev),category) for (rev,category)in documents]

print (featuresets[:5])

training_sets=featuresets[:1900]
testing_sets=featuresets[1900:]

#Now make a classifier.
# classifier =nltk.NaiveBayesClassifier.train(training_sets)

classifier_f=open("naivebayes.pickle","rb")
classifier=pickle.load(classifier_f)
classifier_f.close()

print("Naive Bayes Algo accuracy:",(nltk.classify.accuracy(classifier,testing_sets))*100)
classifier.show_most_informative_features(15)  #want to show 15 most informative features

'''
How to save your train algorithm?


'''
# save_classifier=open("naivebayes.pickle","wb")
# pickle.dump(classifier,save_classifier)
# save_classifier.close()



