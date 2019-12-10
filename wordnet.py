
'''

WordNet is a lexical database for the english language which
was created by princton and is part of the NLTK corpus

'''

from nltk.corpus import wordnet

'''
Then, we're going to use the term "program" to find synsets like so:
Synset is a special kind of a simple interface that is present in NLTK to look up words in WordNet. 
Synset instances are the groupings of synonymous words that express the same concept.
'''
syns=wordnet.synsets("program")

#an example of a synset
print(syns[0].name)

#just the word
print(syns[0].lemmas()[0].name())

#definition of that first synset
print(syns[0].definition())
#examples of the word in use
print(syns[0].examples())

#now find the synnomys and antonyms of a word

synonyms=[]
antonyms=[]

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        print(l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))


#now compare the similarity of two words and their tenses with incorporating Wu and Palmar method

w1= wordnet.synset("ship.n.01")
w2=wordnet.synset("boat.n.01")

print(w1.wup_similarity (w2))


w1= wordnet.synset("ship.n.01")
w2=wordnet.synset("cat.n.01")

print(w1.wup_similarity (w2))


w1= wordnet.synset("ship.n.01")
w2=wordnet.synset("car.n.01")

print(w1.wup_similarity (w2))

