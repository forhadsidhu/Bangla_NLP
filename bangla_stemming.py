from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from bengali_stemmer.rafikamal2014 import RafiStemmer



"""
Stem is the form of a word before any inflectional affixes are added
it is like word

"""



ps=PorterStemmer()
example_words=["করাল","কইরালাইতেছে","নাম্বারটা","উচ্চারন","করিজাইও"]
stemmer = RafiStemmer()

for w in example_words:
    print(stemmer.stem_word(w))
