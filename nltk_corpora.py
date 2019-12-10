
#loication of nltk
import nltk
print(nltk.__file__)

import nltk
nltk.download('gutenberg')


from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample=gutenberg.raw("bible-kjv.txt")
tok=sent_tokenize(sample)
print(tok[5:15])
