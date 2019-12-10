
'''
A very similar operation to stemming is called lemmatizing. The major difference between these is,
 as you saw earlier,
 stemming can often create non-existent words, whereas lemmas are actual words.

 default pos in lemmatizing is n
'''

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geek"))
print(lemmatizer.lemmatize("better"))
print(lemmatizer.lemmatize("geekc"))

print(lemmatizer.lemmatize("best",pos="a"))
print(lemmatizer.lemmatize("good",pos='a'))