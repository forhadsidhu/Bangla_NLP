"""
 POS:: parts of speech tagging
 creates word and corresponding parts of speech
"""

import nltk
nltk.download('state_union')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
              #unsupervisied machine learning tokenizer


train_text=state_union.raw("2006-GWBush.txt")
sample_text=state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer=PunktSentenceTokenizer(sample_text)
tokenized=custom_sent_tokenizer.tokenize(sample_text)


def process_content():
    try:
        for i in tokenized:
            print(i)
            words=nltk.word_tokenize(i)
            tagged=nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))


# process_content()

'''
   bangla parts of speech taggig 

'''
import nltk
from nltk.corpus import indian
from nltk.tag import tnt
import string

'''
  Define tnt from nltk.tag for tagging each token in a sentence with supplementary information.
   TnT() is a statistical tagger which follows second-order Markov model. 
  This model is used for probability prediction of time series and sequence.
'''



def bangla_pos_tagger():
    tagged_set='bangla.pos'
    word_set=indian.sents(tagged_set)
    count=0

    for sen in word_set:
        count=count+1
        sen="".join()
