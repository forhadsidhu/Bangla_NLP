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
nltk.download('indian')
from nltk.corpus import indian
from nltk.tag import tnt
import string

'''
  Define tnt from nltk.tag for tagging each token in a sentence with supplementary information.
   TnT() is a statistical tagger which follows second-order Markov model. 
  This model is used for probability prediction of time series and sequence.
'''



def bangla_pos_tagger():
    tagged_set='bangla.pos'  #pre-trained Indian corpus is stored in bngla.pos
    word_set=indian.sents(tagged_set) #From Bengali corpus read the Bengali sentence and put them variable word_set
    count=0
    '''
    Using a for loop count all sentences which present in the corpus.
     startswith()-function is used to check the string is started with String “ ‘ “
    
    Here set the training percentage is 0.96 since the dataset is not sufficient.
    '''
    for sen in word_set:
        count=count+1
        sen="".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in  sen]).strip()
        print(count,sen)
        print('Total sentences in the tagged files are',count)

        train_perc=.96
        train_rows=int(train_perc*count)
        test_rows=train_rows+1

        print("Sentences to be trained",train_rows,"Sentences to be tested against",test_rows)

    data=indian.tagged_sents(tagged_set)
    train_data=data[:train_rows]
    test_data=data[test_rows:]

    '''
    now tokenize and check the parts of speech
    '''
    pos_tagger=tnt.TnT()
    pos_tagger.train(train_data)
    pos_tagger.evaluate(test_data)

    sentence="আমি ভাত খাই নাই অনেক দিন হল ।"

    tokenized=nltk.word_tokenize(sentence)
    print(pos_tagger.tag(tokenized))

if __name__ == '__main__':
    bangla_pos_tagger()
