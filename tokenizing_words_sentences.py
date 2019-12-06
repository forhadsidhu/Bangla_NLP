import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize,word_tokenize


# tokenizing -word tokenizers.......sentence tokenizers
# lexicon and corporas
# corpora -body of text ex: medical journals,presidential speeches, english language
# lexicon - words and their names
# investor speak 'bull' =someone who is positive about the market
# english-speak 'bull' =scary animal you dont want eunning you

if __name__=='__main__':

    example_text="হ্যালো মি এক্স কেমন আছেন? ভাল, আপনি কেমন আছেন?"

    print(sent_tokenize(example_text))
    #print(word_tokenize(example_text))

    for i in word_tokenize(example_text):
        print(i)