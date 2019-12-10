'''

Chinking means condition r oigula chara baki sob chunk(weksath) kora
cndition r POS gula alada rakha
'''

import nltk

nltk.download('indian')
from nltk.corpus import indian
from nltk.tag import tnt
import string


def bangla_chunk():
    # Parts of speech tagging part................

    tagged_set = 'bangla.pos'  # pre-trained Indian corpus is stored in bngla.pos
    word_set = indian.sents(tagged_set)  # From Bengali corpus read the Bengali sentence and put them variable word_set
    count = 0
    '''
    Using a for loop count all sentences which present in the corpus.
     startswith()-function is used to check the string is started with String “ ‘ “

    Here set the training percentage is 0.96 since the dataset is not sufficient.
    '''
    for sen in word_set:
        count = count + 1
        sen = "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in sen]).strip()
        print(count, sen)
        print('Total sentences in the tagged files are', count)

        train_perc = .96
        train_rows = int(train_perc * count)
        test_rows = train_rows + 1

        print("Sentences to be trained", train_rows, "Sentences to be tested against", test_rows)

    data = indian.tagged_sents(tagged_set)
    train_data = data[:train_rows]
    test_data = data[test_rows:]

    '''
    now tokenize and check the parts of speech
    '''
    pos_tagger = tnt.TnT()
    pos_tagger.train(train_data)
    pos_tagger.evaluate(test_data)

    sentence = "আমি ভাত খাই নাই অনেক দিন হল । বিজেআইটি একটি কম্পানি , কি কম্পানি সেটা জানার দরকার নাই । "

    tokenized = nltk.word_tokenize(sentence)
    words = pos_tagger.tag(tokenized)
    '''
    RB.?  = any form  of RB
    NNP  =we are required

    '''
    chunkGram = r"""Chunk: {<.*>+}
                          }<VB.? |IN|DT|TO|NN>+{"""
    chunkParser = nltk.RegexpParser(chunkGram)  # regex parser use krci
    chunked = chunkParser.parse(words)
    chunked.draw()
    # print(chunked)


if __name__ == '__main__':
    bangla_chunk()