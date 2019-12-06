
'''
 which word has no meaning in text,called stopword
'''
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence="এটা হল একটা উধারন যেটা দিয়ে আমরা কাজ করব এবং যেটা অনেক প্যাড়া দিবে ।"

stopwords={
}


if __name__=='__main__':

    # map the stopwards in dictionary
    with open('stopwords-bn.txt','r',encoding = 'utf-8') as f:
        x=f.readlines()

        for i in range(0,len(x)):
            ss=x[i]
            word=ss.split('\n')[0]
            #print(word)
            stopwords.update({word:1})

    #now tokenize the sentence and check its existance....
    for i in word_tokenize(example_sentence):
        if i in stopwords.keys():
            print(i)





        f.close()








