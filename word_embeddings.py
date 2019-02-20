from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
import fasttext
import os
import pandas as pd
import numpy as np
#import libinfersent as infersent
#import libglove as glove
#import libglovetwitter as glove
#import libfasttext as fasttext
#import fasttext
def getEmbeddings(params):

    X_train = params['X_train']
    X_test = params['X_test']
    Y_train = params['Y_train']
    Y_test = params['Y_test']
    
    if params['emsource'] =="fasttext":
        X_train = pd.DataFrame(fasttext.get_vectors(X_train.tolist()))
        X_test = pd.DataFrame(fasttext.get_vectors(X_test.tolist()))
    elif params['emsource']=="glove":
        X_train = pd.DataFrame(glove.get_vectors(X_train.tolist()))
        X_test = pd.DataFrame(glove.get_vectors(X_test.tolist()))
    elif params['emsource']=='infersent':
        X_train  = pd.DataFrame(infersent.get_vectors(X_train.tolist()) )
        X_test  = pd.DataFrame(infersent.get_vectors(X_test.tolist()) )
    print(X_train.head())
    
    X_train.fillna(0,inplace=True)
    X_test.fillna(0,inplace=True)
        
    return X_train,Y_train,X_test,Y_test
