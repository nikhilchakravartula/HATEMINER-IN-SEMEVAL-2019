import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


def fit_predict(X_train, y_train, X_test, y_test, emr,label,mode):
    print("*****NAIVE BAYES*********")
    cv  = CountVectorizer(min_df=3,max_features=None, 
                             strip_accents='unicode',analyzer='word',
                             token_pattern=r'\w{2,}',
                             ngram_range=(1,2),
                             stop_words='english',
                             binary=True
                             )
    #Vectorizing input
    #print("Transforming train")
    X_train = np.array((cv.fit_transform(X_train).toarray()))
    #print("Transforming test")
    X_test = np.array((cv.transform(X_test).toarray()))



    clf =  MultinomialNB()
    #print("Naive Bayes train")
    scores = cross_val_score(clf,X_train,y_train,cv=5,scoring='f1_macro')
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X_train, y_train):
       X_train1, X_test1 = X_train[train_index], X_train[test_index]
       y_train1, y_test1 = y_train[train_index], y_train[test_index]
       clf.fit(X_train1, y_train1)
       y_pred1 = clf.predict(X_test1)
       emr['nb'][label].append(np.equal(y_test1 , y_pred1))
    

    print("Cross validation average")
    #print(scores)
    print(scores.mean())
   
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if mode!='train':
        print("----DEV SET RESULTS----")
        print("=======================")
        print( confusion_matrix(y_test, y_pred))
        print( classification_report(y_test, y_pred, digits=4))
    return y_pred,emr


