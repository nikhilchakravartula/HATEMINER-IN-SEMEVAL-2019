import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
def fit_predict(X_train, y_train, X_test, y_test, emr,label,mode):

    print("***********LOGREG*************")
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test) 
    clf = linear_model.LogisticRegression(class_weight = 'balanced',C=0.5,penalty='l2')
    skf = StratifiedKFold(n_splits=5)
    for train_index, test_index in skf.split(X_train, y_train):
       X_train1, X_test1 = X_train[train_index], X_train[test_index]
       y_train1, y_test1 = y_train[train_index], y_train[test_index]
       clf.fit(X_train1, y_train1)
       y_pred1 = clf.predict(X_test1)
       emr['logreg'][label].append(np.equal(y_test1 , y_pred1))

    scores = cross_val_score(clf,X_train,y_train,cv=5,scoring='f1_macro')
    print("Cross validation average")
    print(scores.mean())
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if mode!='train':
        print("----DEV SET RESULTS----")
        print("=======================")
        print( confusion_matrix(y_test, y_pred))
        print( classification_report(y_test, y_pred, digits=4))
    return y_pred,emr

