import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold



def fit_predict(X_train, y_train, X_test, y_test,emr,label,mode):
    print("**********SVM************")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    kernels = ['poly','rbf','linear']
    for kernel in [kernels[1]]:
        clf = svm.SVC(kernel=kernel,random_state=0, tol=1e-5)
        skf = StratifiedKFold(n_splits=5)
        for train_index, test_index in skf.split(X_train, y_train):
           X_train1, X_test1 = X_train[train_index], X_train[test_index]
           y_train1, y_test1 = y_train[train_index], y_train[test_index]
           clf.fit(X_train1, y_train1)
           y_pred1 = clf.predict(X_test1)
           emr['svm'][label].append(np.equal(y_test1 , y_pred1))
        
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
