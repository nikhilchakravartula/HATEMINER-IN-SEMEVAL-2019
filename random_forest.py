import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
def fit_predict(X_train, y_train, X_test, y_test, emr,label):
    """tfidf = TfidfVectorizer(min_df=3,max_features=None, 
                             strip_accents='unicode',analyzer='word',
                             token_pattern=r'\w{1,}',ngram_range=(1,2),
                             use_idf=1,smooth_idf=1,stop_words='english',
                             )
    print("Transforming train")
    X_train = tfidf.fit_transform(X_train)
    print("Transforming test")
    X_test = tfidf.transform(X_test)"""
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    print("RF train")
    print("#Features=%d #instances=%d" % (X_train.shape[1], X_train.shape[0]))
    clf.fit(X_train, y_train)
    feature_importances = pd.DataFrame(clf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
    print(feature_importances)
    print("RF predict")
    y_pred = clf.predict(X_test)
    print( confusion_matrix(y_test, y_pred))
    print( classification_report(y_test, y_pred, digits=4))
    #Dump the model
    #pickle_file = 'svm_model.pickle'
    #pickle.dump(clf, open(pickle_file,"wb"))
    weights=[]
    """id2word={v: k for k, v in tfidf.vocabulary_.items()}
    print(clf.coef_.shape)
    print(clf.intercept_)
    for index, weight in enumerate(clf.coef_[0].tolist()):
        weights.append((index, weight))
    print(len(tfidf.vocabulary_))
    top_weights = sorted(weights, key=lambda x:x[1], reverse=True)[:100]
    words_weights = [(id2word[wid], weight) for wid, weight in top_weights]
    print(words_weights)
    top_weights = sorted(weights, key=lambda x:x[1], reverse=True)[-100:]
    words_weights = [(id2word[wid], weight) for wid, weight in top_weights]
    print(words_weights)"""
    return y_pred

