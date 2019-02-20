import pandas as pd
import preprocessor as p
from nltk.stem import PorterStemmer as ps
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import TweetTokenizer
import naive_bayes
import svm
import nltk
import random_forest
import xg_boost
nltk.download('punkt')
import argparse
import word_embeddings as we
from sklearn.naive_bayes import MultinomialNB
import os
import logreg
import string
import random_forest
import libinfersent
import re
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import libekphrasis as ek
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
import csv

def stemmer(stemmerObj,sentence):
    twtknr = TweetTokenizer()
    #print(sentence)
    if sentence is None or isinstance(sentence,str)==False:
        return ' '
    sentence = [stemmerObj.stem(word) for word in twtknr.tokenize(sentence)]
    return " ".join(str(x) for x in sentence)


def replace(word):
    if word.lower() in contractions.contractions:
        return contractions.contractions[word.lower()]
    return word

def replaceContractions(sentence):
    twtknr = TweetTokenizer()
    sentence = [replace(word) for word in twtknr.tokenize(sentence)]
    return " ".join(str(x) for x in sentence)

def removePunc(sentence,translator):
    return  sentence.translate(translator)

def removeNums(sentence,translator):
    return sentence.translate(translator)



 
def preprocess(train,test):

    if preprocess==False:
        return train['text'],test['text']
    
    pre_train_fn = './preprocess_train.tsv'
    pre_dev_fn = './preprocess_test.tsv'
    if os.path.isfile(pre_train_fn) == True and os.path.isfile(pre_dev_fn) == True:
        return pre_train_fn,pre_dev_fn
    
    x_train = train['text']
    y_train = train['HS']

    x_test = test['text']
    y_test = test['HS']
    print(x_test.head())
    p.set_options(p.OPT.URL,p.OPT.MENTION,p.OPT.EMOJI,p.OPT.SMILEY)
    x_train = [p.clean(sentence) for sentence in x_train]
    x_test = [p.clean(sentence) for sentence in x_test]

    #stemmerObj = ps()   

    x_train = [replaceContractions(sentence) for sentence in x_train]
    x_test = [replaceContractions(sentence) for sentence in x_test]

    translator =  str.maketrans(string.punctuation.replace('#',''),' '*len(string.punctuation.replace('#','')))

    x_train = [removePunc(sentence,translator) for sentence in x_train]
    x_test = [removePunc(sentence,translator) for sentence in x_test]

    translator = str.maketrans(string.digits,' '*len(string.digits)) 
    x_train = [removeNums(sentence,translator) for sentence in x_train]
    x_test  = [removeNums(sentence,translator) for sentence in x_test]

    
    x_train = [ " ".join(ek.text_processor.pre_process_doc(sentence)) for sentence in x_train]
    x_test = [ " ".join(ek.text_processor.pre_process_doc(sentence)) for sentence in x_test]
    
    #x_train = [ stemmer(stemmerObj,sentence) for sentence in x_train ]
    #x_test = [ stemmer(stemmerObj,sentence) for sentence in x_test]

    
    train['text'] = pd.Series(x_train)
    test['text']= pd.Series(x_test)

    if os.path.isfile(pre_train_fn) == False:
        f = open(pre_train_fn,"w")
        train.to_csv(f,sep=',')
    if os.path.isfile(pre_dev_fn) == False:
        f = open(pre_dev_fn,"w")
        test.to_csv(f,sep=',')
    print("Completed writing preprocessed files")
    return pre_train_fn,pre_dev_fn

def EMR(ytruths,ypreds):
    count=0
    ytrue1 = ytruths[0]
    ytrue2 = ytruths[1]
    ytrue3 = ytruths[2]
    ypred1,ypred2,ypred3 = ypreds[0],ypreds[1]

    for i in range(len(ypred1)):
        if ypred1[i]==ytrue1[i] and ypred2[i]==ytrue2[i] and ypred3[i]==ytrue3[i]:
            count = count+1

    print(count)

def printLabelSeperator(label):
    print("-----------------------   "+label+"   -----------------------")
    print("=============================================================")

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-clf","--classifier", help="echo the classifier you use here",action="store",default="logreg",choices=['nb','logreg','svm','xgb','all'])
    parser.add_argument("-emsrc","--emsource", help="echo the source of embedding you use here",action="store",default="",choices=['glove','fasttext','infersent'])
    parser.add_argument("-pre","--preprocess",help="To tell if preprocess has to happen to text",action="store", default="F",choices=['T','F'])
    parser.add_argument("-op","--writetofile",help="Writes to the output file",action="store",default="F")
    parser.add_argument("-label","--label",help="Which label to calculate on",action="store",default="all",choices=['HS','TR','AG'])
    parser.add_argument("-train","--train",help="Train file",action="store",default="")
    parser.add_argument("-test","--test",help="Test file",action="store",default="")
    parser.add_argument("-sep","--seperator",help="Seperator in train and test file",action="store",default="\t")
    parser.add_argument("-mode","--mode",help="Mode to execute in. Train mode: Do not validate on dev set. dev mode: Validate on dev set",action="store",default="train",choices=['train','dev'])
    args = parser.parse_args()
   
    trainFile = args.train
    testFile = args.test
    seperator = args.seperator

    train = pd.read_csv(trainFile,sep=seperator)
    test = pd.read_csv(testFile,sep=seperator)
    


    print("Args preprocess is")
    print(args.preprocess)
    if args.preprocess is not None and args.preprocess=="T":
        print("Preprocessing the dataset")
        trainFile,testFile = preprocess(train,test)
        train = pd.read_csv(trainFile,sep=',')
        test = pd.read_csv(testFile,sep=',')
    train.fillna('abcxyz',inplace=True)
    test.fillna('abcxyz',inplace=True)
    results ={'svm':[],'logreg':[],'nb':[],'xgb':[]}
    emr = {'svm':{'HS':[],'TR':[],'AG':[]},'logreg':{'HS':[],'TR':[],'AG':[]},'nb':{'HS':[],'TR':[],'AG':[]},'xgb':{'HS':[],'TR':[],'AG':[]}}
        


    params ={'emsource':args.emsource,'X_train':train['text'],'X_test':test['text'],'Y_train':train['HS'],'Y_test':test['HS'],'ipFile':trainFile}
    X_train,Y_train,X_test,Y_test = we.getEmbeddings(params)
    lab = 'HS'
    printLabelSeperator(lab)
    if args.classifier == "logreg" or args.classifier=="all":
        ypred,emr = logreg.fit_predict(X_train,Y_train,X_test,Y_test,emr,lab,args.mode)
        results['logreg'].append(ypred)
    if args.classifier=="svm" or args.classifier=="all":
        ypred,emr = svm.fit_predict(X_train,Y_train,X_test,Y_test,emr,lab,args.mode)
        results['svm'].append(ypred)
    if args.classifier=="nb":
        ypred,emr = naive_bayes.fit_predict(X_train,Y_train,X_test,Y_test,emr,lab,args.mode)
        results['nb'].append(ypred)
    if args.classifier=="xgb" or args.classifier=="all":
        ypred,emr = xg_boost.fit_predict(X_train,Y_train,X_test,Y_test,emr,lab,args.mode)
        results['xgb'].append(ypred)


    Y_train = train['TR']
    Y_test = test['TR'] 

    #Classify TR
    lab = 'TR'
    printLabelSeperator(lab)
    if args.classifier == "logreg" or args.classifier=="all":
        ypred,emr = logreg.fit_predict(X_train,Y_train,X_test,Y_test,emr,lab,args.mode)
        results['logreg'].append(ypred)
    if args.classifier=="svm" or args.classifier=="all":
        ypred,emr = svm.fit_predict(X_train,Y_train,X_test,Y_test,emr,lab,args.mode)
        results['svm'].append(ypred)
    if args.classifier=="nb":
        ypred,emr = naive_bayes.fit_predict(X_train,Y_train,X_test,Y_test,emr,lab,args.mode)
        results['nb'].append(ypred)
    if args.classifier=="xgb" or args.classifier=="all":
        ypred,emr = xg_boost.fit_predict(X_train,Y_train,X_test,Y_test,emr,lab,args.mode)
        results['xgb'].append(ypred)




    lab = 'AG'
    printLabelSeperator(lab)
    Y_train = train['AG']
    Y_test = test['AG']
    if args.classifier == "logreg" or args.classifier=="all":
        ypred,emr = logreg.fit_predict(X_train,Y_train,X_test,Y_test,emr,lab,args.mode)
        results['logreg'].append(ypred)
    if args.classifier=="svm" or args.classifier=="all":
        ypred,emr = svm.fit_predict(X_train,Y_train,X_test,Y_test,emr,lab,args.mode)
        results['svm'].append(ypred)
    if args.classifier=="nb":
        ypred,emr = naive_bayes.fit_predict(X_train,Y_train,X_test,Y_test,emr,lab,args.mode)
        results['nb'].append(ypred)
    if args.classifier=="xgb" or args.classifier=="all":
        ypred,emr = xg_boost.fit_predict(X_train,Y_train,X_test,Y_test,emr,lab,args.mode)
        results['xgb'].append(ypred)

    count=0
    ytrue1 = test['HS']
    ytrue2 = test['TR']
    ytrue3 = test['AG']

   
    print("Cross validation average for EMR") 
    for key1 in emr.keys():
        score = 0.0
        #print(key1,emr[key1])
        if len(emr[key1]['HS'])>0:
            for i in range(0,5):
                score += ((np.array(emr[key1]['HS'][i])==np.array(emr[key1]['TR'][i]))==np.array(emr[key1]['AG'][i])).sum()
            print(key1 , score/len(X_train))

    if args.mode!='train':
        print("Dev set EMR")
        for key in results.keys():
            count = 0
            if len(results[key])!=0:
                ypred1,ypred2,ypred3 = pd.Series(results[key][0]),pd.Series(results[key][1]),pd.Series(results[key][2])
                for i in range(len(ypred1)):
                    if ypred1[i]==ytrue1[i] and ypred2[i]==ytrue2[i] and ypred3[i]==ytrue3[i]:
                        count = count+1
                print(key,count/len(ypred1))
                break; 



    if args.writetofile !="F":
        test.drop(['text','HS','TR','AG'],inplace=True,axis=1)
        key = args.classifier
        ypred1,ypred2,ypred3 = results[key][0],results[key][1],results[key][2]
        print(args.label)
        if args.label =="all":
            test = pd.concat([test['id'],pd.Series(ypred1),pd.Series(ypred2),pd.Series(ypred3)],axis=1)
        if args.label =="HS":
            test = pd.concat([test['id'],pd.Series(ypred1)],axis=1)
        test.to_csv(args.writetofile,sep="\t",index=False,header=False)

    exit(0)
