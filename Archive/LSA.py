#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 21:56:23 2018

@author: carolina
"""

from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
import nltk  
from sklearn import metrics
from string import punctuation
from nltk.stem.porter import PorterStemmer
from sklearn import naive_bayes
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report


n_features = 1000
n_lsa_comp=150


stemmer = PorterStemmer()

def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in punctuation]
    
    stemmed = []
    for word in tokens:
        stemmed.append(stemmer.stem(word))
    
    return stemmed



y=words_topicwprice['price']
X=words_topicwprice['all_words']


def lsa_reg(X,y,n_lsa_comp,n_features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    
    
    tf_vectorizer = TfidfVectorizer(max_features=n_features,
                                        ngram_range=(1,2),
                                        stop_words='english',
                                        tokenizer=tokenize)
    
    #dtm: Document-Term Matrix
    tf_train = tf_vectorizer.fit_transform(X_train)
    
    tf_test = tf_vectorizer.transform(X_test)
    
    lsa = TruncatedSVD(n_lsa_comp, algorithm = 'arpack')
    
    dtm_lsa_train = lsa.fit_transform(tf_train)
    #dtm_lsa_train = Normalizer(copy=False).fit_transform(dtm_lsa_train)
    
    dtm_lsa_test = lsa.transform(tf_test)
    #dtm_lsa_test = Normalizer(copy=False).transform(dtm_lsa_test)
    
    
    explained_var=np.sum(lsa.explained_variance_ratio_)
    
    d = defaultdict(list)
    for i in range(len(dtm_lsa_train.T)):
        k='comp{}'.format(i)
        v= list(dtm_lsa_train.T[i])
        d[k]=v
    
    df_lsa_train=pd.DataFrame(d)  
    
    d = defaultdict(list)
    for i in range(len(dtm_lsa_test.T)):
        k='comp{}'.format(i)
        v= list(dtm_lsa_test.T[i])
        d[k]=v
    
    df_lsa_test=pd.DataFrame(d) 
    
    
    pipeline=Pipeline([
        ('poly',preprocessing.PolynomialFeatures(degree=2)),
        ('standardscaler',StandardScaler(copy=True, with_mean=True, with_std=True)),
        ('lasso',LassoCV(cv=3,alphas=(0.5,1,10,100)))])
    
    
    pipeline.fit(df_lsa_train,y_train)
    
    y_pred_train_lsa=pipeline.predict(df_lsa_train)
    
    r2_train_lsa=metrics.r2_score(y_train, y_pred_train_lsa)
    
    y_pred_test_lsa=pipeline.predict(df_lsa_test)
    
    r2_test_lsa=metrics.r2_score(y_test, y_pred_test_lsa)

return [explained_var, r2_train_lsa, r2_test_lsa, n_lsa_comp, n_features]

resultslist=[]

resultsl=lsa_reg(X,y,150,5000)

resultslist.append(resultsl)

res=np.array(resultslist)

lsa_results=pd.DataFrame({'n_features':res.T[4],
                         'n_lsa_comp':res.T[3],
                         'explained_var':res.T[0],
                         'r2_train_lsa':res.T[1],
                         'r2_test_lsa':res.T[2]}) 

