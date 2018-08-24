#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 05:46:38 2018

@author: carolina
"""
import numpy as np
import pandas as pd
import pickle
from nltk.corpus import brown
import nltk
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn import metrics
from sklearn.pipeline import Pipeline
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')


with open('words_topic2_all.pkl', 'rb') as handle:
    words_topic2_all= pickle.load(handle)


 
observations = extract()  

speech=words_topic2_all

speech.info()

speech['tags_all']=speech['all_words_tkn_x'].apply(nltk.pos_tag)

def tag_fd(tagged_text):
    """ Fuction returns list of frequencies of most commonly used part of speech"""

    tag_fd = nltk.FreqDist(tag for (word, tag) in tagged_text)
    
    return tag_fd.most_common()

speech['tags_all_fq']=speech['tags_all'].apply(tag_fd) 

# the command below helps look at the tag names and meaning.
nltk.help.upenn_tagset()

def n_verbs(tag_fd):
    """ sum all verbs VB, VBD, VGB,VBN,VBP,VBZ """
    verbs=0
    for i in range(len(tag_fd)):   
        
        if tag_fd[i][0]=='VB': verbs =tag_fd[i][1]
        if tag_fd[i][0]=='VBD':verbs =tag_fd[i][1]+verbs
        if tag_fd[i][0]=='VGB':verbs =tag_fd[i][1]+verbs
        if tag_fd[i][0]=='VBN':verbs =tag_fd[i][1]+verbs
        if tag_fd[i][0]=='VBP':verbs =tag_fd[i][1]+verbs
        if tag_fd[i][0]=='VBZ':verbs =tag_fd[i][1]+verbs
    return verbs       
        

def n_adj(tag_fd):
    """ sum all verbs JJ, JJR, JJS """
    adj=0
    for i in range(len(tag_fd)):   
        
        if tag_fd[i][0]=='JJ': adj =tag_fd[i][1]
        if tag_fd[i][0]=='JJR':adj =tag_fd[i][1]+adj
        if tag_fd[i][0]=='JJS':adj =tag_fd[i][1]+adj
    return adj
    
def n_numeral(tag_fd):
    """ CD: numeral, cardinal"""  
    CD=0
    for i in range(len(tag_fd)):    
        if tag_fd[i][0]=='CD': CD =tag_fd[i][1]
    
    return CD

def n_nouns(tag_fd):
    """ Sums all nouns NN, NNP, NNPS, NNS"""
    nouns=0
    for i in range(len(tag_fd)):   
        
        if tag_fd[i][0]=='NN': nouns =tag_fd[i][1]
        if tag_fd[i][0]=='NNP':nouns =tag_fd[i][1]+nouns
        if tag_fd[i][0]=='NNPS':nouns =tag_fd[i][1]+nouns
        if tag_fd[i][0]=='NNS':nouns =tag_fd[i][1]+nouns
    return nouns
 
speech['n_verbs']=speech['tags_all_fq'].apply(n_verbs)
speech['n_adj']=speech['tags_all_fq'].apply(n_adj)
speech['n_numeral']=speech['tags_all_fq'].apply(n_numeral)
speech['n_nouns']=speech['tags_all_fq'].apply(n_nouns)

speech['p_verbs']=speech['n_verbs']/speech['n_words']
speech['p_adj']=speech['n_adj']/speech['n_words']
speech['p_numeral']=speech['n_numeral']/speech['n_words']
speech['p_nouns']=speech['n_nouns']/speech['n_words']

gp=(observations[['product_id','star_rating','helpful_votes','total_votes','verified_purchase']]
                         .groupby(['product_id']))    

review_info=pd.DataFrame({'product_id':list(gp.groups.keys())})


Total_votes =gp['total_votes'].sum()

review_info['total_votes']=pd.DataFrame({'total_votes':list(Total_votes.values)})

review_info['helpful_votes']=pd.DataFrame({'total_votes':list(gp['helpful_votes'].sum().values)})

review_info['verified_purchase']=pd.DataFrame({'verified_purchase':list(gp['verified_purchase'].count().values)})

observations['star_rating']=pd.to_numeric(observations['star_rating'])

review_info['star_rating']=pd.DataFrame({'star_rating':list(gp['star_rating'].mean().values)})

review_info.info()

import json

with open('data.json', encoding='utf-8-sig') as json_file:
    data = json.load(json_file)
    
    data2=[]
for item in data:
    if type(item)==dict:
        data2.append(item)
        
price=[]
product_id=[]
name=[]
category=[]
availability=[]
original_price=[]

for i in range(len(data2)):
    if data2[i]['SALE_PRICE']==None:
        price.append(np.nan)
        
    else:
        p=data2[i]['SALE_PRICE'].replace(',','')
        ind=p.find('-')
        if ind==-1:
            price.append(float(p[1:]))
        else:
            price.append(float(p[1:ind]))
    
    product_id.append(data2[i]['URL'][-10:])
    name.append(data2[i]['NAME'])
    
    if data2[i]['CATEGORY']==None:
        category.append(np.nan)
    else: category.append(data2[i]['CATEGORY'])
    
    availability.append(data2[i]['AVAILABILITY'])
    original_price.append(data2[i]['ORIGINAL_PRICE'])

price_inc=pd.DataFrame({'product_id':product_id,
                        'name':name,
                        'category':category,
                        'availability':availability,
                        'price':price,
                        'original_price':original_price})



review_info_wprice=pd.merge(price_inc,review_info,how='left', on='product_id')    

review_info_wprice=pd.merge(review_info_wprice,speech,how='left', on='product_id')    

review_info_wprice.dropna(inplace=True)


y=review_info_wprice['price']
X=review_info_wprice[['star_rating','helpful_votes','total_votes','verified_purchase','p_verbs','p_adj','p_numeral','p_nouns','n_words']]
  

  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

pipeline=Pipeline([('poly',PolynomialFeatures(degree=2)),
                    ('standardscaler',StandardScaler(copy=True, with_mean=True, with_std=True)),
                    ('lasso',LassoCV(cv=3,alphas=(0.5,1,10,100)))])


pipeline.fit(X_train,y_train)

y_train_pred=pipeline.predict(X_train)

r2_train=metrics.r2_score(y_train, y_train_pred)

y_pred_test=pipeline.predict(X_test)

r2_test_lsa=metrics.r2_score(y_test, y_pred_test)
