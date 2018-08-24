#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 08:21:58 2018

@author: Carolina
"""
#!pip install --upgrade pip
#!pip3 install pyldavis
import gensim  
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import KMeans
import pyldavis, pyLDAvis.sklearn
from IPython.display import display
import pyLDAvis
import pyLDAvis.gensim

def extract():
    import csv
    import numpy as np
    import pandas as pd


    observations= pd.read_csv('amazon_reviews_us_Sports_v1_00.tsv', sep='\t', header=0, error_bad_lines=False)
    
    observations.drop_duplicates(inplace=True)
    
    observations.dropna(inplace=True)
    
    return observations


observations=extract()

# gruop reviews for same product_id
rev_cont=observations[['product_id','review_body']].groupby(observations['product_id'])

rev_by_prod=pd.DataFrame({'product_id':list(observations['product_id'].unique())})

def get_reviews(product_id):
    """extract all reviews for the same product in a string format
    list(rev_cont.get_group(product_id)['review_body'].values)
    will give a list of reviews
    the output is the joined list of reviews
    """
    return ",".join(list(rev_cont.get_group(product_id)['review_body'].values))
    
rev_by_prod['review_bodies']= rev_by_prod['product_id'].apply(get_reviews)


prod_words =pd.merge(rev_by_prod,observations[['product_id','product_title']],how='left', on='product_id')

prod_words.drop_duplicates(inplace=True)

prod_words['all_words']=prod_words['product_title'] +' '+ prod_words['review_bodies']

# tokenize reviews
prod_words['all_words_tkn'] = prod_words['all_words'].apply(text_to_word_sequence)

#remove stop words
stop_words = set(stopwords.words('english'))

def stpout(lst):
    return [word for word in lst if word not in stop_words]

prod_words['all_words_tkn']  = prod_words['all_words_tkn'].apply(stpout)

pickle.dump(prod_words, open('prod_words.pkl', 'wb'))


#model

# create dictionary
dictionary = gensim.corpora.Dictionary(list(prod_words['all_words_tkn'].values))

dictionary2=dictionary

#filter less than no_below documents (absolute number) 
#or more than no_above documents (fraction of total corpus size, not absolute number).

dictionary.filter_extremes(no_below=5, no_above=0.3)

values= list(prod_words['all_words_tkn'].values)

corpus = [dictionary.doc2bow(text) for text in values]

#gensim.corpora.MmCorpus.serialize('title_prod.mm', corpus)

#corpus = gensim.corpora.MmCorpus('title_prod.mm')

lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=5)

vis = pyLDAvis.gensim.prepare(lda,corpus,dictionary) 
pyLDAvis.save_html(vis, 'lda.html')

lda.print_topics()

lda_vect=[]
for i in range(len(corpus)):
    i=0
    lda_vect.append(lda[corpus[i]])


### loking closer into topics

tp_prob=[]
topic=[]
prob=[]
for i in range(len(prod_words)):
    tp_prob.append(max(lda[corpus[i]], key=lambda x: x[1]))
    topic.append(tp_prob[i][0])
    prob.append(tp_prob[i][1])

lda_info=pd.DataFrame({'product_id':list(prod_words['product_id']),'prod_title':list(prod_words['product_title']),'lda_topic':topic, 'lda_prob':prob})

lda_info.describe().T

topics=lda_info

topics=topics[topics['lda_prob']>=0.9]

topics0=topics[topics['lda_topic']==0]

topics1=topics[topics['lda_topic']==1]

topics2=topics[topics['lda_topic']==2]

topics3=topics[topics['lda_topic']==3]

topics4=topics[topics['lda_topic']==4]

##spliting  topic 4 in 5 separate subtopics, many observations in a topic

words_topic4=pd.merge(topics4,prod_words[['product_id','all_words_tkn']],how='left', on='product_id')

dictionary4 = gensim.corpora.Dictionary(list(words_topic4['all_words_tkn'].values))

dictionary4.filter_extremes(no_below=5, no_above=0.3)

values= list(words_topic4['all_words_tkn'].values)

corpus4 = [dictionary4.doc2bow(text) for text in values]

lda4 = gensim.models.LdaModel(corpus4, id2word=dictionary4, num_topics=5)

lda4.print_topics()

tp_prob=[]
topic=[]
prob=[]

for i in range(len(words_topic4)):
    tp_prob.append(max(lda4[corpus4[i]], key=lambda x: x[1]))
    topic.append(tp_prob[i][0])
    prob.append(tp_prob[i][1])

lda4_info=pd.DataFrame({'product_id':list(words_topic4['product_id']),'prod_title':list(words_topic4['prod_title']),'lda4_topic':topic, 'lda4_prob':prob})

lda4_info.describe().T

##spliting  topic 3 in 5 separate subtopics

words_topic3=pd.merge(topics3,prod_words[['product_id','all_words_tkn']],how='left', on='product_id')

dictionary3 = gensim.corpora.Dictionary(list(words_topic3['all_words_tkn'].values))

dictionary3.filter_extremes(no_below=5, no_above=0.3)

values= list(words_topic3['all_words_tkn'].values)

corpus3 = [dictionary3.doc2bow(text) for text in values]

lda3 = gensim.models.LdaModel(corpus3, id2word=dictionary3, num_topics=5)

lda3.print_topics()

tp_prob=[]
topic=[]
prob=[]

for i in range(len(words_topic3)):
    tp_prob.append(max(lda3[corpus3[i]], key=lambda x: x[1]))
    topic.append(tp_prob[i][0])
    prob.append(tp_prob[i][1])

lda3_info=pd.DataFrame({'product_id':list(words_topic3['product_id']),'prod_title':list(words_topic3['prod_title']),'lda3_topic':topic, 'lda3_prob':prob})

lda3_info.describe().T

pickle.dump(topics, open('topics.pkl', 'wb'))


## trying to pick a topic to do the price estimates
#estimating average number of words per topic
#topic0
words_topic0=pd.merge(topics0,prod_words[['product_id','all_words_tkn']],how='left', on='product_id')

words_topic0['n_words']=words_topic0['all_words_tkn'].apply(len)

avg0=np.average(words_topic0['n_words'])

#topic1
words_topic1=pd.merge(topics1,prod_words[['product_id','all_words_tkn']],how='left', on='product_id')

words_topic1['n_words']=words_topic1['all_words_tkn'].apply(len)

avg1=np.average(words_topic1['n_words'])

avg_prob1=np.average(words_topic1['lda_prob'])

words_topic1.describe()

#topic2
words_topic2=pd.merge(topics2,prod_words[['product_id','all_words_tkn']],how='left', on='product_id')

words_topic2['n_words']=words_topic2['all_words_tkn'].apply(len)

avg2=np.average(words_topic2['n_words'])

avg_prob2=np.average(words_topic2['lda_prob'])

words_topic2.describe()

#topic3
words_topic3['n_words']=words_topic3['all_words_tkn'].apply(len)

avg3=np.average(words_topic3['n_words'])

#topic4
words_topic4['n_words']=words_topic4['all_words_tkn'].apply(len)

avg4=np.average(words_topic4['n_words'])

values= list(words_topic3['all_words_tkn'].values)

#spliting topic 1 into 5 topics
dictionary1 = gensim.corpora.Dictionary(list(words_topic1['all_words_tkn'].values))

dictionary1.filter_extremes(no_below=5, no_above=0.3)

values= list(words_topic1['all_words_tkn'].values)

corpus1 = [dictionary1.doc2bow(text) for text in values]

lda1 = gensim.models.LdaModel(corpus1, id2word=dictionary1, num_topics=5)

lda1.print_topics()

tp_prob=[]
topic=[]
prob=[]

for i in range(len(words_topic1)):
    tp_prob.append(max(lda1[corpus1[i]], key=lambda x: x[1]))
    topic.append(tp_prob[i][0])
    prob.append(tp_prob[i][1])

lda1_info=pd.DataFrame({'product_id':list(words_topic1['product_id']),'prod_title':list(words_topic1['prod_title']),'lda_topic':topic, 'lda_prob':prob})

lda1_info.describe().T 

#spliting topic 0 into 5 topics
dictionary0 = gensim.corpora.Dictionary(list(words_topic0['all_words_tkn'].values))

dictionary0.filter_extremes(no_below=5, no_above=0.3)

values= list(words_topic0['all_words_tkn'].values)

corpus0 = [dictionary0.doc2bow(text) for text in values]

lda0 = gensim.models.LdaModel(corpus0, id2word=dictionary0, num_topics=5)

lda0.print_topics()

tp_prob=[]
topic=[]
prob=[]

for i in range(len(words_topic0)):
    tp_prob.append(max(lda0[corpus0[i]], key=lambda x: x[1]))
    topic.append(tp_prob[i][0])
    prob.append(tp_prob[i][1])

lda0_info=pd.DataFrame({'product_id':list(words_topic0['product_id']),'prod_title':list(words_topic0['prod_title']),'lda_topic':topic, 'lda_prob':prob})

lda0_info.describe()

## spliting topic o in 5 subtopics
dictionary2 = gensim.corpora.Dictionary(list(words_topic2['all_words_tkn'].values))

dictionary2.filter_extremes(no_below=5, no_above=0.3)

values= list(words_topic2['all_words_tkn'].values)

corpus2 = [dictionary2.doc2bow(text) for text in values]

lda2 = gensim.models.LdaModel(corpus2, id2word=dictionary2, num_topics=5)

lda2.print_topics()

tp_prob=[]
topic=[]
prob=[]

for i in range(len(words_topic2)):
    tp_prob.append(max(lda2[corpus2[i]], key=lambda x: x[1]))
    topic.append(tp_prob[i][0])
    prob.append(tp_prob[i][1])

lda2_info=pd.DataFrame({'product_id':list(words_topic2['product_id']),'prod_title':list(words_topic2['prod_title']),'lda_topic':topic, 'lda_prob':prob})

lda2_info.describe()

topics21=lda2_info


topics21=topics21[topics21['lda_prob']>=0.9]

topics210=topics21[topics21['lda_topic']==0]

topics211=topics21[topics21['lda_topic']==1]

topics212=topics21[topics21['lda_topic']==2]

topics213=topics21[topics21['lda_topic']==3]

topics214=topics21[topics21['lda_topic']==4]

topics210.describe()
topics211.describe()
topics212.describe()
topics213.describe()
topics214.describe()

    
""" verifying that the  LDA topics matches Kmeans """


# organizing LDA output into vectors

lda_vect3=[]
for i in range(len(lda_vect)):
    lda_vect2=[]   
    for j in range(5):
        flag=0 
        
        for k in range(len(lda_vect[i])):
            if (lda_vect[i][k][0]==0) and j==0:
                lda_vect2.append(lda_vect[i][k][1])
                flag=1
            elif (lda_vect[i][k][0]==1) and j==1:
                lda_vect2.append(lda_vect[i][k][1])
                flag=1
            elif (lda_vect[i][k][0]==2) and j==2:
                lda_vect2.append(lda_vect[i][k][1])
                flag=1
            elif (lda_vect[i][k][0]==3) and j==3:
                lda_vect2.append(lda_vect[i][k][1])
                flag=1
            elif (lda_vect[i][k][0]==4) and j==4:
                lda_vect2.append(lda_vect[i][k][1])
                flag=1
        if flag==0:
            lda_vect2.append((0))           
            
    lda_vect3.append(lda_vect2)  

X_lda=np.array(lda_vect3)

kmeans = KMeans(n_clusters=5, random_state=0).fit(X_lda)

lda_info[lda_info['lda_topic']==4].describe()
# examining results for kmeans

lda_info['kmean_label']=pd.DataFrame({'kmean_label':list(kmeans.labels_)})

lda_info[(lda_info['lda_topic']==0) & (lda_info['kmean_label']==4)].count()/lda_info[lda_info['lda_topic']==0].count()

lda_info[(lda_info['lda_topic']==1) & (lda_info['kmean_label']==3)].count()/lda_info[lda_info['lda_topic']==1].count()

lda_info[(lda_info['lda_topic']==2) & (lda_info['kmean_label']==1)].count()/lda_info[lda_info['lda_topic']==2].count()

lda_info[(lda_info['lda_topic']==3) & (lda_info['kmean_label']==2)].count()/lda_info[lda_info['lda_topic']==3].count()

lda_info[(lda_info['lda_topic']==4) & (lda_info['kmean_label']==0)].count()/lda_info[lda_info['lda_topic']==4].count()


pickle.dump(lda2_info, open('lda2_info.pkl', 'wb'))

pickle.dump(words_topic2, open('words_topic2.pkl', 'wb'))

words_topic2_all=pd.merge(words_topic2,prod_words,how='left', on='product_id')

pickle.dump(words_topic2_all, open('words_topic2_all.pkl', 'wb'))

#visualizations

pyLDAvis.enable_notebook()

# Create the visualization
vis = pyLDAvis.gensim.prepare(lda,corpus,dictionary) ## too big for my computer
#lda_input=[lda0,lda1,lda2,lda3,lda4]
#cospus_input= [corpus0,corpus1,corpus2,corpus3,corpus4]
#dictionay_input=[dictionary0,dictionary1,dictionary2,dictionary3,dictionary4]

#vis = pyLDAvis.gensim.prepare(lda0,corpus0,dictionary0)

#display_inputs=pd.DataFrame({'lda':lda_input,'corpus':cospus_input,'dictionary':dictionay_input})
                           
#pickle.dump(display_inputs, open('display_inputs.pkl', 'wb'))

pyLDAvis.save_html(vis, 'lda.html')