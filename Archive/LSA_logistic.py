#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:21:31 2018

@author: carolina
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:16:23 2018

@author: carolina
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc

# estimating categories

cat_count=list(range(3))

cat_count[2]=y[y>=100].count()

cat_count[1]=y[y<100].count()-y[y<=20].count()

cat_count[0]=y[y<20].count()

#Assigning categories


cat=[]
for i in range(len(y)):
    p= y.iloc[i]
    if p>=100:cat.append('>=100')
    elif (p<100) and (p>=20):cat.append('100-20')
    elif p<20:cat.append('>20')


y_cat=pd.DataFrame({'price':y.values, 'cat':cat})


y=y_cat['cat']
X=words_topicwprice['all_words']


#def lsa_reg(X,y,n_lsa_comp,n_features):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

n_features = 1000
n_lsa_comp=30

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

####Naive-Bayes
bayes= naive_bayes.GaussianNB() # The likelihood of the features is assumed to be Gaussian


bayes.fit(df_lsa_train,y_train)

y_pred=bayes.predict(df_lsa_test)
naive_conf=confusion_matrix(y_test, y_pred)
naive_conf


print(classification_report(y_test, y_pred))

y_pred_train_lsa=bayes.predict(df_lsa_train)
    
acc_train_lsa=accuracy_score(y_train, y_pred_train_lsa)
    
y_pred_test_lsa=bayes.predict(df_lsa_test)
    
acc_test_lsa=metrics.accuracy_score(y_test, y_pred_test_lsa)

#splitting in categories
#cat1: >20

ind=y[y=='>20'].index

ind_train=y_train.index

ind_test=y_test.index

cat1=[]
y_testcat1=[]
y_traincat1=[]

for i in range(len(y)):
    if i in ind_train:
       if i in ind:y_traincat1.append(1)
       else:y_traincat1.append(0)
    else:
       if i in ind:y_testcat1.append(1)
       else:y_testcat1.append(0)


ind=y[y=='100-20'].index

cat2=[]
y_testcat2=[]
y_traincat2=[]

for i in range(len(y)):
    if i in ind_train:
       if i in ind:y_traincat2.append(1)
       else:y_traincat2.append(0)
    else: 
       if i in ind:y_testcat2.append(1)
       else:y_testcat2.append(0)
    

cat3=[]
y_testcat3=[]
y_traincat3=[]
ind=y[y=='>=100'].index

for i in range(len(y)):
    if i in ind_train:
       if i in ind:y_traincat3.append(1)
       else:y_traincat3.append(0)
    else: 
       if i in ind:y_testcat3.append(1)
       else:y_testcat3.append(0)
    

    
y_train_cat=pd.DataFrame({'cat1':y_traincat1, 'cat2':y_traincat2, 'cat3':y_traincat3})

y_test_cat=pd.DataFrame({'cat1':y_testcat1,'cat2':y_testcat2,'cat3':y_testcat3})

y=y_test_cat['cat1']

# logistic regression
#cat1
tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty':['l2','l1']}

log_reg_cat1 = GridSearchCV(LogisticRegression(), tuned_parameters, scoring='roc_auc', cv=3, refit=True)
    
log_reg_cat1.fit(df_lsa_train,y_train_cat['cat1'])

y_score=log_reg_cat1.predict_proba(df_lsa_test)[:,1]
fpr_l1, tpr_l1,_ = roc_curve(y_test_cat['cat1'], y_score)
roc_l1 = auc(fpr_l1, tpr_l1)

y_pred_train_l1=log_reg_cat1.predict(df_lsa_train)
    
acc_train_l1=accuracy_score(y_train_cat['cat1'], y_pred_train_l1)
    
y_pred_test_l1=log_reg_cat1.predict(df_lsa_test)
    
acc_test_l1=metrics.accuracy_score(y_test_cat['cat1'], y_pred_test_l1)

#cat2

log_reg_cat2 = GridSearchCV(LogisticRegression(), tuned_parameters, scoring='roc_auc', cv=3, refit=True)
    
log_reg_cat2.fit(df_lsa_train,y_train_cat['cat2'])

y_score=log_reg_cat2.predict_proba(df_lsa_test)[:,1]
fpr_l2, tpr_l2,_ = roc_curve(y_test_cat['cat2'], y_score)
roc_l2 = auc(fpr_l2, tpr_l2)

y_pred_train_l2=log_reg_cat2.predict(df_lsa_train)
    
acc_train_l2=accuracy_score(y_train_cat['cat2'], y_pred_train_l2)
    
y_pred_test_l2=log_reg_cat2.predict(df_lsa_test)
    
acc_test_l2=metrics.accuracy_score(y_test_cat['cat2'], y_pred_test_l2)

#cat3

log_reg_cat3 = GridSearchCV(LogisticRegression(), tuned_parameters, scoring='roc_auc', cv=3, refit=True)
    
log_reg_cat3.fit(df_lsa_train,y_train_cat['cat3'])

y_score=log_reg_cat3.predict_proba(df_lsa_test)[:,1]
fpr_l3, tpr_l3,_ = roc_curve(y_test_cat['cat1'], y_score)
roc_l3 = auc(fpr_l3, tpr_l3)

y_pred_train_l3=log_reg_cat3.predict(df_lsa_train)
    
acc_train_l3=accuracy_score(y_train_cat['cat3'], y_pred_train_l3)
    
y_pred_test_l3=log_reg_cat3.predict(df_lsa_test)
    
acc_test_l3=metrics.accuracy_score(y_test_cat['cat3'], y_pred_test_l3)

#GradiantBoosting
#cat1
learning_rates = np.logspace(-4, -0.5, 30)
max_features= range(1,5)
tuned_parameters = [{'learning_rate': learning_rates, 'max_features':max_features,'n_estimators':[300] }]
n_folds = 3

GBC_1=GridSearchCV(GradientBoostingClassifier(), tuned_parameters, scoring='roc_auc', cv=n_folds, refit=True)

GBC_1.fit(df_lsa_train,y_train_cat['cat1'])

y_score=GBC_1.predict_proba(df_lsa_test)[:,1]
fpr_gbc1, tpr_gbc1,_ = roc_curve(y_test_cat['cat1'], y_score)
roc_gbc1 = auc(fpr_gbc1, tpr_gbc1)

y_pred_train_g1=GBC_1.predict(df_lsa_train)
    
acc_train_g1=accuracy_score(y_train_cat['cat1'], y_pred_train_g1)
    
y_pred_test_g1=GBC_1.predict(df_lsa_test)
    
acc_test_g1=metrics.accuracy_score(y_test_cat['cat1'], y_pred_test_g1)

#cat2
GBC_2=GridSearchCV(GradientBoostingClassifier(), tuned_parameters, scoring='roc_auc', cv=n_folds, refit=True)

GBC_2.fit(df_lsa_train,y_train_cat['cat2'])

y_score=GBC_2.predict_proba(df_lsa_test)[:,1]
fpr_gbc2, tpr_gbc2,_ = roc_curve(y_test_cat['cat2'], y_score)
roc_gbc2 = auc(fpr_gbc2, tpr_gbc2)

y_pred_train_g2=GBC_2.predict(df_lsa_train)
    
acc_train_g2=accuracy_score(y_train_cat['cat2'], y_pred_train_g2)
    
y_pred_test_g2=GBC_2.predict(df_lsa_test)
    
acc_test_g2=metrics.accuracy_score(y_test_cat['cat2'], y_pred_test_g2)

#cat3
GBC_3=GridSearchCV(GradientBoostingClassifier(), tuned_parameters, scoring='roc_auc', cv=n_folds, refit=True)

GBC_3.fit(df_lsa_train,y_train_cat['cat3'])

y_score=GBC_3.predict_proba(df_lsa_test)[:,1]
fpr_gbc3, tpr_gbc3,_ = roc_curve(y_test_cat['cat3'], y_score)
roc_gbc3 = auc(fpr_gbc3, tpr_gbc3)

y_pred_train_g3=GBC_3.predict(df_lsa_train)
    
acc_train_g3=accuracy_score(y_train_cat['cat3'], y_pred_train_g3)
    
y_pred_test_g3=GBC_3.predict(df_lsa_test)
    
acc_test_g3=metrics.accuracy_score(y_test_cat['cat3'], y_pred_test_g3)


#plt.figure(figsize= [10,10])

# Plotting our Baseline..
# Two subplots, the axes array is 1-d

fig, axes = plt.subplots(3, sharex=True, figsize=(10,13))
axes[0].set_title("ROC curves for evaluated models ", size = 25)
axes[0].set_title('Category 1 Price < 20')
axes[0].plot(fpr_l1,tpr_l1,label='Logistic Regression')
axes[0].plot(fpr_gbc1,tpr_gbc1,label='GradiantBoosting')
axes[0].plot(fpr_gbc1,fpr_gbc1)
#axes[0].ylabel('TPR', size = 10,rotation = 0,labelpad = 35)

axes[1].set_title('Category 2 Price betwwen 20 and 100')
axes[1].plot(fpr_l2,tpr_l2,label='Logistic Regression')
axes[1].plot(fpr_gbc2,tpr_gbc2,label='GradiantBoosting')
axes[1].plot(fpr_gbc2,fpr_gbc2)

axes[2].set_title('Category 3 Price > 100')
axes[2].plot(fpr_l3,tpr_l3,label='Logistic Regression')
axes[2].plot(fpr_gbc3,tpr_gbc3,label='GradiantBoosting')
axes[2].plot(fpr_gbc3,fpr_gbc3)
#axes[2].xlabel('FPR', size = 10)
#axes[2].ylabel('TPR', size = 15,rotation = 0,labelpad = 35)
plt.legend(loc='best',prop={'size': 12})

# plt.xlabel('Coefficients', size = 15, labelpad = 15)
# plt.ylabel('Variables        ', size = 20, rotation = 0, labelpad = 35)
plt.xticks(size=14)
plt.yticks(size=14)
ax = plt.gca()
ax.yaxis.set_label_coords(-0.06,0.97)
ttl = ax.title
ttl.set_position([.5, 1.05]);

resuls_cat2=[]
resuls_cat=[n_features,n_lsa_comp,
             acc_train_l1,acc_train_l2,acc_train_l3,
             acc_test_l1,acc_test_l2,acc_test_l3,
             roc_l1,roc_l2,roc_l3,
             roc_gbc1,roc_gbc2,roc_gbc3]

resuls_cat2.append(resuls_cat)

resuls_catl=np.array(resuls_cat2)

df_resuls_cat=pd.DataFrame({'n_features':resuls_catl.T[0],
                        'n_lsa_comp':resuls_catl.T[1],
                        'acc_train_l1':resuls_catl.T[2],
                        'acc_train_l2':resuls_catl.T[3],
                        'acc_train_l3':resuls_catl.T[4],
                        'acc_test_l1':resuls_catl.T[5],
                        'acc_test_l2':resuls_catl.T[6],
                        'acc_test_l3':resuls_catl.T[7],
                        'roc_l1':resuls_catl.T[8],
                        'roc_l2':resuls_catl.T[9],
                        'roc_l3':resuls_catl.T[10],
                        'roc_gbc1':resuls_catl.T[11],
                        'roc_gbc2':resuls_catl.T[12],
                        'roc_gbc3':resuls_catl.T[13]})
    
df_resuls_cat.T

