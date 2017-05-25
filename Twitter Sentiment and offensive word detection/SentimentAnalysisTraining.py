
# coding: utf-8

# In[7]:

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import cPickle as pickle


# In[8]:

df=pd.read_csv('TwitterTrainingData.csv','rb',delimiter=',',error_bad_lines=False)


# In[9]:

tweets_train=df.ix[:90000,3].values
labels_train=df.ix[:90000,1].values
tweets_test=df.ix[90000:100000,3].values
labels_test=df.ix[90000:100000,1].values

# to remove the non ascii characters fromt the tweets
for k in xrange(len(tweets_train)):
    tweets_train[k]=''.join([i if ord(i) < 128 else '' for i in str(tweets_train[k])])

    # to remove the non ascii characters fromt the tweets
for k in xrange(len(tweets_test)):
    tweets_test[k]=''.join([i if ord(i) < 128 else '' for i in str(tweets_test[k])])


# In[10]:

#using the count vectorizer, tfidf transformer for preprocessing of the text, and naive bayes classifier for training
text_clf_NB = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer(use_idf=False)),('clf', MultinomialNB()),])
text_clf_NB.fit(tweets_train,labels_train)
#save the model as a pickle file
pickle.dump(text_clf_NB,open('NbClassifier.p','wb'))

predictions_NB=text_clf_NB.predict(tweets_test)

np.mean(predictions_NB==labels_test)


# In[11]:

#using the count vectorizer, tfidf transformer for preprocessing of the text, and Svm classifier for training
text_clf_SVM = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer(use_idf=False)),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
text_clf_SVM.fit(tweets_train, labels_train)
#save the model as pickle file
pickle.dump(text_clf_SVM,open('SvmClassifier.p','wb'))

predictions_SVM = text_clf_SVM.predict(tweets_test)

np.mean(predictions_SVM == labels_test) 


# In[12]:

#reporting the stats about the performance of each of the classifier
print(metrics.classification_report(labels_test, predictions_NB))

print(metrics.classification_report(labels_test, predictions_SVM))


# In[ ]:



