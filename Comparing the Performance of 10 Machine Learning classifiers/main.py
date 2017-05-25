#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:40:02 2016

@author: sandeepkatypally

"""

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import timeit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from table_reader import nb
import Orange as orange
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
#
def a12(list_1, list_2):
    more = same = 0.0
    for x in sorted(list_1):
        for y in sorted(list_2):
            if x == y:
                same += 1
            elif x > y:
                more += 1
    return (more + 0.5*same) / (len(list_1)*len(list_2))


def plotting(fileIndex, characteristics,fever,(plotNumber)):
    #plotting

    for model in characteristics:
        if model=="NB":
            pass
        else:
            plotNumber[fileIndex].plot(characteristics[model],0,'bo')
        plotNumber[fileIndex].plot(characteristics['NB'],0,'ro',marker='o',picker=3)
        
    plotNumber[0].set_title(' Comparison of NB over 5*5 Cross Validation')
    fever.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fever.axes[:-1]], visible=False)
    plotNumber[0].set_yticklabels([])
    plotNumber[5].set_ylabel('Datasets')
    plotNumber[-1].set_xlabel('Accuracy')    
    
allFiles=[['ant-1.3.csv','ant-1.4.csv','ant-1.5.csv','ant-1.6.csv','ant-1.7.csv'],['camel-1.0.csv','camel-1.2.csv','camel-1.4.csv','camel-1.6.csv'],['ivy-1.1.csv','ivy-1.4.csv','ivy-2.0.csv'],['jedit-3.2.csv','jedit-4.0.csv','jedit-4.1.csv','jedit-4.2.csv','jedit-4.3.csv'],['log4j-1.0.csv','log4j-1.1.csv'],['lucene-2.0.csv','lucene-2.2.csv','lucene-2.4.csv'],['velocity-1.4.csv','velocity-1.5.csv','velocity-1.6.csv'],['synapse-1.0.csv','synapse-1.1.csv','synapse-1.2.csv'],['xalan-2.4.csv','xalan-2.5.csv','xalan-2.6.csv','xalan-2.7.csv'],['xerces-1.2.csv','xerces-1.3.csv','xerces-1.4.csv','xerces-init.csv']]
plotNumber=()
fever,(plotNumber) = plt.subplots(10, sharex=True, sharey=True)
val=0.
performance={}
for fileIndex,fileset in enumerate(allFiles):
    try:
        performance=pickle.load( open( "withSmote/%s.p"%(fileset[0]), "rb" ) )
        print("pickle file found")
        #plotting
        plotting(fileIndex,performance[str(fileIndex)]['accuracy'],fever,(plotNumber))      
    except:
        fileset=['ant-1.3.csv','ant-1.4.csv','ant-1.5.csv','ant-1.6.csv','ant-1.7.csv']
        print("pickle file NOT found")
        count=0
        for f in fileset:
            fp=open('Input Data/'+f)
            header=fp.readline()
            if count==0:
                count=+1
                dataset=np.loadtxt(fp,delimiter=",",dtype=str)
                dataset=np.array(filter(lambda x:(x[-1]=='1' or x[-1]=='0'),dataset),dtype=str)
            else:
                subdataset=np.loadtxt(fp,delimiter=",",dtype=str)
                subdataset=np.array(filter(lambda x:(x[-1]=='1' or x[-1]=='0'),subdataset),dtype=str)
                dataset=np.append(dataset,subdataset,0)
        
        #features
        X=dataset[:,3:-1]
        X=X.astype(float)
        count1,count2=0,0
        Xtable=orange.data.discretization.DiscretizeTable(X)
        X_discretized=np.array(np.zeros_like(X),dtype='str')
        for i in range(np.shape(X)[0]):
            for j in range(np.shape(X)[1]):
                    X_discretized[i][j]=str(Xtable[i][j])
        
        print(np.shape(X))
        print(np.shape(X_discretized))
        encoder=LabelEncoder()
        for i in range(np.shape(X_discretized)[1]):
            X_discretized[:,i]=encoder.fit_transform(X_discretized[:,i])
        X
        #labels
        Y=dataset[:,-1]
        Y=Y.astype(bool)
        posNegRatio=round(float((np.count_nonzero(Y)))/(np.size(Y)-np.count_nonzero(Y)),2)
        print("positive/negative = %s"%(round(float((np.count_nonzero(Y)))/(np.size(Y)-np.count_nonzero(Y)),2)))
        sm = SMOTE(kind='regular')
        
        #classfiers
        models=[]
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        models.append(('Perceptron',Perceptron()))
        models.append(('QDA',QuadraticDiscriminantAnalysis()))
        models.append(('Random Forests',RandomForestClassifier()))
        models.append(('Multi Layer Perceptron',MLPClassifier()))
        
        
        #dictionaries to hold performance characteristics for the crossval of all the models
        precisionAll={}
        recallAll={}
        fscoreAll={}
        accuracyAll={}
        timeAll={}
        #dictionary to hold all the performance characteristics of all models by file
        performance[str(fileIndex)]={}
        
        #class to do cross val
        kf= StratifiedKFold(n_splits=5)
        for name,model in models:
            precisionAll[name],recallAll[name],fscoreAll[name],accuracyAll[name]=[],[],[],[]
            precision,recall,fscore,accuracy=[],[],[],[]
            start = timeit.default_timer()
            for train, test in kf.split(X,Y):
                #Xsampled, Ysampled = sm.fit_sample((X_discretized[train].astype(int)), (Y[train].astype(int)))
                clf=model
                clf=clf.fit(X_discretized[train].astype(float),Y[train].astype(float))
                ypredict=clf.predict(X[test])
                precision_,recall_,fscore_,_=precision_recall_fscore_support(Y[test],ypredict,average='binary')
                accuracy.append(accuracy_score(Y[test],ypredict))
                precision.append(precision_)
                recall.append(recall_)
                fscore.append(fscore_)
#        for train, test in kf.split(X,Y):
#            Xsampled, Ysampled = sm.fit_sample(X[train], Y[train])
#            print(nb('ant-1.3.csv','ant-1.3.csv'))
            stop = timeit.default_timer()
            precisionAll[name],recallAll[name],fscoreAll[name],accuracyAll[name],timeAll[name]=round(np.mean(precision)*100,3),round(np.mean(recall)*100,3),round(np.mean(fscore)*100,3),round(np.mean(accuracy)*100,3),round((stop-start)*1000,3)
        print(fileset[0])
        for model in accuracyAll:
            print('%s %.1f'%(model,accuracyAll[model]))#,accuracyAll[model][1]))            
        #print('Precision of classifiers is %s'%precisionAll)
        #print('Recall of classifiers is %s'%recallAll)
        #print('F Beta score of classifiers is %s'%fscoreAll)
        #print('accuracy of classifiers is %s'%accuracyAll)
        performance[str(fileIndex)]['precision'],performance[str(fileIndex)]['recall'],performance[str(fileIndex)]['fscore'],performance[str(fileIndex)]['accuracy'],performance[str(fileIndex)]['time']=precisionAll,recallAll,fscoreAll,accuracyAll,timeAll
        pickle.dump(performance,open('%s.p'%(fileset[0]),'wb'))
        
        #performanceCharacteristics=['accuracy','recall','precision','fscore','time']
        #plotting
        plotting(fileIndex,performance[str(fileIndex)]['accuracy'],fever,(plotNumber))
plt.show()
