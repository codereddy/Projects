# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:34:02 2015

@author: sandeep reddy katypally

"""

import numpy as np
import string
from Bio import SeqIO as seqio
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import datasets

dataset=[0,0,7,2,6,4,3,9,7,5,13,5,76,4,6]
positiveData=[]  #list to store positive nucleosome sequence data
linkerData=[]  # list to store the linker sequence data

# clears the overlaps in the DNA sequence data from the DNA sequencing experiment output
def overlapclearer(file_, positiveDic):
    l=list()
    m=list()
    n =list()
    f =open(file_,'r')
    for line in f:
        while True:
            if line !='\n':
                n[:]=[]
                m.append(line[0:-31])
                n=list(line[-31:-1])
            else:
                m=m+n
                l=''.join(m)
                if l!='':
                   positiveDic.append(l)
                m[:]=[]
                n[:]=[]            
                break
            break   
    #print(list_)

           



#moving average calculates the rolling average of the parameter list to smoothen the signal        
#values should be always greater than window to get correct result
def movingaverage(values,window ):
    weights =np.repeat(1.0, window)/window       #acts as a uniform filter, can also use other filters like gausian depending of the features we are trying to evaluate for
    smas = np.convolve(values, weights, 'valid')     #using the filter to take the moving average of the sequence.
    return smas # as a numpy array

#making features like mean, max, min, standard deviation from the window sequences
def makefeatures(propertyvalues):
    smallMatrix=[]
    #print ("values are", propertyvalues)
    #print ("average is", np.mean(propertyvalues))
    #print ("maximum value is",np.max(propertyvalues))
    #print ("median value is",np.median(propertyvalues))
    #print ("minimum value is",np.min(propertyvalues))
    #print ("standard deviation is",np.std(propertyvalues))
    if isMaxSignificant(np.max(propertyvalues),np.mean(propertyvalues),np.std(propertyvalues))==1 |isMinsignificant(np.min(propertyvalues),np.mean(propertyvalues),np.std(propertyvalues))==1 :     #we are only taking the sequences with significant maximum or min values. if min or max is not significant then '0' is alloted in the matrix   
        smallMatrix.append(np.mean(propertyvalues))
        smallMatrix.append(np.std(propertyvalues))
        smallMatrix.append(np.median(propertyvalues))
        if isMaxSignificant(np.max(propertyvalues),np.mean(propertyvalues),np.std(propertyvalues))==1:
            smallMatrix.append(np.max(propertyvalues))
        else:
            smallMatrix.append("0")
        if isMinsignificant(np.min(propertyvalues),np.mean(propertyvalues),np.std(propertyvalues))==1:
            smallMatrix.append(np.min(propertyvalues))
        else:
            smallMatrix.append("0")
    return smallMatrix

def isMaxSignificant(maxima, mean, stndev):
    if (maxima-2*stndev>mean):
        return 1
    else:
        return 0

def isMinsignificant(minima, mean, stndev):
    if(minima+2*stndev<mean):
        return 1
    else:
        return 0

#dnascanner takes in the window size, file in fasta format and the genome sequence number in the file and scan it with the certain parameter
def dnascanner(windowsize,dataset_list, paramfile):
   
    s=dataset_list    
    #print dataset_list
    #print len(s)
    paramdictionary = {}
    parammatrix=[]
    paramvalues=[]  #the whole set of values for the sequence 
    
    with open(paramfile) as f:
        for line in f:
           (key, val) = line.split()
           paramdictionary[key] = val
    #print paramdictionary
    for i,value in enumerate(s):
        for (X,Y) in zip(s[i][0::1], s[i][1::1]):
            _dinucleotide=[X,Y]
            dinucleotide=string.join(_dinucleotide,'')
            correspvalue=paramdictionary[dinucleotide]
            #corespvalue is the value of the particular parameter for that dinucleotide
            
            paramvalues.append(float(correspvalue))            
        parammatrix.append(paramvalues)
        del paramvalues
        paramvalues=[]
        
    #print parammatrix
    featureMatrix=[]
    for i in range(len(parammatrix)):
        plt.plot(movingaverage(parammatrix[i],windowsize))
        if makefeatures(movingaverage(parammatrix[i],windowsize))!=[]:
            featureMatrix.append(makefeatures(movingaverage(parammatrix[i],windowsize)))
    return featureMatrix
    
#executing the functions

overlapclearer('group21.txt', positiveData)
overlapclearer('seq_linker.txt', linkerData)
#dnascanner(dataset,4)
positiveSet=[]
linkerSet=[]    
positiveSet= dnascanner(10,positiveData,'randparam.txt')
linkerSet= dnascanner(5,linkerData,'randparam.txt')
#dnascanner(dataset,4)
trainingposSet=[]
traininglinkerSet=[]
trainingposSet=positiveSet[::2]
testingposSet=positiveSet[1::2]
traininglinkerSet=linkerSet[::2]
testinglinkerSet=linkerSet[1::2]
#print len(positiveSet)
#print len(linkerSet)
#print len(trainingposSet)
#print len(traininglinkerSet)
#print len(testingposSet)
#print len(testinglinkerSet)
totalTrainingSet=[]
totalTestingSet=[]
totalTrainingSet=trainingposSet+traininglinkerSet
totalTestingSet=testingposSet+testinglinkerSet
#print len(totalTrainingSet)
#print len(totalTestingSet)

trainingLabelList=[]

for i in range(len(trainingposSet)):
    trainingLabelList.append('1')
for i in range(len(traininglinkerSet)):
    trainingLabelList.append('0')
#print labelList
#print len(trainingLabelList)


testingLabelList=[]

for i in range(len(testingposSet)):
    testingLabelList.append('1')
for i in range(len(testinglinkerSet)):
    testingLabelList.append('0')

#initialize SVM classifier to train the positive and negative data
    
classifier=svm.SVC()
classifier.fit(totalTrainingSet,trainingLabelList)
#SVMs decision function depends on some subset of the training data, called the support vectors. Some properties of these support vectors can be found in members support_vectors_, support_ and n_support, so use these if we need to know the properties of support vectors in the future;
predictions=[]
predictions= classifier.predict(totalTestingSet)
predictions=list(predictions)
print predictions
print len(predictions)
print testingLabelList
print len(testingLabelList)
positiveCount=float(0)
negativeCount=float(0)
accuracy=float()
for i in range(len(predictions)):
    if testingLabelList[i]==predictions[i]:
        positiveCount=positiveCount+1
    else:
        negativeCount=negativeCount+1
print positiveCount
print negativeCount
accuracy= float(positiveCount/(negativeCount+positiveCount))*100
print accuracy
