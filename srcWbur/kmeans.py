#! /usr/bin/env python
#coding=utf-8
from __future__ import division
import random
import math
import ig

def pearson(source,target):
    sum_source=sum([value for value in source.values()])
    sum_target=sum([value for value in target.values()])
    
    sum_source_sq=sum([pow(value,2) for value in source.values()])
    sum_target_sq=sum([pow(value,2) for value in target.values()])
    
    p_sum=sum([source[word]*target[word] for word in source if word in target])
    
    num=p_sum-(sum_source*sum_target/len(source))
    den=math.sqrt((sum_source_sq-pow(sum_source,2)/len(source))*(sum_target_sq-pow(sum_target,2)/len(source)))
    return den

def cosine(source,target):
    numerator=sum([source[word]*target[word] for word in source if word in target])
    sourceLen=math.sqrt(sum([value*value for value in source.values()]))
    targetLen=math.sqrt(sum([value*value for value in target.values()]))
    denominator=sourceLen*targetLen
    if denominator==0:
        return 0
    else:
        return numerator/denominator

def getCentre(documents):
    centre={}
    for document in documents:
        for word in document.words:
            if word not in centre:
                centre[word]=0
            centre[word]+=1
    return dict([(word,centre[word]/len(documents)) for word in centre])

def clustering(documents,k=5):
    # init
    centres=[]
    i=0
    while i<k:
        index=int(random.random()*len(documents))
        if index not in centres:
            centres.append(index)
            i+=1
    centres=[[documents[i].words,[]] for i in centres]
    
    for i in range(20):
        # clear the cluster list
        print 'No.%d' %(i+1)
        for centre in centres:
            centre[1]=[]
        
        for j,document in enumerate(documents):
            index,maxsimilar=-1,0
            for k,centre in enumerate(centres):
                similar=cosine(document.words,centre[0])
                if similar>maxsimilar:
                    index,maxsimilar=k,similar
            centres[index][1].append(document)
            
        for centre in centres:
            if len(centre[1])>0:
                print len(centre[1])
                newCentre=getCentre(centre[1])
                centre[0]=newCentre
        
    return centres

def clustering_ig(posTrains,negTrains):    
    ig.selectedFeatures(posTrains+negTrains,rate=0.5)
    
    clusters=clustering(posTrains)
    cdocuments=[]
    for cluster in clusters:
        for document in cluster[1]:
            cdocuments.append(document)
            
    for document in cdocuments+negTrains:
        document.rebuild()
    
    for cluster in clusters:
        cluster[0]=getCentre(cluster[1])
    
    return clusters
    
