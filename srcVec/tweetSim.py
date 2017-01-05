#!/usr/bin/python
# -*- coding: UTF-8 -*-

## function
## convert tweet into dense vector using gensim, a doc2vec tool

import sys
import os
import re
import time
from pprint import pprint
from collections import defaultdict
from collections import Counter

import numpy as np
from sklearn import cluster
import falconn

from gensim import corpora, models, similarities

import tweetVec
sys.path.append(os.path.expanduser("~") + "/Scripts/")
from fileOperation import loadStopword

sys.path.append("../src/")
from util import snpLoader
from util import fileReader


def getTweetSims(rawTweetFilename):
    rawTweetHash = fileReader.loadTweets(rawTweetFilename) # tid:text
    rawTweets = rawTweetHash.values()
    texts = raw2Texts(rawTweets, False, False, None)
    doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model"
    #tweetVec.usingDoc2vec(texts, doc2vecModelPath)
    doc2vecModel = models.doc2vec.Doc2Vec.load(doc2vecModelPath)

    ###############################
    # how many tweets are semantically similar to given tweet
    simThreshold = 0.95
    simFreqCount = []
    simNN = []
    for docid, docvec in enumerate(doc2vecModel.docvecs):
        nn = doc2vecModel.docvecs.most_similar(docid, topn=10000)
        nn = [item for item in nn if item[1] > simThreshold]
        #nn = [item for item in enumerate(nn) if item[1] > simThreshold]
        simFreqCount.append(len(nn))
        simNN.append(nn)

    # sort tweet by simFreq
    sortedTweet_byCount = sorted(enumerate(simFreqCount), key = lambda item: -item[1])
    freqCounts = [item[1] for item in sortedTweet_byCount]
    print Counter(freqCounts).most_common()
    for docid, freqCount in sortedTweet_byCount[:]:
        if "apple" not in texts[docid]:
            continue
        if freqCount == 0:
            break
        print "############################"
        print "**", docid, freqCount
        print "**", texts[docid]
        for item in simNN[docid]:
            print item[0], item[1], " ".join(texts[int(item[0][5:])])
    return 

    ###############################
    # clustering tweet vecs

    # kmeans
    kmeans = cluster.KMeans(n_clusters=100).fit(doc2vecModel.docvecs)
    #print kmeans.labels_
    labelCounter = Counter(kmeans.labels_)
    #print labelCounter.most_common(10)
    #print kmeans.cluster_centers_
    docDist = kmeans.transform(doc2vecModel.docvecs)
    for label, num in labelCounter.most_common(5):
        dataIn = [item[0] for item in enumerate(kmeans.labels_) if item[1] == label]
        dists = list(docDist[:,label])
        distsIn = [(docid, dists[docid]) for docid in dataIn]
        sortedData = sorted(distsIn, key = lambda item: item[1])
        print "############################"
        print "**", label, num
        for docid, dist in sortedData[:20]:
            print docid, dist, "\t", " ".join(texts[docid])


    # lsh clustering with falconn
    #dataset = np.array(doc2vecModel.docvecs)


##############
def getArg(args, flag):
    arg = None
    if flag in args:
        arg = args[args.index(flag)+1]
    return arg

def parseArgs(args):
    arg1 = getArg(args, "-in")
    if arg1 is None:
        print "Usage: python tweetSim.py -in inputTweetFilename"
        print "eg: ../ni_data/word/tweetCleanText01"
        sys.exit(0)
    return arg1
####################################################
if __name__ == "__main__":
    rawTweetFilename = parseArgs(sys.argv)
    print "Program starts at ", time.asctime()

    simsTweet = getTweetSims(rawTweetFilename)

    print "Program ends at ", time.asctime()
