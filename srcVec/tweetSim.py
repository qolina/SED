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

sys.path.append(os.path.expanduser("~") + "/Scripts/")
from fileOperation import loadStopword

sys.path.append("../src/")
from util import snpLoader
from util import fileReader

def raw2Texts(rawTweetFilename):
    # read raw tweet
    rawTweetHash = fileReader.loadTweets(rawTweetFilename) # tid:text
    rawTweets = rawTweetHash.values()
    print "## End of reading file. [raw tweet file][cleaned text]  #tweets: ", len(rawTweetHash), rawTweetFilename

    # remove stopwords and tokenize
    stopWords = loadStopword("../data/stoplist.dft")
    texts = [[word for word in document.lower().split() if word not in stopWords] for document in rawTweets]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > 1] for text in texts]
    #pprint(texts[:10])
    return texts

def prepCorpus(rawTweetFilename, dictPath, corpusPath):
    texts = raw2Texts(rawTweetFilename)

    dictionary = corpora.Dictionary(texts)
    dictionary.save(dictPath)

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(corpusPath, corpus)
    #print(corpus[:5])

def getNearestNeighbor(index, simThreshold):
    neighbors = []
    for docSims in index:
        sortedSims = sorted(enumerate(docSims), key=lambda item: -item[1])
        selectedSims = [item for item in sortedSims if item[1] > simThreshold]
        neighbors.append(selectedSims)
        print len(selectedSims), selectedSims[:10]
        break
    return neighbors

def outputNeighbor(neighbors):
    return 0

# lsi example in gensim
def lsi_vec_sim(dictPath, corpusPath):
    lsiModelPath = "../ni_data/tweetVec/model.lsi"
    indexPath = "../ni_data/tweetVec/tweets.index"

    ###############################
    # load dictionary and corpus
    #if os.path.exists(dictPath):
    #    dictionary = corpora.Dictionary.load(dictPath)
    #    corpus = corpora.MmCorpus(corpusPath)
    #    print "## corpus loaded."
    #else:
    #    print "## Error: No dictionary and corpus found in ", dictPath
    #    return -1

    # corpus to model
    #tfidf = models.tfidfmodel(corpus)
    #corpus_tfidf = tfidf[corpus]
    #lsi = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics = 300)
    #print "## corpus transformation end. [LSI] at ", time.asctime()
    #lsi.save(lsiModelPath)
    
    ###############################
    # calculate similarity
    #lsi = models.LsiModel.load(lsiModelPath)
    #corpus_lsi = lsi[corpus_tfidf]
    #print corpus_tfidf[0]
    #print corpus_lsi[0]
    #index = similarities.MatrixSimilarity(corpus_lsi)
    #print "## corpus similarity index end. at ", time.asctime()
    #index.save(indexPath)

    ###############################
    index = similarities.MatrixSimilarity.load(indexPath)
    simThreshold = 0.9
    neighbors = getNearestNeighbor(index, simThreshold)

    # example of a doc to whole corpus
    #doc = "This a test of apple watch release"
    #vec_bow = dictionary.doc2bow(doc.lower().split())
    #vec_lsi = lsi[vec_bow]
    #sims = index[vec_lsi]
    #sims = sorted(enumerate(sims), key=lambda item: -item[1])
    #print sims[:10]

def getTweetSimsLSI(rawTweetFilename):
    dictPath = "../ni_data/tweetVec/tweets.dict"
    corpusPath = "../ni_data/tweetVec/tweets.mm"

    ###############################
    # prepare corpus from raw
    #prepCorpus(rawTweetFilename, dictPath, corpusPath)

    ###############################
    # represent corpus with lsi vec, calculate similarity
    lsi_vec_sim(dictPath, corpusPath)


def getTweetSims(rawTweetFilename):
    #texts = raw2Texts(rawTweetFilename)
    rawTweets = fileReader.loadTweets(rawTweetFilename).values()
    texts = [[word for word in document.lower().split()] for document in rawTweets]
    doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model"

    ###############################
    # build doc2vec model and save it
    # do build_vocab and train automatically if initialize documents in building Doc2Vec model

    #taggedTexts = [models.doc2vec.TaggedDocument(words=text, tags=["SENT_%s" %tid]) for tid, text in enumerate(texts)]
    #doc2vecModel = models.Doc2Vec(taggedTexts, size=100, window=5, workers=1, iter=10)
    #doc2vecModel.save(doc2vecModelPath)
    ##print doc2vecModel.docvecs[0]

    doc2vecModel = models.doc2vec.Doc2Vec.load(doc2vecModelPath)

    ###############################
    # how many tweets are semantically similar to given tweet
    simThreshold = 0.95
    simFreqCount = []
    simNN = []
    for docid, docvec in enumerate(doc2vecModel.docvecs):
        nn = doc2vecModel.docvecs.most_similar(docid, topn=False)
        #nn = [item for item in nn if item[1] > simThreshold]
        nn = [item for item in enumerate(nn) if item[1] > simThreshold]
        simFreqCount.append(len(nn))
        simNN.append(nn)

    # sort tweet by simFreq
    sortedTweet_byCount = sorted(enumerate(simFreqCount), key = lambda item: -item[1])
    freqCounts = [item[1] for item in sortedTweet_byCount]
    print Counter(freqCounts).most_common()
    for docid, freqCount in sortedTweet_byCount[:]:
        if "apple" not in texts[docid]:
            continue
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
