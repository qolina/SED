#!/usr/bin/python
# -*- coding: UTF-8 -*-

## function
## convert tweet in format of tweetid[\t]tweetText into denseVector
## by doc2vec, lsi, tfidf etc

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
from util import fileReader

from getSim import getSim, getDF, getBursty, filtering_by_zscore

####################################################
def raw2Texts(rawTweets, rmStop, rmMinFreq, minFreq):
    # remove stopwords and tokenize
    stopWords = {}
    if rmStop:
        stopWords = loadStopword("../data/stoplist.dft")
    texts = [[word for word in document.lower().split() if word not in stopWords] for document in rawTweets]

    if not rmMinFreq:
        return texts

    # remove words that appear only minFreq
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    texts = [[token for token in text if frequency[token] > minFreq] for text in texts]
    return texts

def prepCorpus(texts, dictPath, corpusPath):
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

# lsi example in gensim
def lsi_vec(dictPath, corpusPath, lsiModelPath):
    # load dictionary and corpus
    if os.path.exists(dictPath):
        dictionary = corpora.Dictionary.load(dictPath)
        corpus = corpora.MmCorpus(corpusPath)
        print "## corpus loaded."
    else:
        print "## Error: No dictionary and corpus found in ", dictPath
        return -1

    # corpus to model
    tfidf = models.tfidfmodel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics = 300)
    print "## corpus transformation end. [LSI] at ", time.asctime()
    lsi.save(lsiModelPath)

def lsi_vec_sim(dictPath, corpusPath, lsiModelPath, simIndexPath):
    dictionary = corpora.Dictionary.load(dictPath)
    corpus = corpora.MmCorpus(corpusPath)
    tfidf = models.tfidfmodel(corpus)
    corpus_tfidf = tfidf[corpus]

    # calculate similarity
    lsi = models.LsiModel.load(lsiModelPath)
    corpus_lsi = lsi[corpus_tfidf]
    print corpus_tfidf[0]
    print corpus_lsi[0]
    index = similarities.MatrixSimilarity(corpus_lsi)
    print "## corpus similarity index end. at ", time.asctime()
    index.save(simIndexPath)

    ###############################
    index = similarities.MatrixSimilarity.load(simIndexPath)
    simThreshold = 0.9
    neighbors = getNearestNeighbor(index, simThreshold)

    # example of a doc to whole corpus
    #doc = "This a test of apple watch release"
    #vec_bow = dictionary.doc2bow(doc.lower().split())
    #vec_lsi = lsi[vec_bow]
    #sims = index[vec_lsi]
    #sims = sorted(enumerate(sims), key=lambda item: -item[1])
    #print sims[:10]


def usingDoc2vec(texts, doc2vecModelPath):
    # build doc2vec model and save it
    # do build_vocab and train automatically if initialize documents in building Doc2Vec model
    taggedTexts = [models.doc2vec.TaggedDocument(words=text, tags=["SENT_%s" %tid]) for tid, text in enumerate(texts)]
    doc2vecModel = models.Doc2Vec(taggedTexts, size=100, window=5, workers=1, iter=10)
    doc2vecModel.save(doc2vecModelPath)


def texts2LSIvecs(texts, dictPath, corpusPath):
    texts = raw2Texts(rawTweetFilename, True, True, 1)
    # prepare corpus from raw
    prepCorpus(texts, dictPath, corpusPath)

    lsi_vec(dictPath, corpusPath)

def texts2vecs(texts, doc2vecModelPath):
    texts = raw2Texts(texts, False, False, None)
    print "## text leximization finished. ", time.asctime()
    usingDoc2vec(texts, doc2vecModelPath)
    print "## doc2vec model saved. ", time.asctime()


def loadTweetsFromDir(dataDirPath):
    dayTweetNumHash = {} # dayStr:#tweetNum
    tweetTexts_all = []
    seqTidHash = {} # seqId in all: tweet_id
    seqDayHash = {} # seqId in all: dayStr
    for fileItem in sorted(os.listdir(dataDirPath)):
        if not fileItem.startswith("tweetCleanText"):
            continue
        dayStr = fileItem[-2:]
        if dayStr == "06":
            break
        rawTweetHash = fileReader.loadTweets(dataDirPath + "/" + fileItem) # tid:text
        #print "## End of reading file. [raw tweet file][cleaned text]  #tweets: ", len(rawTweetHash), fileItem
        tids = rawTweetHash.keys()[:1000]
        texts = rawTweetHash.values()[:1000]

        for seqid, tid in enumerate(tids, start=len(tweetTexts_all)):
            seqTidHash[seqid] = tid
            seqDayHash[seqid] = dayStr
        tweetTexts_all.extend(texts)
        dayTweetNumHash[dayStr] = len(texts)

    return tweetTexts_all, seqTidHash, seqDayHash, dayTweetNumHash


##############
def getArg(args, flag):
    arg = None
    if flag in args:
        arg = args[args.index(flag)+1]
    return arg

# arguments received from arguments
def parseArgs(args):
    arg1 = getArg(args, "-in")
    if arg1 is None: # nessensary argument
        print "Usage: python tweetVec.py -in inputFileDir"
        sys.exit(0)
    return arg1


####################################################
if __name__ == "__main__":
    dataDirPath = parseArgs(sys.argv)
    print "Program starts at ", time.asctime()

    tweetTexts_all, seqTidHash, seqDayHash, dayTweetNumHash = loadTweetsFromDir(dataDirPath)

    print dayTweetNumHash
    dayArr = sorted(dayTweetNumHash.keys())
    print dayArr

    ##############
    # doc2vec
    doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model"
    #texts2vecs(tweetTexts_all, doc2vecModelPath)

    thred_radius_dist = 0.5
    ngDistArray, ngIdxArray = getSim(doc2vecModelPath, thred_radius_dist)
    simDfDayArr = getDF(ngIdxArray, seqDayHash)
    #zscoreDayArr = getBursty(simDfDayArr, dayTweetNumHash)

    #for day in dayArr:
    #    if day != "01":
    #        continue
    #    thred_zscore = 1.0
    #    burstySeqIdArr = filtering_by_zscore(zscoreDayArr, seqDayHash, day, thred_zscore)
    #    print len(burstySeqIdArr)

    ##############
    # used for lsi, tfidf
    dictPath = "../ni_data/tweetVec/tweets.dict"
    corpusPath = "../ni_data/tweetVec/tweets.mm"
    lsiModelPath = "../ni_data/tweetVec/model.lsi"
    simIndexPath = "../ni_data/tweetVec/tweets.index"
    #texts2LSIvecs(tweetTexts_all, dictPath, corpusPath)
    #lsi_vec_sim(dictPath, corpusPath, lsiModelPath, simIndexPath)

    print "Program ends at ", time.asctime()
