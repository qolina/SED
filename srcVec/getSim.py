#!/usr/bin/python
# -*- coding: UTF-8 -*-

## function
## given a numpy array like dataset
## calculate its similarity matrix

import os
import sys
import time
import math
import numpy as np
from gensim import models, similarities
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors, LSHForest
from collections import Counter

sys.path.append("../srcOie/")
from pivotRanking import JaccardSim

sys.path.append(os.path.expanduser("~") + "/Scripts/")
from hashOperation import statisticHash

def getSim(doc2vecModelPath, thred_radius_dist):
    doc2vecModel = models.doc2vec.Doc2Vec.load(doc2vecModelPath)
    dataset = np.array(doc2vecModel.docvecs)
    #dataset = dataset[:10000,]

    #######################
    # nearest neighbor method 1
    ## too slow for scipy calculate nearest neighbors
    #simMatrix = pairwise.cosine_similarity(dataset)
    #distMatrix = pairwise.cosine_distances(dataset)
    #nns_fromSim = [sorted(enumerate(distMatrix[i]), key = lambda a:a[1])[:20] for i in range(distMatrix.shape[0])]
    #print "## Similarity Matrix obtained at", time.asctime()

    #######################
    # nearest neighbor method 2
    thred_n_neighbors = 10
    # choosing best
    #nnModels = [NearestNeighbors(radius=thred_radius_dist),
    #            NearestNeighbors(n_neighbors=thred_n_neighbors, algorithm='auto'), 
    #            NearestNeighbors(n_neighbors=thred_n_neighbors, algorithm='ball_tree'),
    #            NearestNeighbors(n_neighbors=thred_n_neighbors, algorithm='kd_tree'),
    #            LSHForest(random_state=30, n_neighbors=thred_n_neighbors)
    #            ]
    #for modelIdx in range(len(nnModels)):
    #    if modelIdx in [0, 2, 3, 4]: # choose auto
    #        continue
    #    nnModel = nnModels[modelIdx]
    #    nnModel.fit(dataset)
    #    if modelIdx == 0:
    #        ngDistArray, ngIdxArray = nnModel.radius_neighbors()
    #    elif modelIdx == 4:
    #        ngDistArray, ngIdxArray = nnModel.kneighbors(dataset)
    #    else:
    #        ngDistArray, ngIdxArray = nnModel.kneighbors()
    #    print ngDistArray.shape, ngIdxArray.shape
    #    print "## Nearest neighbor with model ", modelIdx, " obtained at", time.asctime()
    #    eval_sklearn_nnmodel(nns_fromSim, ngIdxArray)

    # using
    nnModel = NearestNeighbors(radius=thred_radius_dist)
    nnModel.fit(dataset)
    ngDistArray, ngIdxArray = nnModel.radius_neighbors()
    print "## Nearest neighbor with radius ", thred_radius_dist, " obtained at", time.asctime()
    print ngDistArray.shape, ngIdxArray.shape, np.max(ngDistArray), np.min(ngDistArray)
    return ngDistArray, ngIdxArray

#######################
# statistic tweetSimDf by day
# tweetSimDfDayArr: [nnDay_Counter_seqid0, seq1, ...]
# nnDay_Counter_seqid: (day, tweet_nn_num)
def getDF(ngIdxArray, seqDayHash):
    tweetSimDfDayArr = []
    for docid, nnIdxs in enumerate(ngIdxArray):
        nnDays = [seqDayHash.get(seqid) for seqid in nnIdxs]
        nnDay_count = Counter(nnDays)
        if docid < 150:
            print nnDay_count.items()
        tweetSimDfDayArr.append(nnDay_count)
    print "## Tweets simDF by day obtained at", time.asctime()
    return tweetSimDfDayArr


# zscoreDayArr: [zscoreDay_seqid0, seq1, ...]
# zscoreDay_seqid: [(day, zscore), (day, zscore)]
def getBursty(simDfDayArr, dayTweetNumHash):
    TweetNum_all = sum(dayTweetNumHash.values())
    zscoreDayArr = []
    for docid, nnDayCounter in enumerate(simDfDayArr):
        if docid < 150:
            print nnDayCounter
        docSimDF_all = sum(nnDayCounter.values())
        est_prob = docSimDF_all*1.0/TweetNum_all
        zscoreDay = []
        for day, simDf in nnDayCounter.items():
            if simDf < 1:
                continue
            TweetNum_day = dayTweetNumHash[day]
            mu = est_prob * TweetNum_day
            # sigma2 = Nt*p*(1-p) = mu*(1-p)
            sigma = math.sqrt(mu*(1-est_prob))
            zscore = (simDf*1.0-mu)/sigma
            zscoreDay.append((day, zscore))
        zscoreDayArr.append(zscoreDay)
    print "## Tweets zscore by day obtained at", time.asctime()
    return zscoreDayArr


# choose docs appear in specific time window (day)
def filtering_by_zscore(zscoreDayArr, seqDayHash, day, thred_zscore):
    burstySeqIdArr = []
    zscores = []
    for docid, zscoreDay in enumerate(zscoreDayArr):
        if seqDayHash[docid] != day:
            continue
        if docid < 20:
            print zscoreDay
        zscore = dict(zscoreDay).get(day)
        if zscore is None:
            continue
        #zscores.append(round(zscore, 1))
        zscores.append(math.floor(zscore))
        if zscore > thred_zscore:
            burstySeqIdArr.append(docid)
    print "## Tweets filtering by zscore ", thred_zscore, " obtained at", time.asctime()
    print Counter(zscores).most_common(20)
    return burstySeqIdArr

###########################################################
## nns_fromSim is groudTruth nearest neighbors
# result:
#                    20_ng,1k_data  10_ng,10k_data
# nnModel, auto:        0.3465      0.3339
#          ball_tree:   0.3465      0.3339
#          kd_tree:     0.3465      0.3339
#          lshforest:   0.3819      0.3204
def eval_sklearn_nnmodel(nns_fromSim, rngIdxArray):
    # test nearest neighbor
    evalNNModel_scoreArr = []
    for docid in range(rngIdxArray.shape[0]):
        nn_onedoc_sim = [item[0] for item in nns_fromSim[docid]]
        nn_onedoc_nnmodel = list(rngIdxArray[docid])
        score = JaccardSim(nn_onedoc_sim, nn_onedoc_nnmodel)
        evalNNModel_scoreArr.append(score)

        #print common
        #print set(nn_onedoc_nnmodel)-common
        #print set(nn_onedoc_sim)-common
        #print union
        #print score
    print "jaccard sim", np.mean(np.asarray(evalNNModel_scoreArr))
