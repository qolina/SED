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
from sklearn import cluster
from collections import Counter
import falconn

sys.path.append("../srcOie/")
from pivotRanking import JaccardSim

sys.path.append(os.path.expanduser("~") + "/Scripts/")
from hashOperation import statisticHash

def getSim(dataset, thred_radius_dist):
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
    #nnModel = NearestNeighbors(radius=thred_radius_dist)
    nnModel = LSHForest(radius=thred_radius_dist)
    nnModel.fit(dataset)
    ngDistArray, ngIdxArray = nnModel.radius_neighbors(dataset)
    print ngDistArray.shape, ngIdxArray.shape
    print "## Nearest neighbor with radius ", thred_radius_dist, " obtained at", time.asctime()
    return ngDistArray, ngIdxArray

def getSim_falconn(dataset, thred_radius_dist):
    dataset = dataset.astype(np.float32)
    dataset -= np.mean(dataset, axis=0)
    num_points, dim = dataset.shape
    para = falconn.get_default_parameters(num_points, dim)
    nnModel = falconn.LSHIndex(para)
    nnModel.setup(dataset)
    print "## Sim falconn data setup", time.asctime()
    rate_probes = 2
    ngIdxList = []
    for dataIdx in range(num_points):
        #nnModel.set_num_probes(nnModel.get_num_probes() * rate_probes)
        nn_keys = nnModel.find_near_neighbors(dataset[dataIdx,:], thred_radius_dist)
        ngIdxList.append(nn_keys)
    ngIdxList = np.asarray(ngIdxList)
    print "## Nearest neighbor [Falconn_lsh] with radius ", thred_radius_dist, ngIdxList.shape, " obtained at", time.asctime()
    return ngIdxList

#######################
# statistic tweetSimDf by day
# tweetSimDfDayArr: [nnDay_Counter_seqid0, seq1, ...]
# nnDay_Counter_seqid: (day, tweet_nn_num)
def getDF(ngIdxArray, seqDayHash, timeWindow):
    tweetSimDfDayArr = []
    for docid, nnIdxs in enumerate(ngIdxArray):
        nnDays = [seqDayHash.get(seqid) for seqid in nnIdxs]
        if timeWindow is not None:
            date = int(seqDayHash.get(docid))
            if (date <= timeWindow[1]) or (date >= 31+timeWindow[0]):
                nnDay_count = None
            else:
                date_inTimeWin = range(date+timeWindow[0], date+timeWindow[1]+1)
                nnDays = [item for item in nnDays if item in date_inTimeWin]
                nnDay_count = Counter(nnDays)
        #if docid in range(50, 100):
        #    print nnDay_count.items()
        tweetSimDfDayArr.append(nnDay_count)
    print "## Tweets simDF by day obtained at", time.asctime()
    return tweetSimDfDayArr

# zscoreArr = [zscore_seqid0, seq1, ...]
def getBursty2(simDfDayArr, seqDayHash):
    zscoreArr = []
    for docid, nnDayCounter in enumerate(simDfDayArr):
        if nnDayCounter is None:
            zscoreArr.append(None)
            continue
        dfs = np.asarray(nnDayCounter.values(), dtype=np.float32)
        df_currentDay = nnDayCounter[seqDayHash.get(docid)]
        mu = np.mean(dfs)
        sigma = np.std(dfs)
        zscore = round((df_currentDay-mu)/sigma, 4)
        zscoreArr.append((seqDayHash.get(docid), zscore))

# zscoreDayArr: [zscoreDay_seqid0, seq1, ...]
# zscoreDay_seqid: [(day, zscore), (day, zscore)]
def getBursty(simDfDayArr, dayTweetNumHash):
    TweetNum_all = sum(dayTweetNumHash.values())
    zscoreDayArr = []
    for docid, nnDayCounter in enumerate(simDfDayArr):
        docSimDF_all = sum(nnDayCounter.values())
        est_prob = docSimDF_all*1.0/TweetNum_all
        zscoreDay = []
        zscoreTest = []
        for day, simDf in nnDayCounter.items():
            if simDf < 1:
                continue
            TweetNum_day = dayTweetNumHash[day]
            mu = est_prob * TweetNum_day
            # sigma2 = Nt*p*(1-p) = mu*(1-p)
            sigma = math.sqrt(mu*(1-est_prob))
            #print docid, day, simDf, mu, est_prob, sigma
            zscore = round((simDf*1.0-mu)/sigma, 4)
            zscoreDay.append((day, zscore))
            #if zscore > 5.0:
            #    zscoreTest.append(math.floor(zscore))
        #if docid in range(50, 70) or docid in range(150, 170):
        #if len(zscoreTest) > 0:
        if docid in [82, 67, 656, 893, 249, 22359, 22283, 22317, 22297, 22275, 33536, 33547, 33554, 33543, 33679]:
            print "#################################"
            print nnDayCounter.most_common()
            print sorted(zscoreDay, key = lambda a:a[1], reverse=True)
        zscoreDayArr.append(zscoreDay)
    print "## Tweets zscore by day obtained at", time.asctime()
    return zscoreDayArr


# choose docs appear in specific time window (day)
def filtering_by_zscore(zscoreDayArr, seqDayHash, day, thred_zscore):
    burstySeqIdArr = []
    zscores = []
    for docid, zscoreDay in enumerate(zscoreDayArr):
        if zscoreDay is None:
            continue
        if seqDayHash[docid] != day:
            continue
        #if docid < 20:
        #    print zscoreDay
        zscore = dict(zscoreDay).get(day)
        if zscore is None:
            continue
        #zscores.append(round(zscore, 1))
        zscores.append(math.floor(zscore))
        if zscore > thred_zscore:
            burstySeqIdArr.append(docid)
    print "## Tweets filtering by zscore ", thred_zscore, " obtained at", time.asctime()
    print "## statistic of zscore", Counter(zscores).most_common()
    return burstySeqIdArr

def clustering(documents, feaVecs, topK_c, topK_t, burstySeqIdArr):
    # kmeans
    kmeans = cluster.KMeans(n_clusters=100).fit(feaVecs)
    clusterScore = []
    labelCounter = Counter(kmeans.labels_)
    docDist = kmeans.transform(feaVecs)
    for label in labelCounter.keys():
        dataIn = [item[0] for item in enumerate(kmeans.labels_) if item[1] == label]
        dataOut = [docid for docid in range(len(documents)) if docid not in dataIn]
        distsIn = docDist[[dataIn],label]
        distsOut = docDist[[dataOut],label]

        dotNum = labelCounter[label]
        inSim = 1-np.mean(distsIn)
        outDist = np.mean(distsOut)
        score = dotNum * inSim * outDist
        clusterScore.append(score)

    #for label, num in labelCounter.most_common(10):
    for label, score in sorted(enumerate(clusterScore), key = lambda a:a[1], reverse=True)[:topK_c]:
        dataIn = [item[0] for item in enumerate(kmeans.labels_) if item[1] == label]
        dists = list(docDist[:,label])
        distsIn = [(docid, dists[docid]) for docid in dataIn]
        sortedData = sorted(distsIn, key = lambda item: item[1])
        print "############################"
        print "** cluster", label, score, ", #tweet", labelCounter[label]
        for docid, dist in sortedData[:topK_t]:
            print burstySeqIdArr[docid], dist, "\t", documents[docid]


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
