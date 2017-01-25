#!/usr/bin/python
# -*- coding: UTF-8 -*-

## function
## given a numpy array like dataset
## calculate its similarity matrix

import os
import sys
import time
import timeit
import math
import numpy as np
from gensim import models, similarities
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors, LSHForest, KDTree
from sklearn import cluster
from collections import Counter
from scipy.spatial.distance import cosine, sqeuclidean
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
    #thred_n_neighbors = 10
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
    #nnModel = LSHForest(radius=thred_radius_dist)
    nnModel.fit(dataset)
    ngDistArray, ngIdxArray = nnModel.radius_neighbors(dataset)
    #nnModel = KDTree(dataset, leaf_size=1.5*dataset.shape[0], metric="euclidean")
    #ngIdxArray = nnModel.query_radius(dataset, thred_radius_dist)
    print ngDistArray.shape, ngIdxArray.shape
    print "## Nearest neighbor with radius ", thred_radius_dist, " obtained at", time.asctime()
    return ngDistArray, ngIdxArray

# nns_fromSim used for eval lsh performance when training
def getSim_falconn(dataset, thred_radius_dist, trainLSH, trained_num_probes, nns_fromSim):
    dataset = prepData_forLsh(dataset)
    num_setup_threads = 20
    para = getPara_forLsh(dataset)
    para.num_setup_threads = num_setup_threads
    #para.l = 10 # num of hash tables
    #para.k = 5 # num of hash funcs per table
    nnModel = getLshIndex(para, dataset)

    # mainly train num_probes
    if trainLSH:
        print "## default l, k, prob", para.l, para.k, nnModel.get_num_probes()

        #distMatrix = pairwise.cosine_distances(dataset)
        #nns_fromSim = [sorted(enumerate(distMatrix[i]), key = lambda a:a[1])[:10] for i in range(distMatrix.shape[0])]
        #print "## nn by cosine sim obtained.", time.asctime()

        prob_cands = range(para.l, 200, 100)
        best_num_probes, max_jc = para.l, 0.0
        for num_probes in prob_cands:
            t1 = timeit.default_timer()
            ngIdxArray, indexedInCluster, clusters = getLshNN(dataset, nnModel, thred_radius_dist, num_probes)
            t2 = timeit.default_timer()
            print "## probs", num_probes, "\t time", round(t2-t1, 2)
            jc = eval_sklearn_nnmodel(nns_fromSim, ngIdxArray)
            if jc > max_jc:
                max_jc = jc
                best_num_probes = num_probes
        return best_num_probes

    ngIdxArray, indexedInCluster, clusters = getLshNN_op1(dataset, nnModel, thred_radius_dist, trained_num_probes)
    print "## Nearest neighbor [Falconn_lsh] with radius ", thred_radius_dist, ngIdxArray.shape, " obtained at", time.asctime()
    return ngIdxArray, indexedInCluster, clusters

def prepData_forLsh(dataset):
    dataset = dataset.astype(np.float32)
    #dataset -= np.mean(dataset, axis=0)
    dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
    return dataset

def getPara_forLsh(dataset):
    num_points, dim = dataset.shape
    para = falconn.get_default_parameters(num_points, dim)
    return para

def getLshIndex(para, dataset):
    nnModel = falconn.LSHIndex(para)
    nnModel.setup(dataset)
    print "## sim falconn data setup", time.asctime()
    return nnModel

def getLshNN_op1(dataset, nnModel, thred_radius_dist, trained_num_probes):
    ngIdxList= []
    indexedInCluster = {}
    clusters = []
    for dataidx in range(dataset.shape[0]):
        if dataidx in indexedInCluster: 
            nn_keys = None
        else:
            indexedInCluster[dataidx] = len(clusters)
            clusters.append([dataidx])

            nnModel.set_num_probes(trained_num_probes)
            # nn_keys: (id1, id2, ...)
            nn_keys = nnModel.find_near_neighbors(dataset[dataidx,:], thred_radius_dist)

            for key in nn_keys:
                if cosine(dataset[dataidx,:], dataset[key, :]) < 0.1:
                    indexedInCluster[key] = indexedInCluster[dataidx]

        ngIdxList.append(nn_keys)
        if (dataidx+1) % 10000 == 0:
            print "## completed", dataidx+1, len(clusters), time.asctime()
    ngIdxList = np.asarray(ngIdxList)
    return ngIdxList, indexedInCluster, clusters

def getLshNN_op2(dataset, nnModel, thred_radius_dist, trained_num_probes):
    ngIdxList= []
    print dataset.shape
    for dataidx in range(dataset.shape[0]):
        query_vec = dataset[dataidx,:]
        nnModel.set_num_probes(trained_num_probes)
        cand = nnModel.get_unique_candidates(query_vec)
        cand = [idx for idx in cand if idx>dataidx-130000 and idx<dataidx+130000]

        #distMatrix = pairwise.cosine_distances(dataset[cand, :], query_vec)
        #query_vec = np.asarray([query_vec])
        #distMatrix = pairwise.pairwise_distances(dataset[cand, :], query_vec, metric='sqeuclidean', n_jobs=5)
        #distMatrix = pairwise.euclidean_distances(dataset[cand, :], query_vec, squared=True)
        #distMatrix = [sqeuclidean(dataset[idx,:], query_vec) for idx in cand]
        # nn_keys: (id1, id2, ...)
        #nn_keys = [idx for idx, dist in zip(cand, distMatrix) if dist <= thred_radius_dist]
        nn_keys = nnModel.find_near_neighbors(dataset[dataidx,:], thred_radius_dist)

        ngIdxList.append(nn_keys)
        if (dataidx+1) % 10000 == 0:
            print "## completed", dataidx+1, time.asctime()
    ngIdxList = np.asarray(ngIdxList)
    return ngIdxList

#######################
# statistic tweetSimDf by day
# tweetSimDfDayArr: [nnDay_Counter_seqid0, seq1, ...]
# nnDay_Counter_seqid: (day, tweet_nn_num)
def getDF(ngIdxArray, seqDayHash, timeWindow, dataset, tweetTexts_all):
    tweetSimDfDayArr = []
    for docid, nnIdxs in enumerate(ngIdxArray):
        nnDay_count = None
        if nnIdxs is None: tweetSimDfDayArr.append(nnDay_count)
        nnDays = [seqDayHash.get(seqid) for seqid in nnIdxs]
        if timeWindow is not None:
            date = int(seqDayHash.get(docid))
            if (date > 0-timeWindow[0]) and (date <= 31-timeWindow[1]):
                date_inTimeWin = [str(item).zfill(2) for item in range(date+timeWindow[0], date+timeWindow[1]+1)]
                nnDays = [item for item in nnDays if item in date_inTimeWin]
            else:
                nnDays = None
        if nnDays is not None:
            nnDay_count = Counter(nnDays)
        tweetSimDfDayArr.append(nnDay_count)
    print "## Tweets simDF by day obtained at", time.asctime()
    return tweetSimDfDayArr

# zscoreDayArr = [zscore_seqid0, seq1, ...]
def getBursty_tw1(simDfDayArr, seqDayHash):
    zscoreDayArr = []
    statisticDfsNegDiff = [Counter() for i in range(32)]
    statisticDf = [Counter() for i in range(32)]
    for docid, nnDayCounter in enumerate(simDfDayArr):
        if nnDayCounter is None or len(nnDayCounter) < 3:
            zscoreDayArr.append(None)
            continue
        date = seqDayHash.get(docid)
        dfs = np.asarray(nnDayCounter.values(), dtype=np.float32)
        df_currentDay = nnDayCounter[date]
        mu = np.mean(dfs)
        sigma = np.std(dfs)
        zscore = 0.0
        if df_currentDay != mu:
            zscore = round((df_currentDay-mu)/sigma, 4)

        dfs_sorted = sorted(nnDayCounter.items(), key = lambda a:a[0])
        dfs_diff = [(int(day)-int(date), df-df_currentDay) for day, df in dfs_sorted]
        for window, diff in dfs_diff:
            if diff < 0:
                statisticDfsNegDiff[int(date)][window] += 1
            statisticDf[int(date)][window] += 1

        if docid in range(80030, 80090) or docid == 46909: #docid in range(100000, 100100) or
            print "###############"
            print "## doc", docid, "\t date", date, "\t df_day", 
            print df_currentDay, "\t mu", mu, "\t sigma", sigma, "\t zscore", zscore
            print dfs_sorted
            print dfs_diff

        zscoreDayArr.append([(date, zscore)])
    print "## Tweets zscore in time window obtained at", time.asctime()
    for i in range(1, 32):
        dfDiffInWin = sorted(statisticDf[i].items(), key = lambda a:a[0])
        dfNegDiffInWin = sorted(statisticDfsNegDiff[i].items(), key = lambda a:a[0])
        if len(dfDiffInWin) < 1:
            continue
        ratio = [round(num*1.0/statisticDf[i][day], 4) for day, num in dfNegDiffInWin]
        print "************Date", i
        print dfDiffInWin
        print dfNegDiffInWin
        print ratio
    return zscoreDayArr

def getBursty_tw2(simDfDayArr, seqDayHash, dayTweetNumHash):
    zscoreDayArr = []
    for docid, nnDayCounter in enumerate(simDfDayArr):
        if nnDayCounter is None or len(nnDayCounter) < 3:
            zscoreDayArr.append(None)
            continue
        TweetNum_all_tw = sum([dayTweetNumHash[day] for day in dayTweetNumHash if day in nnDayCounter])
        docSimDF_all = sum(nnDayCounter.values())
        est_prob = docSimDF_all*1.0/TweetNum_all_tw
        zscoreDay = []
        date = seqDayHash.get(docid)
        df_currentDay = nnDayCounter[date]
        TweetNum_day = dayTweetNumHash[date]
        if df_currentDay < 1:
            zscore = -99.0
        else:
            mu = est_prob * TweetNum_day
            sigma = math.sqrt(mu*(1-est_prob))
            zscore = round((df_currentDay*1.0-mu)/sigma, 4)
        #print docid, date, df_currentDay, mu, est_prob, sigma, zscore
        zscoreDay.append((date, zscore))

        #if docid in range(50, 70) or docid in range(150, 170):
        if docid in range(100000, 100050):
            print "#################################"
            print nnDayCounter.most_common()
            print sorted(zscoreDay, key = lambda a:a[1], reverse=True)
        zscoreDayArr.append(zscoreDay)
    print "## Tweets zscore by day obtained at", time.asctime()
    return zscoreDayArr


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
            sigma = math.sqrt(mu*(1-est_prob))
            #print docid, day, simDf, mu, est_prob, sigma
            zscore = round((simDf*1.0-mu)/sigma, 4)
            zscoreDay.append((day, zscore))
            #if zscore > 5.0:
            #    zscoreTest.append(math.floor(zscore))
        #if docid in range(50, 70) or docid in range(150, 170):
        #if len(zscoreTest) > 0:
        if docid in range(100000, 102000):
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
        if seqDayHash[docid] != day:
            continue
        if zscoreDay is None:
            continue
        zscore = None
        if len(zscoreDay) == 1:
            if zscoreDay[0][0] == day:
                zscore = zscoreDay[0][1] # zscoreDay from getBursty_tw
        else:
            print zscoreDay
            zscore = dict(zscoreDay).get(day) # zscoreDay from getBursty
        if zscore is None:
            continue
        #zscores.append(round(zscore, 1))
        zscores.append(math.floor(zscore))
        if zscore > thred_zscore:
            burstySeqIdArr.append(docid)
    print "## Tweets filtering by zscore ", thred_zscore, " in day ", day, " obtained at", time.asctime()
    print "## statistic of zscore", Counter(zscores).most_common()
    return burstySeqIdArr

def clustering(documents, feaVecs, topK_c, topK_t, burstySeqIdArr):
    # kmeans
    kmeans = cluster.KMeans(n_clusters=50).fit(feaVecs)
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

    for label, score in sorted(enumerate(clusterScore), key = lambda a:a[1], reverse=True)[:topK_c]:
        dataIn = [item[0] for item in enumerate(kmeans.labels_) if item[1] == label]

        sumFlag = 3
        ######################
        # sum method 1_ by dist
        if sumFlag == 1:
            dists = list(docDist[:,label])
            distsIn = [(docid, dists[docid]) for docid in dataIn]
            sortedData = sorted(distsIn, key = lambda item: item[1])[:topK_t]
        # sum method 2, by same num
        elif sumFlag == 2:
            textsIn = [documents[docid] for docid in dataIn]
            dataIn_unique_top = [(dataIn[idx], num) for idx, num in sumCluster(textsIn, topK_t, False, None)]
        # sum method 3, by nn of thred_0.9. Work Best
        elif sumFlag == 3:
            vecsIn = feaVecs[dataIn,:]
            dataIn_unique_top = [(dataIn[idx], num) for idx, num in sumCluster(vecsIn, topK_t, True, 0.9)]

        ######################

        ######################
        # output
        print "############################"
        print "** cluster", label, score, ", #tweet", labelCounter[label]
        #for docid, score in sortedData:
        for docid, score in dataIn_unique_top:
            print burstySeqIdArr[docid], score, "\t", documents[docid]

# if nnFlag = True, dataset = textsIn; else, dataset = vecsIn
def sumCluster(dataset, K, nnFlag, sameTweetSimThred):
    #simMatrix = None
    #if nnFlag:
    #    simMatrix = pairwise.cosine_similarity(dataset)
    sameTweetClusters = [[0]]
    for seqid, text in enumerate(dataset[1:], start=1):
        added = None
        for stcid, stc in enumerate(sameTweetClusters):
            sameFlag = False
            if simMatrix is None:
                if text == dataset[stc[0]]:
                    sameFlag = True
            else:
                if simMatrix[seqid][stc[0]] > sameTweetSimThred:
                    sameFlag = True

            if sameFlag:
                stc.append(seqid)
                added = (stcid, stc)
                break
        if added is None:
            sameTweetClusters.append([seqid])
        else:
            sameTweetClusters[added[0]] = added[1]
    sameTweetClusterNum = [(stcid, len(stc)) for stcid, stc in enumerate(sameTweetClusters)]
    top = sorted(sameTweetClusterNum, key = lambda a:a[1], reverse=True)[:K]
    top = [(sameTweetClusters[item[0]][0], item[1]) for item in top]
    return top

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

        if docid in range(3):
            common = set(nn_onedoc_nnmodel) & set(nn_onedoc_sim)
            union = set(nn_onedoc_nnmodel) | set(nn_onedoc_sim)
            print "#### docid", docid
            print common
            print set(nn_onedoc_sim)-common
            print set(nn_onedoc_nnmodel)-common
            print union
            print len(nn_onedoc_sim), len(nn_onedoc_nnmodel), len(common), len(union), score
    jcSim = round(np.mean(np.asarray(evalNNModel_scoreArr)), 4)
    print "jaccard sim", jcSim
    return jcSim
