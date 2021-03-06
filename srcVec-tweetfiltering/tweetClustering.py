import os
import sys
import time
import timeit
import math
from collections import Counter

import numpy as np
from sklearn import cluster
from sklearn.metrics import pairwise
from scipy.sparse import csr_matrix, issparse, vstack

def getNgramComp(compName):
    words = compName.split()
    grams = []
    for i in range(len(words)):
        for j in range(i, len(words)):
            if i == j and words[i] == "&": continue
            grams.append(" ".join(words[i:j+1]))
    return set(grams)


def findComp_name(headline, snp_comp):
    matchedComp = [(gram, compName) for compName in snp_comp for gram in getNgramComp(compName) if " "+gram+" " in " "+headline+" "]
    if len(matchedComp) == 0:
        return None
    matchScore = [round(len(gram.split())*1.0/len(compName.split()), 2) for gram, compName in matchedComp]
    fullMatch = [matchedComp[idx] for idx in range(len(matchedComp)) if matchScore[idx] >= 0.8]
    if len(fullMatch) < 1: return None
    #print fullMatch
    if len(fullMatch) > 1:
        match_unique = {}
        for item in fullMatch:
            if item[1] in match_unique:
                if len(item[0]) > len(match_unique[item[1]]):
                    match_unique[item[1]] = item[0]
            else:
                match_unique[item[1]] = item[0]
        fullMatch = [(item[1], item[0]) for item in match_unique.items()]
    return fullMatch


def compInCluster(textsIn, snp_comp, symCompHash, filtLong, nameOnly):
    compsMatch = []
    for docText, dnum in textsIn:
        comps = compInDoc(docText.lower(), snp_comp, symCompHash, filtLong, nameOnly)
        if comps is None: continue
        for comp in comps:
            compsMatch.extend([comp[1]]*dnum)
    compsMatch = Counter(compsMatch)
    return compsMatch

def topCompInCluster(compsMatch, tNumIn, compNum):
    compsIn = [(comp, count) for comp, count in compsMatch.most_common(compNum) if count >= tNumIn/2]
    return compsIn

def compInDoc(docText, snp_comp, symCompHash, filtLong, nameOnly):
    comps_name = findComp_name(docText, snp_comp)

    if nameOnly:
        comps = comps_name
    else:
        comps = [(word, symCompHash[word[1:]]) for word in docText.split() if word[0]=='$' and word[1:] in symCompHash]
        if comps_name is not None:
            comps.extend(comps_name)

    if filtLong and len(comps) > 4:
        #print "** Long comps", comps
        return None
    return comps


def sumACluster(dist, vecsIn, topK_t, sameTweetThred):
    if dist == "cosine":
        distMatrix = pairwise.cosine_distances(vecsIn)
    elif dist == "eu":
        distMatrix = pairwise.euclidean_distances(vecsIn, vecsIn)

    sameTweetClusters = [[0]]
    for seqid, text in enumerate(vecsIn[1:], start=1):
        added = None
        for stcid, stc in enumerate(sameTweetClusters):
            sameFlag = False
            if distMatrix[seqid][stc[0]] <= sameTweetThred:
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
    numIn = len(sameTweetClusterNum)
    top = sorted(sameTweetClusterNum, key = lambda a:a[1], reverse=True)[:min(topK_t, numIn)]
    top = [(sameTweetClusters[item[0]][0], item[1]) for item in top]
    return top

def cashInCluster(textsIn):
    cashIn = [(word, num) for text, num in textsIn for word in text.split() if word[0] == '$']
    cashIn = [(word, num) for word, num in cashIn if len(word)> 1 and word[1].isalpha()==True]
    return cashIn

def distToSeed(tweetVecs, seedTweetVecs):
    #seedNews = []
    distToSeedTweets = pairwise.euclidean_distances(tweetVecs, seedTweetVecs[range(10),:])
    distToSeedTweets = np.mean(distToSeedTweets)#/len(tweetVecs)

    distToSeedNews = pairwise.euclidean_distances(tweetVecs, seedTweetVecs[range(10, 20),:])
    distToSeedNews = np.mean(distToSeedNews)#/len(tweetVecs)

    return distToSeedTweets, distToSeedNews


# clusterScore: [score_c1, score_c2, ...]
def clusterScoring(cLabels, tLabels, docDist, cDensity, cTexts, cComps, seedTweetVecs, feaVecs):
    clusterScore = []
    clusterScoreDetail = []

    for label, textsIn, density in zip(cLabels, cTexts, cDensity):
        dataIn = [item[0] for item in enumerate(tLabels) if item[1] == label]
        dataOut = [item[0] for item in enumerate(tLabels) if item[0] not in dataIn]
        vecsIn = feaVecs[dataIn, :]
        compsIn = cComps[label].items()
        cashIn = cashInCluster(textsIn)
        tNumIn = len(dataIn)

        distsIn = None
        if docDist is not None:
            distsIn = docDist[dataIn,label]
        compsNum = sum([item[1] for item in compsIn])
        cashNum = sum([item[1] for item in cashIn])
        if cashNum in [0, 1]: cashNum += 0.1
        if compsNum in [0, 1]: compsNum += 0.1

        dotNum = math.log(tNumIn)
        #inDist = np.mean(distsIn)
        #compScore = float(compsNum)/tNumIn
        #cashScore = float(cashNum)/tNumIn

        inDist_center = None
        if distsIn is not None:
            inDist_center = 1.0/(1.0 + math.exp(np.mean(distsIn)))
        inDist = 1.0/(1.0 + math.exp(density))
        #compScore = float(compsNum)/tNumIn
        #cashScore = float(tNumIn)/cashNum
        compScore = math.log(1+float(compsNum)/tNumIn)
        cashScore = 1/math.log(1+cashNum)
        distToST = distToSeed(vecsIn, seedTweetVecs)
        distToSTweet = math.log(1+distToST[0])
        distToSNews = math.log(2-distToST[1])

        scoreArr = [dotNum, inDist, compScore, cashScore, distToSTweet] #, distToSNews]
        #scoreArr = [dotNum, inDist_center, compScore, cashScore, distToSTweet]
        score = np.prod(scoreArr)

        # all
        clusterScore.append(score)
        # separate
        #clusterScore.append(np.prod([inDist, compScore, cashScore, distToSTweet]))
        #clusterScore.append(np.prod([dotNum, compScore, cashScore, distToSTweet]))
        #clusterScore.append(np.prod([dotNum, inDist, cashScore, distToSTweet]))
        #clusterScore.append(np.prod([dotNum, inDist, compScore, distToSTweet]))
        #clusterScore.append(np.prod([dotNum, inDist, compScore, cashScore]))

        #scoreArr = [tNumIn, dotNum, inDist, compScore, compsNum, cashScore, distToSTweet]
        scoreArr = [round(item, 2) for item in scoreArr] + compsIn
        clusterScoreDetail.append(scoreArr)
    return clusterScore, clusterScoreDetail

def clusterSummary(sumFlag, clusterScore, cLabels, tLabels, docDist, cDocs_zip, feaVecs, topK_c, topK_t):
    topK_c = min(topK_c, len(clusterScore))
    # ranking and summarize clusters
    tweetClusters = []
    for label, score in sorted(zip(cLabels, clusterScore), key = lambda a:a[1], reverse=True)[:topK_c]:
        dataIn = [item[0] for item in enumerate(tLabels) if item[1] == label]
        topK_t = min(topK_t, len(dataIn))
        ######################
        # sum method 1_ by dist
        if sumFlag == 1:
            dists = list(docDist[:,label])
            distsIn = [(docid, dists[docid]) for docid in dataIn]
            sortedData = sorted(distsIn, key = lambda item: item[1])[:topK_t]
        # sum method 2, by same num
        elif sumFlag == 2:
            dataIn_unique_top = Counter(dict(cDocs_zip[label])).most_common(topK_t)
        # sum method 3, by nn of thred_0.9. Work Best
        elif sumFlag == 3:
            vecsIn = feaVecs[dataIn,:]
            dataIn_unique_top = [(dataIn[idx], num) for idx, num in sumACluster("eu", vecsIn, topK_t, 0.2)]
        ######################
        tweetClusters.append((label, score, dataIn_unique_top))

    return tweetClusters

# algor: "kmeans", "affi", "spec", "agg"(ward-hierarchical), "dbscan"
def clusterTweets(algor, documents, feaVecs, clusterArg, snp_comp, symCompHash):
    docDist = None
    if algor == "kmeans" or algor == "default":
        # kmeans: fast, a little bit worse performance than agglomerative
        clusterModel = cluster.KMeans(n_clusters=clusterArg).fit(feaVecs)
        docDist = clusterModel.transform(feaVecs)
    elif algor == "affi":
        # affinity: too slow
        clusterModel = cluster.AffinityPropagation().fit(feaVecs)
        clusterCenters = clusterModel.cluster_centers_
        docDist = pairwise.euclidean_distances(feaVecs, clusterCenters) #, squared=True)
    elif algor == "spec":
        # spectral: too slow
        clusterModel = cluster.SpectralClustering(n_clusters=clusterArg).fit(feaVecs)
    elif algor == "agg":
        #AgglomerativeClustering
        clusterModel = cluster.AgglomerativeClustering(n_clusters=clusterArg).fit(feaVecs)
    elif algor == "dbscan":
        clusterModel = cluster.DBSCAN(eps=clusterArg, min_samples=5, metric='euclidean', algorithm='auto', n_jobs=10).fit(feaVecs)

    tLabels = clusterModel.labels_
    #if algor == "dbscan":
    #    if -1 in tLabels:
    #        tLabels = [item+1 for item in tLabels]
    #print tLabels
    cLabels = sorted(Counter(tLabels).keys())
    if -1 in cLabels: 
        print "Cluster -1: ", list(tLabels).count(-1)
        cLabels.remove(-1)
    

    cTexts = []
    cDocs_zip = []
    cComps = []
    centroids = []
    cDensity = []
    #print cLabels
    for clbl in cLabels:
        dataIn = [item[0] for item in enumerate(tLabels) if item[1] == clbl]
        vecsIn = feaVecs[dataIn, :]
        if not issparse(vecsIn):
            #centroids.append(csr_matrix(csr_matrix.mean(vecsIn, axis=0)))
            centroids.append(np.mean(vecsIn, axis=0))
        textsIn = [documents[docid] for docid in dataIn]
        textsIn = Counter(textsIn).items()
        dataIn_zip = [(documents.index(text), num) for text, num in textsIn]
        compsIn = compInCluster(textsIn, snp_comp, symCompHash, False, True)
        cTexts.append(textsIn)
        cComps.append(compsIn)
        cDocs_zip.append(dataIn_zip)
        inDist = pairwise.euclidean_distances(vecsIn, vecsIn)
        cDensity.append(np.mean(inDist))
        if 0:
            print clbl, cDensity[-1]
            for item in textsIn: print item
            print compsIn

    # spectral
    if docDist is None:
        #if issparse(centroids):
        #    print "Sparse centroid"
        #    centroids = vstack(centroids, format='csr')
        #    docDist = None
        if not issparse(feaVecs):
            docDist = pairwise.euclidean_distances(feaVecs, centroids)
    return cLabels, tLabels, docDist, cDensity, cTexts, cComps, cDocs_zip


def clustering(algor, documents, feaVecs, clusterArg, topK_c, topK_t, burstySeqIdArr, snp_comp, symCompHash, seedTweetVecs):

    if algor == "agg" and issparse(feaVecs):
        feaVecs = feaVecs.toarray()
    cLabels, tLabels, docDist, cDensity, cTexts, cComps, cDocs_zip = clusterTweets(algor, documents, feaVecs, clusterArg, snp_comp, symCompHash)
    print "## Clustering done. algorithm", algor, " #cluster", len(cLabels), time.asctime()
    if len(cLabels) < 5:
        return None, None
    # scoring clusters
    clusterScore, clusterScoreDetail = clusterScoring(cLabels, tLabels, docDist, cDensity, cTexts, cComps, seedTweetVecs, feaVecs)
    print "## Clustering scoring done.", time.asctime()

    sumFlag = 3
    tweetClusters = clusterSummary(sumFlag, clusterScore, cLabels, tLabels, None, cDocs_zip, feaVecs, topK_c, topK_t)
    print "## Clustering summary done.", time.asctime()
    return tweetClusters, clusterScoreDetail


def outputTCluster(tweetClusters, documents, clusterScoreDetail):
    for outIdx, citem in enumerate(tweetClusters):
        clabel, cscore, docsIn = citem
        print "############################"
        print "** cluster", outIdx, clabel, cscore, ", #tweet", sum([repeatNum for docid, repeatNum in docsIn])
        #print "dotNum, inDist, outDist, compScore, cashScore, distToSTweet, distToSNews"
        print "dotNum, inDist, compScore, cashScore, distToSTweet"
        print clusterScoreDetail[clabel]
        for docid, dscore in docsIn:
            print docid, dscore, "\t", documents[docid]


