import os
import sys
import time
import timeit
import math
from collections import Counter

import numpy as np
from sklearn import cluster
from sklearn.metrics import pairwise

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


def compInCluster(textsIn, snp_comp, symCompHash, filtLong):
    compsMatch = []
    for docText, dnum in textsIn:
        comps = compInDoc(docText.lower(), snp_comp, symCompHash, filtLong)
        if comps is None: continue
        for comp in comps:
            compsMatch.extend([comp[1]]*dnum)
    compsMatch = Counter(compsMatch)
    return compsMatch

def topCompInCluster(compsMatch, tNumIn, compNum):
    compsIn = [(comp, count) for comp, count in compsMatch.most_common(compNum) if count >= tNumIn/2]
    return compsIn

def compInDoc(docText, snp_comp, symCompHash, filtLong):
    comps_name = findComp_name(docText, snp_comp)
    comps = [(word, symCompHash[word[1:]]) for word in docText.split() if word[0]=='$' and word[1:] in symCompHash]
    if comps_name is not None:
        comps.extend(comps_name)
    if filtLong and len(comps) > 4:
        print "** Long comps", comps
        return None
    return comps


def sumACluster(vecsIn, K, sameTweetSimThred):
    simMatrix = pairwise.cosine_similarity(vecsIn)
    sameTweetClusters = [[0]]
    for seqid, text in enumerate(vecsIn[1:], start=1):
        added = None
        for stcid, stc in enumerate(sameTweetClusters):
            sameFlag = False
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

def cashInCluster(textsIn):
    cashIn = [(word, num) for text, num in textsIn for word in text.split() if word[0] == '$']
    return cashIn

# clusterScore: [score_c1, score_c2, ...]
def clusterScoring(tLabels, docDist, cTexts, cComps):
    clusterScore = []

    for label, textsIn in enumerate(cTexts):
        dataIn = [item[0] for item in enumerate(tLabels) if item[1] == label]
        dataOut = [item[0] for item in enumerate(tLabels) if item[1] != label]
        compsIn = cComps[label].items()
        cashIn = cashInCluster(textsIn)
        tNumIn = len(dataIn)

        distsIn = docDist[dataIn,label]
        distsOut = docDist[dataOut,label]
        compsNum = sum([item[1] for item in compsIn])
        cashNum = sum([item[1] for item in cashIn])
        if cashNum in [0, 1]: cashNum += 0.1
        if compsNum in [0, 1]: compsNum += 0.1

        dotNum = math.log(tNumIn)
        inDist = 1.0/(1.0 + math.exp(np.mean(distsIn)))
        outDist = 1.0/(1.0 + math.exp(-np.mean(distsOut)))
        #compScore = float(compsNum)/tNumIn
        #cashScore = float(tNumIn)/cashNum
        compScore = math.log(compsNum)
        cashScore = 1/math.log(cashNum)
        score = [dotNum, inDist, outDist, compScore, cashScore]
        score = np.prod(score)
        clusterScore.append(score)
        #clusterScore.append(dotNum)
    return clusterScore

def clusterSummary(sumFlag, clusterScore, tLabels, docDist, cDocs_zip, feaVecs, topK_c, topK_t):
    topK_c = min(topK_c, len(clusterScore))
    # ranking and summarize clusters
    tweetClusters = []
    for label, score in sorted(enumerate(clusterScore), key = lambda a:a[1], reverse=True)[:topK_c]:
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
            dataIn_unique_top = [(dataIn[idx], num) for idx, num in sumACluster(vecsIn, topK_t, 0.9)]
        ######################
        tweetClusters.append((label, score, dataIn_unique_top))

    return tweetClusters


def clustering(documents, feaVecs, num_Clusters, topK_c, topK_t, burstySeqIdArr, snp_comp, symCompHash):
    # kmeans
    kmeans = cluster.KMeans(n_clusters=num_Clusters).fit(feaVecs)
    tLabels = kmeans.labels_
    #print tLabels
    cLabels = sorted(Counter(tLabels).keys())
    docDist = kmeans.transform(feaVecs)
    cTexts = []
    cDocs_zip = []
    cComps = []
    for clbl in cLabels:
        dataIn = [item[0] for item in enumerate(tLabels) if item[1] == clbl]
        textsIn = [documents[docid] for docid in dataIn]
        textsIn = Counter(textsIn).items()
        dataIn_zip = [(documents.index(text), num) for text, num in textsIn]
        compsIn = compInCluster(textsIn, snp_comp, symCompHash, False)
        cTexts.append(textsIn)
        cComps.append(compsIn)
        cDocs_zip.append(dataIn_zip)

    # scoring clusters
    clusterScore = clusterScoring(tLabels, docDist, cTexts, cComps)

    sumFlag = 3
    tweetClusters = clusterSummary(sumFlag, clusterScore, tLabels, docDist, cDocs_zip, feaVecs, topK_c, topK_t)
    return tweetClusters

def outputTCluster(tweetClusters, documents):
    for clabel, cscore, docsIn in tweetClusters:
        print "############################"
        print "** cluster", clabel, cscore, ", #tweet", sum([repeatNum for docid, repeatNum in docsIn])
        for docid, dscore in docsIn:
            print docid, dscore, "\t", documents[docid]


