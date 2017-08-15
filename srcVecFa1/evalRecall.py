import sys
import time
from collections import Counter

import numpy as np
from sklearn.metrics import pairwise
from zpar import ZPar
from nltk.stem.porter import PorterStemmer

sys.path.append("../srcOie/")
from labelGold import *
from tweetVec import getVec
from pivotRanking import JaccardSim

import Levenshtein # pip install python-Levenshtein

def extractNewsWords(text):
    text = re.sub(";", " ", text)
    text = re.sub("\[\]", "", text)
    return text.split()
   
def tClusterMatchNews_content(textsIn, goldNews):
    stemmer = PorterStemmer()
    matchedNews_c = []
    for newsIdx, newsText in enumerate(goldNews):
        newsWords = extractNewsWords(newsText)

        tWords = []
        tNumIn = sum([item[1] for item in textsIn])
        for textsItem in textsIn[:min(2, len(textsIn))]:
            text, dnum = textsItem
            tWords.extend(text.lower().split())

        commonWords = [nw for nw in newsWords for tw in tWords if Levenshtein.ratio(nw, tw) >= 0.8]
        if len(commonWords) > 0.4 * len(newsWords):
            matchedNews_c.append(newsIdx)

    if len(matchedNews_c) == 0:
        return None
    return matchedNews_c

def evalTClusters(tweetClusters, documents, goldNews, outputDetail):
    trueCluster = []
    matchedNews = []
    for outIdx, cluster in enumerate(tweetClusters):
        clabel, cscore, docsIn = cluster
        tNumIn = sum([dscore for did, dscore in docsIn])
        textsIn = [(documents[docid], num) for docid, num in docsIn]

        matchedNews_c = tClusterMatchNews_content(textsIn, goldNews)

        if outputDetail:
            print "############################"
            print "1-** cluster", outIdx, clabel, cscore, ", #tweet", tNumIn
            for docid, dscore in docsIn:
                #if docsIn[0][1] > 1 and dscore < 2: continue
                print docid, dscore, "\t", documents[docid]

        if matchedNews_c is None: continue

        trueCluster.append(clabel)
        matchedNews.extend(matchedNews_c)

        if outputDetail:
            sortedNews = Counter(matchedNews_c).most_common()
            for idx, sim in sortedNews:
                print '-MNews', idx, sim, goldNews[idx]

    matchedNews = Counter(matchedNews)
    #print "True", trueCluster
    return len(trueCluster), len(matchedNews)


def outputEval(Nums):
    print "## Eval newsMatchedCluster", sum(Nums[0]), sum(Nums[1]), round(float(sum(Nums[0])*100)/sum(Nums[1]), 2)
    print "## Eval sysMatchedNews", sum(Nums[2]), sum(Nums[3]), round(float(sum(Nums[2])*100)/sum(Nums[3]), 2)

def outputEval_day(Nums):
    print "## newsPre", Nums[0][-1], Nums[1][-1], ("%.2f" %(Nums[0][-1]*100.0/Nums[1][-1])), "\t",
    print "## newsRecall", Nums[2][-1], Nums[3][-1], "\t", round(float(Nums[2][-1]*100)/Nums[3][-1], 2)

def evalOutputFAEvents(dayClusters, outputDays, devDays, testDays, topK_c, Kc_step, goldFA):
    outputDetail = False

    for sub_topK_c in range(Kc_step, topK_c+1, Kc_step):
        if sub_topK_c == topK_c: outputDetail = True

        dev_Nums = [[], [], [], []] # trueCNums, cNums, matchNNums, nNums 
        test_Nums = [[], [], [], []]
        for cItem in dayClusters:
            if cItem is None: continue
            day, texts_day, dataset_day, tweetClusters = cItem
            if tweetClusters is None: continue
            if day not in outputDays: continue

            sub_tweetClusters = tweetClusters[:sub_topK_c]
            goldNews = [item[1] for item in goldFA if item[0] == day]

            if outputDetail:
                print "## News in day", day
                for item in goldNews:
                    print item
                print "## Output details of Clusters in day", day

            trueCNum, matchNNum = evalTClusters(sub_tweetClusters, texts_day, goldNews, outputDetail)

            if day in devDays:
                dev_Nums[0].append(trueCNum)
                dev_Nums[1].append(len(sub_tweetClusters))
                dev_Nums[2].append(matchNNum)
                dev_Nums[3].append(len(goldNews))
                outputEval_day(dev_Nums)
            if day in testDays:
                test_Nums[0].append(trueCNum)
                test_Nums[1].append(len(sub_tweetClusters))
                test_Nums[2].append(matchNNum)
                test_Nums[3].append(len(goldNews))
                outputEval_day(test_Nums)
                
        ##############

        ##############
        # output evaluation metrics_recall
        if sum(dev_Nums[1]) > 0:
            print "** Dev exp in topK_c", sub_topK_c
            outputEval(dev_Nums)
        if sum(test_Nums[1]) > 0:
            print "** Test exp in topK_c", sub_topK_c
            outputEval(test_Nums)
        ##############

