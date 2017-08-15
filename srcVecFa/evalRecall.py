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

def loadGold(filename):
    gold_topics = {}
    for item in os.listdir(filename): # eg_item = 7_11_2012_0_0.txt
        timeWindow = item[item.find("2012")+5:-4]
        gold_topics_timeWindow = open(filename+"/"+item, "r").readlines()
        gold_topics_timeWindow = [topic.strip() for topic in gold_topics_timeWindow]
        gold_topics[timeWindow] = gold_topics_timeWindow
    return gold_topics


def extractNewsWords(text):
    tabArr = text.split("\t")
    mandatoryWords = tabArr[0].split(";")
    if len(tabArr) > 1:
        optimalWords = ";".join([wstr for wstr in tabArr[1:] if len(wstr)>0]).split(";")
    else: optimalWords = []
    #print text
    #print mandatoryWords, optimalWords
    return mandatoryWords, optimalWords

    #text = re.sub(";", " ", text)
    #text = re.sub("\[\]", "", text)
    #return text.split()

# return None or matchedString
def wordMatch(gword, sword):
    gword = gword.strip()
    if gword[0] == "[" and gword[-1] == "]":
        gwords = gword.split(" ")
        match_each_gwords = [word for word in gwords if Levenshtein.ratio(word, sword) >= 0.8]
        if len(match_each_gwords) == 0: return None
        else: return " ".join(match_each_gwords)
    else:
        if Levenshtein.ratio(gword, sword) >= 0.8: return gword
        else: return None


def tClusterMatchNews_content(textsIn, goldNews):
    #stemmer = PorterStemmer()
    matchedNews_c = []
    kpre = []
    krec = []
    for newsIdx, newsText in enumerate(goldNews):
        #print newsIdx, newsText
        mandWords, optWords = extractNewsWords(newsText)

        tWords = []
        tNumIn = sum([item[1] for item in textsIn])
        for textsItem in textsIn[:min(2, len(textsIn))]:
            text, dnum = textsItem
            tWords.extend(text.lower().split())

        tWords = [word.strip("@#") for word in tWords]

        mandMatched = set([nw for nw in mandWords for tw in tWords if wordMatch(nw, tw) is not None])
        commonWithOpt = set([tw for tw in tWords for nw in optWords if wordMatch(nw, tw) is not None])
        commonWithMand = set([tw for tw in tWords for nw in mandWords if wordMatch(nw, tw) is not None])

        if len(mandMatched) == len(mandWords):
            matchedNews_c.append(newsIdx)
            kpre_val = (len(commonWithMand) + len(commonWithOpt))*1.0/len(tWords)
            krec_val = (len(mandMatched) + len(commonWithOpt))*1.0/(len(mandWords) + len(optWords))
            kpre.append(kpre_val)
            krec.append(krec_val)
            #print len(mandWords),len(optWords), len(mandMatched), len(commonWithOpt), kpre_val, krec_val
            #print commonWithMand

    if len(matchedNews_c) == 0:
        return None, None, None
    return matchedNews_c, kpre, krec

def evalTClusters(tweetClusters, documents, goldNews, outputDetail):
    trueCluster = []
    matchedNews = []
    kpreArr = []
    krecArr = []
    for outIdx, cluster in enumerate(tweetClusters):
        clabel, cscore, docsIn = cluster
        tNumIn = sum([dscore for did, dscore in docsIn])
        textsIn = [(documents[docid], num) for docid, num in docsIn]

        matchedNews_c, kpre, krec = tClusterMatchNews_content(textsIn, goldNews)

        if outputDetail:
            print "############################"
            print "1-** cluster", outIdx, clabel, cscore, ", #tweet", tNumIn#, compsIn
            for docid, dscore in docsIn:
                #if docsIn[0][1] > 1 and dscore < 2: continue
                print docid, dscore, "\t", documents[docid]

        if matchedNews_c is None: continue

        trueCluster.append((outIdx, matchedNews_c))
        matchedNews.extend(matchedNews_c)
        kpreArr.extend(kpre)
        krecArr.extend(krec)

        if outputDetail:
            sortedNews = Counter(matchedNews_c).most_common()
            for idx, sim in sortedNews:
                print '-MNews', idx, sim, goldNews[idx]

    matchedNews = Counter(matchedNews)
    if 0:
        print "TrueClusters", trueCluster
        matchNewsDetails = {}
        for cid, nids in trueCluster:
            for nid in set(nids): 
                if nid in matchNewsDetails:
                    matchNewsDetails[nid].append(cid)
                else:
                    matchNewsDetails[nid] = [cid]
        print "MCNewsDetail", sorted(matchNewsDetails.items(), key = lambda a:a[0])
    return len(trueCluster), len(matchedNews), kpreArr, krecArr


def outputEval(Nums):
    print "## Eval newsMatchedCluster", sum(Nums[0]), sum(Nums[1]), round(float(sum(Nums[0])*100)/sum(Nums[1]), 2)
    print "## Eval sysMatchedNews", sum(Nums[2]), sum(Nums[3]), round(float(sum(Nums[2])*100)/sum(Nums[3]), 2)

def outputEval_day(Nums):
    print "## newsPre", Nums[0][-1], Nums[1][-1], ("%.2f" %(Nums[0][-1]*100.0/Nums[1][-1])), "\t",
    print "## newsRecall", Nums[2][-1], Nums[3][-1], "\t", round(float(Nums[2][-1]*100)/Nums[3][-1], 2)

def evalOutputFAEvents(dayClusters, outputDays, devDays, testDays, topK_c, Kc_step, goldFA):
    outputDetail = False

    for sub_topK_c in range(Kc_step, topK_c+1, Kc_step):
        print "KC", sub_topK_c
        if sub_topK_c == topK_c: outputDetail = True

        evalMetrics = []
        dev_Nums = [[], [], [], []] # trueCNums, cNums, matchNNums, nNums 
        test_Nums = [[], [], [], []]
        for cItem in dayClusters:
            if cItem is None: continue
            day, texts_day, dataset_day, tweetClusters = cItem
            if tweetClusters is None: continue
            if day not in outputDays: continue

            tr = 0.0
            kp = 0.0
            kr = 0.0
            matchNNum = 0

            sub_tweetClusters = tweetClusters[:sub_topK_c]
            goldNews = [item[1] for item in goldFA if item[0] == day][0]

            if outputDetail:
                print "## News in day", day
                for item in goldNews:
                    print item
                print "## Output details of Clusters in day", day

            trueCNum, matchNNum, kpreArr, krecArr = evalTClusters(sub_tweetClusters, texts_day, goldNews, outputDetail)

            tr = matchNNum*1.0/len(goldNews)
            if len(kpreArr) > 0: kp = np.mean(kpreArr)
            if len(krecArr) > 0: kr = np.mean(krecArr)
            metrics = [tr, kp, kr]
            evalMetrics.append(metrics)
            print "## topic recall, Keyword pre, rec", metrics

            if day in devDays:
                dev_Nums[0].append(trueCNum)
                dev_Nums[1].append(len(sub_tweetClusters))
                dev_Nums[2].append(matchNNum)
                dev_Nums[3].append(len(goldNews))
                if trueCNum > 0:
                    outputEval_day(dev_Nums)
            if day in testDays:
                test_Nums[0].append(trueCNum)
                test_Nums[1].append(len(sub_tweetClusters))
                test_Nums[2].append(matchNNum)
                test_Nums[3].append(len(goldNews))
                if trueCNum > 0:
                    outputEval_day(test_Nums)
                
        ##############

        ##############
        #output evaluation topic recall, keyword prec, keyword recall
        trs = [item[0] for item in evalMetrics]
        kps = [item[1] for item in evalMetrics]
        krs = [item[2] for item in evalMetrics]
        print "TR, KP, KR", np.mean(trs), np.mean(kps), np.mean(krs)

        ##############
        # output evaluation metrics_recall
        if sum(dev_Nums[1]) > 0:
            print "** Dev exp in topK_c", sub_topK_c
            outputEval(dev_Nums)
        if sum(test_Nums[1]) > 0:
            print "** Test exp in topK_c", sub_topK_c
            outputEval(test_Nums)
        ##############

