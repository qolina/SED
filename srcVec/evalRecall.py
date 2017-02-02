import sys
import time
from collections import Counter

import numpy as np
from sklearn.metrics import pairwise

sys.path.append("../srcOie/")
from labelGold import *
from tweetVec import getVec
from pivotRanking import JaccardSim

from tweetClustering import findComp_name, compInCluster, topCompInCluster

def debugMainNameComp(snp_comp):
    snp_comp1 = [sym_names[i][0]+"\t"+sym_names[i][1]+"\t"+snp_comp[i] for i in range(len(snp_comp)) if len(snp_comp[i].split()) == 1]
    snp_comp2 = [sym_names[i][0]+"\t"+sym_names[i][1]+"\t"+snp_comp[i] for i in range(len(snp_comp)) if len(snp_comp[i].split()) == 2]
    snp_comp3 = [sym_names[i][0]+"\t"+sym_names[i][1]+"\t"+snp_comp[i] for i in range(len(snp_comp)) if len(snp_comp[i].split()) > 2]
    print len(snp_comp1), len(snp_comp2), len(snp_comp3)
    print "\n".join(sorted(snp_comp1))
    print "\n".join(sorted(snp_comp2))
    print "\n".join(sorted(snp_comp3))

#[day_0430, day_0501, ...]
def extractStockNews(stock_newsDir, snp_comp, sentNUM):
    dayNews = []
    for dayDir in sorted(os.listdir(stock_newsDir)):
        if len(dayDir) != 10: continue
        #if int(dayDir[-2:]) > 5: continue
        #if dayDir != "2015-04-30": continue
        newsContents = set()
        for newsfile in sorted(os.listdir(stock_newsDir + dayDir)):
            #print "##############################################################"
            content = open(stock_newsDir + dayDir + "/" + newsfile, "r").read()
            printable = set(string.printable)
            content = filter(lambda x:x in printable, content)
            #print content
            #print "##############################################################"

            #sents = get_valid_news_content(content)
            sents = get_valid_1stpara_news(content)
            if sents is None: continue
            headline = re.sub("^(rpt )?update\s*\d+\s", "", " ".join(sents[:sentNUM]).lower())
            newsContents.add(headline)

        oneDayNews = [] # [(matchedSNPComp, headline), ...]
        # matchedSNPComp: [(matchedPart, WholeCompName), ...]
        fullNameNum = 0
        doubtCounter = Counter()

        for headline in newsContents:
            fullMatch = findComp_name(headline, snp_comp)

            if fullMatch is not None and len(fullMatch) < 3:
                fullNameNum += 1
                oneDayNews.append((fullMatch, headline))

            #doubtMatch = [matchedComp[idx] for idx in range(len(matchedComp)) if matchScore[idx] > 0.33 and matchScore[idx] < 0.66]
            #wrongMatch = [matchedComp[idx] for idx in range(len(matchedComp)) if matchScore[idx] <= 0.33]

        #print "full", fullNameNum, len(newsContents), round(float(fullNameNum)/len(newsContents), 2)
        print "## Stock news extracting done in day", dayDir, " #snp_matched", fullNameNum, " out of all", len(newsContents), time.asctime()
        dayNews.append(oneDayNews)
    return dayNews

def content2vec(word2vecModelPath, dayNews):
    contentNews = []
    dayNewsNum = []
    newsSeqDayHash = {} # newsSeqId in contentNews: day_int
    newsSeqComp = {} # newsSeqId in contentNews: matchComp
    for dayIdx, oneDayNews in enumerate(dayNews):
        newsHeadlines = [item[1] for item in oneDayNews]
        matchedComps = [item[0] for item in oneDayNews]
        dayNewsNum.append(len(newsHeadlines))
        #print newsHeadlines
        for seqid, comp in enumerate(matchedComps, start=len(contentNews)):
            newsSeqDayHash[seqid] = dayIdx
            newsSeqComp[seqid] = comp
        contentNews.extend(newsHeadlines)
    vecNews = getVec('3', None, None, None, word2vecModelPath, contentNews)
    return vecNews, newsSeqDayHash, newsSeqComp

def stockNewsVec(stock_newsDir, snp_comp, word2vecModelPath, newsVecPath, sentNUM):
    dayNews = extractStockNews(stock_newsDir, snp_comp, sentNUM)
    vecNews, newsSeqDayHash, newsSeqComp = content2vec(word2vecModelPath, dayNews)
    outputFile = open(newsVecPath, "w")
    cPickle.dump(dayNews, outputFile)
    cPickle.dump(vecNews, outputFile)
    cPickle.dump(newsSeqDayHash, outputFile)
    cPickle.dump(newsSeqComp, outputFile)

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

def evalTClusters(tweetClusters, feaVecs, documents, vecNewsDay, textNewsDay, newsSeqCompDay, snp_comp, symCompHash):
    trueCluster = []
    matchedNews = []
    for outIdx, cluster in enumerate(tweetClusters):
        clabel, cscore, docsIn = cluster
        tNumIn = sum([dscore for did, dscore in docsIn])
        textsIn = [(documents[docid], num) for docid, num in docsIn]
        compsIn = compInCluster(textsIn, snp_comp, symCompHash, True)
        compsIn = topCompInCluster(compsIn, tNumIn, 2)
        newsMatchComp = [newsIdx for comp, count in compsIn for newsIdx, newsComps in enumerate(newsSeqCompDay) for newsCompItem in newsComps if comp == newsCompItem[0]]
        docIds = [docid for docid, repeatNum in docsIn]
        vecsIn = feaVecs[docIds[0],:]
        simMatrix = pairwise.cosine_similarity(vecsIn, vecNewsDay)
        simToNews = np.sum(simMatrix, axis=0)
        simToNews_byComp = [(idx, simToNews[idx]) for idx in newsMatchComp if simToNews[idx] >= np.mean(simToNews)]
        #print "** ", simToNews.shape, np.max(simToNews), np.min(simToNews), np.mean(simToNews)
        sortedNews = sorted(simToNews_byComp, key = lambda a:a[1], reverse = True)
        if len(simToNews_byComp) > 0:
            trueCluster.append(clabel)
            matchedNews.extend([idx for idx,sim in simToNews_byComp])

        print "############################"
        print "1-** cluster", clabel, cscore, ", #tweet", tNumIn, compsIn
        for docid, dscore in docsIn:
            if dscore < 2: continue
            print docid, dscore, "\t", documents[docid]
        for idx, sim in sortedNews:
            print '-MNews', idx, sim, textNewsDay[idx]

    matchedNews = Counter(matchedNews)
    #print "True", trueCluster
    return len(trueCluster), len(matchedNews)

def outputEval(Nums):
    print "## Eval newsMatchedCluster", sum(Nums[0]), sum(Nums[1]), round(float(sum(Nums[0])*100)/sum(Nums[1]), 2)
    print "## Eval sysMatchedNews", sum(Nums[2]), sum(Nums[3]), round(float(sum(Nums[2])*100)/sum(Nums[3]), 2)

stock_newsDir = '../ni_data/stocknews/'
snpFilePath = "../data/snp500_sutd"
word2vecModelPath = "../ni_data/tweetVec/w2v1010100-en"
newsVecPath = "../ni_data/tweetVec/stockNewsVec2"

###############################################################
if __name__ == "__main__":
    sym_names = snpLoader.loadSnP500(snpFilePath)
    snp_syms = [snpItem[0] for snpItem in sym_names]
    snp_comp = [strUtil.getMainComp(snpItem[1]) for snpItem in sym_names]
    #debugMainNameComp(snp_comp)
    #sys.exit(0)
    stockNewsVec(stock_newsDir, snp_comp, word2vecModelPath, newsVecPath, 2)
    # load news vec
    newsVecFile = open(newsVecPath, "r")
    dayNews = cPickle.load(newsVecFile)
    vecNews = cPickle.load(newsVecFile)
    newsSeqDayHash = cPickle.load(newsVecFile)
    newsSeqComp = cPickle.load(newsVecFile)
 
    day = '05'
    newsSeqIdDay = sorted([newsSeqId for newsSeqId, dayInt in newsSeqDayHash.items() if dayInt == int(day)])
    print len(newsSeqIdDay)
    print vecNews.shape
    vecNewsDay = vecNews[newsSeqIdDay,:]
    newsSeqCompDay = [newsSeqComp[newsSeqId] for newsSeqId in newsSeqIdDay]
