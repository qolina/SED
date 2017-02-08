import os
import sys
import time
import timeit
import cPickle
from collections import Counter
from sklearn.metrics import pairwise

from tweetVec import * #loadTweetsFromDir, texts2TFIDFvecs, trainDoc2Vec, trainDoc2Vec
from getSim import getSim, getSim_falconn
from tweetNNFilt import getDF, getBursty, getBursty_tw1, getBursty_tw2, filtering_by_zscore
from tweetSim import testVec_byNN, testVec_byLSH
from tweetClustering import clustering, outputTCluster, compInDoc
from evalRecall import evalTClusters, stockNewsVec, outputEval

sys.path.append("../src/util/")
import snpLoader
import stringUtil as strUtil

def idxTimeWin(dayTweetNumHash, timeWindow):
    dayTweetNumArr = sorted(dayTweetNumHash.items(), key = lambda a:a[0])
    dayTweetNumArr = [item[1] for item in dayTweetNumArr]
    #print [(i+1, sum(dayTweetNumArr[:i+1])) for i in range(len(dayTweetNumArr))]
    dayWindow = [None]*abs(timeWindow[0])
    for date in sorted(dayTweetNumHash.keys())[abs(timeWindow[0]):]:
        datePre = int(date) + timeWindow[0]
        dateAft = int(date) + timeWindow[1]
        start = sum(dayTweetNumArr[:datePre-1])
        end = sum(dayTweetNumArr[:dateAft])
        dayWindow.append((start, end))
    return dayWindow

def dayNewsExtr(newsDayWindow, newsSeqDayHash, vecNews, dayNews, newsSeqComp):
    newsSeqIdDay = sorted([newsSeqId for newsSeqId, dayInt in newsSeqDayHash.items() if dayInt in newsDayWindow])
    vecNewsDay = vecNews[newsSeqIdDay,:]
    textNewsDay = []
    for item in newsDayWindow:
        textNewsDay.extend(dayNews[item])
    newsSeqCompDay = [newsSeqComp[newsSeqId] for newsSeqId in newsSeqIdDay]
    return vecNewsDay, textNewsDay, newsSeqCompDay

def dayNewsTripExtr(newsDayWindow):
    compTrip_News = []
    for dayInt in newsDayWindow:
        if dayInt == 0:
            suffix = "04-30"
        else:
            suffix = "05-"+str(dayInt).zfill(2)
        tripFile = open("../data/snp/snp_triple_in1st_2015-"+suffix, "r")
        compTrip_News_day = cPickle.load(tripFile)
        compTrip_News.extend(compTrip_News_day)

    return compTrip_News

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

snpFilePath = "../data/snp500_sutd"
stock_newsDir = '../ni_data/stocknews/'
newsVecPath = "../ni_data/tweetVec/stockNewsVec1"
doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model"
#l_doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model.large2013.finP"
#l_doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model.large2013"
l_doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model.large2016.finP.dim100"
#l_doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model.large2016.dim100"
#l_doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model.large2016.dim200"
largeCorpusPath = os.path.expanduser("~")+"/corpus/tweet_finance_data/tweetCleanText2016"
#largeCorpusPath = os.path.expanduser("~")+"/corpus/tweet_finance_data/tweetCleanText2013.test"
word2vecModelPath = "../ni_data/tweetVec/w2v1010100-en"
dictPath = "../ni_data/tweetVec/tweets.dict"
corpusPath = "../ni_data/tweetVec/tweets.mm"

fileSuffix_r = "default"
fileSuffix_w = "db.f6_zs5"

nnFilePath = "../ni_data/tweetVec/finance.nn.35w." + fileSuffix_r
dfFilePath = "../ni_data/tweetVec/finance.df.35w." + fileSuffix_r
zscoreFilePath = "../ni_data/tweetVec/finance.zs.35w." + fileSuffix_r
clusterFilePath = "../ni_data/tweetVec/finance.cluster.35w." + fileSuffix_w


####################################################
if __name__ == "__main__":
    print "Program starts at ", time.asctime()

    devDays = ['06', '07', '08']
    testDays = ['11', '12', '13', '14', '15']
    #testDays = ['18', '19', '20', '21', '22']

    ##############
    # Parameters
    Para_NumT = 350000
    Para_train, Para_test = ('-', '-')
    timeWindow = (-3, 3)
    #timeWindow = None

    # '-': even do not need to load
    # '0': load preCalculated
    # '1': need to calculate
    calNN = '0'
    calDF = '0'
    calZs = '0'
    calCluster = "-"
    trainLSH = False
    thred_radius_dist = 0.4 # sqreu = 2*cosine
    thred_zscore = 5.0
    Para_numClusters = -1
    topK_c, topK_t = 20, 5
    default_num_probes_lsh = 20
    Para_newsDayWindow = [0]
    Paras = [Para_NumT, Para_train, Para_test, timeWindow, calNN, calDF, calZs, calCluster, trainLSH, thred_radius_dist, thred_zscore, Para_numClusters, topK_c, topK_t, default_num_probes_lsh, Para_newsDayWindow]
    print "**Para setting"
    print "Para_NumT, Para_train, Para_test, timeWindow, calNN, calDF, calZs, calCluster, trainLSH, thred_radius_dist, thred_zscore, Para_numClusters, topK_c, topK_t, default_num_probes_lsh, Para_newsDayWindow"
    print Paras
    ##############

    sym_names = snpLoader.loadSnP500(snpFilePath)
    snp_syms = [snpItem[0] for snpItem in sym_names]
    snp_comp = [strUtil.getMainComp(snpItem[1]) for snpItem in sym_names]
    symCompHash = dict(zip(snp_syms, snp_comp))
    #testSent = "RT @WSJ  Google searches on mobile devices now outnumber those on PCs in 10 countries including the U . S and Japan"
    #print compInDoc(testSent.lower(), snp_comp, symCompHash)
    #sys.exit(0)

    dataDirPath = parseArgs(sys.argv)
    tweetTexts_all = None
    tweetTexts_all, seqTidHash, seqDayHash, dayTweetNumHash = loadTweetsFromDir(dataDirPath)
    if Para_NumT != 0:
        tweetTexts_all = tweetTexts_all[:Para_NumT]

    #[None]*timeWindow.extend([(start, end), (start, end), ...])
    # int(date)-1
    dayWindow = idxTimeWin(dayTweetNumHash, timeWindow)
    #sys.exit(0)

    ##############
    # training
    trainDoc2Vec(Para_train, doc2vecModelPath, largeCorpusPath, l_doc2vecModelPath, tweetTexts_all)

    ##############
    # testing/using
    dataset = None
    if Para_test[0] in ['0', '1', '2']:
        dataset = getVec(Para_test[0], doc2vecModelPath, l_doc2vecModelPath, len(tweetTexts_all), word2vecModelPath, None)
    if Para_test.find('3') >= 0:
        dataset_w2v = getVec('3', doc2vecModelPath, l_doc2vecModelPath, len(tweetTexts_all), word2vecModelPath, tweetTexts_all)
        if dataset is None:
            dataset = dataset_w2v
        else:
            # concatenate d2v and w2v
            dataset = np.append(dataset, dataset_w2v, axis=1)
            #dataset = np.add(dataset, dataset_w2v)
            dataset_w2v = None # end of using

    if Para_test[0] == '4':
        dataset = texts2TFIDFvecs(tweetTexts_all, dictPath, corpusPath)

    ##############
    # testing vec's performace by finding nearest neighbor
    if 0:
        dataset = dataset[:20000, :]
        simMatrix = pairwise.cosine_similarity(dataset)
        nns_fromSim = [sorted(enumerate(simMatrix[i]), key = lambda a:a[1], reverse=True)[:100] for i in range(simMatrix.shape[0])]
        print "## Similarity Matrix obtained at", time.asctime()
        testVec_byNN(nns_fromSim, tweetTexts_all, 10)
    ##############
    
    ##############
    # get sim, cal zscore, clustering
    if calNN == '1':
        #ngDistArray, ngIdxArray = getSim(dataset, thred_radius_dist)
        if trainLSH:
            trained_num_probes = getSim_falconn(dataset, thred_radius_dist, trainLSH, None, nns_fromSim)
        trained_num_probes = default_num_probes_lsh
        ngIdxArray, indexedInCluster, clusters = getSim_falconn(dataset, thred_radius_dist, False, trained_num_probes, None, None)
        #ngIdxArray = getSim_falconn(dataset, thred_radius_dist, False, trained_num_probes, None, nnFilePath)
        #testVec_byLSH(ngIdxArray, tweetTexts_all)

        nnFile = open(nnFilePath, "wb")
        cPickle.dump(ngIdxArray, nnFile)
        cPickle.dump(indexedInCluster, nnFile)
        cPickle.dump(clusters, nnFile)
    elif calNN == '0':
        nnFile = open(nnFilePath, "rb")
        ngIdxArray = cPickle.load(nnFile)
        indexedInCluster = cPickle.load(nnFile)
        clusters = cPickle.load(nnFile)
        print "## ngIdxArr loaded", nnFilePath, time.asctime(), ngIdxArray.shape, len(indexedInCluster), len(clusters)
    #sys.exit(0)
    ##############

    ##############
    if calDF == '1':
        simDfDayArr = getDF(ngIdxArray, seqDayHash, timeWindow, dataset, tweetTexts_all, indexedInCluster, clusters)
        ngIdxArray = None # end of using
        dfFile = open(dfFilePath, "wb")
        cPickle.dump(simDfDayArr, dfFile)
    elif calDF == '0':
        dfFile = open(dfFilePath, "rb")
        simDfDayArr = cPickle.load(dfFile)
        print "## simDfDayArr loaded", dfFilePath, time.asctime()

    if calZs == '1':
        if timeWindow is None:
            zscoreDayArr = getBursty(simDfDayArr, dayTweetNumHash)
        else:
            #zscoreDayArr = getBursty_tw1(simDfDayArr, seqDayHash)
            zscoreDayArr = getBursty_tw2(simDfDayArr, seqDayHash, dayTweetNumHash)
        simDfDayArr = None # end of using
        zsFile = open(zscoreFilePath, "wb")
        cPickle.dump(zscoreDayArr, zsFile)
    elif calZs == '0':
        zsFile = open(zscoreFilePath, "rb")
        zscoreDayArr = cPickle.load(zsFile)
        print "## zscoreDayArr obtained", zscoreFilePath, time.asctime()

    if 0:
        df_valid = [seqDayHash[docid] for docid, dfItem in enumerate(simDfDayArr) if dfItem is not None]
        print "## df valid distri in days", Counter(df_valid).most_common()
        zs_valid = [seqDayHash[docid] for docid, zscoreDay in enumerate(zscoreDayArr) if zscoreDay is not None]
        print "## zs valid distri in days", Counter(zs_valid).most_common()

    if 0:
        #print clusters
        tc_valid = [seqDayHash[docid[0]] for docid in clusters]
        print "## tweet clusters valid distri in days", Counter(tc_valid).most_common()

    if 1:
        clusters = [item[0] for item in clusters]
        for pDate in devDays:
            zs_pDate = [(docid, zscoreDay[0][0]) for docid, zscoreDay in enumerate(zscoreDayArr) if zscoreDay is not None if seqDayHash[docid] == pDate]
            print "## Statistic zs in day", pDate, "unique/all", sum([1 for docid, zs in zs_pDate if docid in clusters]), len(zs_pDate)

        for pDate in testDays:
            zs_pDate = [(docid, zscoreDay[0][0]) for docid, zscoreDay in enumerate(zscoreDayArr) if zscoreDay is not None if seqDayHash[docid] == pDate]
            print "## Statistic zs in day", pDate, "unique/all", sum([1 for docid, zs in zs_pDate if docid in clusters]), len(zs_pDate)
            sorted_zs = sorted(zs_pDate, key = lambda a:a[1], reverse=True)
            for docid, zs in sorted_zs[:500]:
                print docid, zs, tweetTexts_all[docid]
    sys.exit(0)
    ##############

    ##############
    # prepare news vec for eval recall
    #stockNewsVec(stock_newsDir, snp_comp, word2vecModelPath, newsVecPath)

    # load news vec
    newsVecFile = open(newsVecPath, "r")
    dayNews = cPickle.load(newsVecFile)
    vecNews = cPickle.load(newsVecFile)
    newsSeqDayHash = cPickle.load(newsVecFile)
    newsSeqComp = cPickle.load(newsVecFile)
    ##############

    ##############
    # filtering tweets, clustering
    if calCluster == "1":
        dayArr = sorted(dayTweetNumHash.keys())
        dayClusters = []
        for day in dayArr:
            if day not in devDays and day not in testDays:
            #if int(day) > 15:
                continue
            burstySeqIdArr = filtering_by_zscore(zscoreDayArr, seqDayHash, day, thred_zscore)
            print "## Tweet filtering by zscore done.", len(burstySeqIdArr), " out of", dayTweetNumHash[day]
            if Para_NumT != 0:
                burstySeqIdArr = [seqid for seqid in burstySeqIdArr if seqid < Para_NumT]
            if len(burstySeqIdArr) < 10:
                #print "## Too less documents current day", day, len(burstySeqIdArr)
                continue

            # obtain bursty texts, featureVectors
            texts_day = [tweetTexts_all[seqid] for seqid in burstySeqIdArr]
            #dataset_day = [dataset[seqid] for seqid in burstySeqIdArr]
            dataset_day = dataset[burstySeqIdArr,:]

            if Para_numClusters != -1:
                numClusters = Para_numClusters
            else:
                numClusters = 100
                if len(burstySeqIdArr) <= 10000:
                    numClusters = 50
                if len(burstySeqIdArr) >= 20000:
                    numClusters = 200
            print "## Begin clustering in ", day, " #tweet, #vecDim", dataset_day.shape, " numClusters", numClusters
            tweetClusters = clustering(texts_day, dataset_day, numClusters, topK_c, topK_t, burstySeqIdArr, snp_comp, symCompHash)
            dayClusters.append((day, texts_day, dataset_day, tweetClusters))

        clusterFile = open(clusterFilePath, "w")
        cPickle.dump(dayClusters, clusterFile)
    elif calCluster == "0":
        clusterFile = open(clusterFilePath, "r")
        dayClusters = cPickle.load(clusterFile)


    ##############
    ## evaluation and output
    step = topK_c
    for sub_topK_c in range(step, topK_c+1, step):

        dev_Nums = [[], [], [], []] # trueCNums, cNums, matchNNums, nNums 
        test_Nums = [[], [], [], []]
        for day, texts_day, dataset_day, tweetClusters in dayClusters:
            #if day not in devDays: continue
            #if day not in testDays: continue
            if day not in devDays and day not in testDays: continue
            sub_tweetClusters = tweetClusters[:sub_topK_c]

            # output
            #outputTCluster(sub_tweetClusters, texts_day)

            newsDayWindow = [int(day)+num for num in Para_newsDayWindow]

            #textNewsDay = dayNewsTripExtr(newsDayWindow)
            #print compTrip_News

            vecNewsDay, textNewsDay, newsSeqCompDay = dayNewsExtr(newsDayWindow, newsSeqDayHash, vecNews, dayNews, newsSeqComp)
            if 1:
                print "## News in day", day
                for item in textNewsDay:
                    print item

            outputDetail = False
            if sub_topK_c == topK_c:
                outputDetail = True
                print "## Output details of Clusters in day", day
            trueCNum, matchNNum = evalTClusters(sub_tweetClusters, dataset_day, texts_day, vecNewsDay, textNewsDay, newsSeqCompDay, snp_comp, symCompHash, outputDetail)
            #trueCNum, matchNNum = evalTClusters(sub_tweetClusters, None, texts_day, None, textNewsDay, None, snp_comp, symCompHash, outputDetail)

            if day in devDays:
                dev_Nums[0].append(trueCNum)
                dev_Nums[1].append(len(sub_tweetClusters))
                dev_Nums[2].append(matchNNum)
                dev_Nums[3].append(len(textNewsDay))
            if day in testDays:
                test_Nums[0].append(trueCNum)
                test_Nums[1].append(len(sub_tweetClusters))
                test_Nums[2].append(matchNNum)
                test_Nums[3].append(len(textNewsDay))
        ##############

        ##############
        # output evaluation metrics_recall
        print "** Dev exp in topK_c", sub_topK_c
        outputEval(dev_Nums)
        print "** Test exp in topK_c", sub_topK_c
        outputEval(test_Nums)
        ##############

    print "Program ends at ", time.asctime()
