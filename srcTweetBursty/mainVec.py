import os
import sys
import math
import time
import cPickle
from collections import Counter
from sklearn.metrics import pairwise

from tweetVec import * #loadTweetsFromDir, texts2TFIDFvecs, trainDoc2Vec, trainDoc2Vec
from getSim import getSim_falconn, getSim_sparse, getSim_dense
from tweetNNFilt import getDF, getBursty
from tweetClustering import clustering, outputTCluster, compInDoc, clusterTweets, getClusterFeatures, clusterScoring, clusterSummary
from evalRecall import evalTClusters, stockNewsVec, outputEval, dayNewsExtr, evalOutputEvents
from statistic import distDistribution, idxTimeWin, getValidDayWind, statGoldNews, zsDistribution
from statistic import output_zsDistri_day, stat_nn_performance, stat_wordNum

from word2vec import loadWord2Vec

sys.path.append("../src/util/")
import snpLoader
import stringUtil as strUtil

seedTweets = ["$71 , 000 in one trade by follwing their signals more info here $cvc $cvd $cve",
"our penny stock pick on $ppch closed up another 51 . 31 today huge news $cce $cadx $grmn",
"our penny stock alerts gained over 3100 in 6 months see our new pick $erx $rig $pot",
"our stock picks have been seeing massive gains this year special update $nov $ery $tza",
"our stock pick on $thcz is up 638 . 15 for our subscribers get our next pick early $tcb $mck $study",
"gains over 2500 in one trade subscribe here $emo $emq $emr",
"largest food and staples retailing earnings 1 $wmt 2 $cvs 3 $wba chart",
"since your tweet was sent $aapl has dropped 3 . 185 see your featured tweet on market parse",
"insider selling silvio barzi sells 5 , 290 shares of mastercard stock $ma",
"volume alert fb 78 . 43 facebook inc $fb hit a high today of 78 . 94 closing the day 05/07/15 at 7    8 . 43 0 . 33 0",

"oracle ceo sees benefit if rival buys salesforce.com",
"dow chemical to sell agrofresh for $860 mln in asset sale drive",
"expedia inc first quarter profit tops expectations",
"conocophillips first quarter profit falls sharply on oil price decline",
"cigna profit beats estimate as it adds more customers",
"solar panel maker first solar reports quarterly loss",
"expedia inc maintains 2015 earnings guidance  cfo",
"mcgraw hill education prepares for ipo",
"obama to push case for trade deal at nike headquarters in oregon",
"tattoo snafu irks inked apple watch wearers"
]

def getClusteringArg(algor, Para_dbscan_eps, Para_numClusters, tnum):
    clusterArg = None
    if algor == "dbscan":
        clusterArg = Para_dbscan_eps 
    else:
        if Para_numClusters != -1:
            clusterArg = Para_numClusters
        else: # for kmeans: numClusters = [50, 100]  double for hierarchical
            clusterArg = 100
            if tnum >= 20000:
                clusterArg = 200
            if algor == "kmeans":
                clusterArg /= 2
    return clusterArg

##############
def getArg(args, flag):
    arg = None
    if flag in args:
        arg = args[args.index(flag)+1]
    return arg

# arguments received from arguments
def parseArgs(args):
    arg1 = getArg(args, "-in")

    arg_validDaysFlag = getArg(args, "-vld") # 'd', 't', 'v'
    arg_outputDaysFlag = getArg(args, "-out") # 'd', 't', 'v'

    # None: even do not need to load
    # '0': load preCalculated
    # '1': need to calculate
    arg_calDF = getArg(args, "-df") #'0', '1' or None
    arg_calZs = getArg(args, "-zs") #'0', '1' or None
    arg_calCluster = getArg(args, "-cls") #'0', '1' or None

    arg_dfFileSuff = getArg(args, "-dffs") # ''
    arg_zsFileSuff = getArg(args, "-zsfs")
    arg_clsFileSuff = getArg(args, "-clsfs")

    arg_trainVec = getArg(args, "-trainvec") # '0', '1', '2'
    arg_testVec = getArg(args, "-testvec") # '0', '1', '2', '3', '4'

    arg_leftTW = getArg(args, "-ltw") # 10
    arg_rightTW = getArg(args, "-rtw") # 0

    arg_tfBurstyFlag = getArg(args, "-tf") # '1' or None
    arg_zsDelta = getArg(args, "-delta") # 5.0

    arg_clusterAlgor = getArg(args, "-cluster") #"kmeans", "affi", "spec", "agg"(ward-hierarchical), "dbscan"
    arg_KcStep = getArg(args, "-kcs") # Kc or 5
    arg_Kc = getArg(args, "-kc") # 20
    arg_Kt = getArg(args, "-kt") # 5

    if arg1 is None or arg_validDaysFlag is None or arg_testVec is None: # nessensary argument
        print "Usage: python mainVec.py -in inputFileDir -vld d -testvec 3"
        print "-vld [-out]:   dev/test/all to process/output"
        print "[-df] [-zs] [-cls] calculate/load/ignore df|zscore|clustering"
        print "[-dffs] [-zsfs] [-clsfs] file suffix of df|zscore|clustering"
        print "[-trainvec] -testvec     tweet vector"
        print "[-ltw] 30 [-rtw] 0"
        print "[-tf]    choose to use bursty tweet filtering"
        print "[-delta]     threshold of bursty zscore"
        print "[-cluster]   clustering algorithm"
        print "[-kc] [-kt] [-kcs] topK_c, topK_t, Kc_step"
        sys.exit(0)

    return arg1, arg_calDF, arg_calZs, arg_calCluster, arg_dfFileSuff, arg_zsFileSuff, arg_clsFileSuff, arg_trainVec, arg_testVec, arg_leftTW, arg_rightTW, arg_validDaysFlag, arg_outputDaysFlag, arg_tfBurstyFlag, arg_zsDelta, arg_clusterAlgor, arg_Kc, arg_KcStep, arg_Kt



snpFilePath = "../data/snp500_sutd"
stock_newsDir = '../ni_data/stocknews/'
newsVecPath = "../ni_data/tweetVec/stockNewsVec1_Loose1"
word2vecModelPath = "../ni_data/tweetVec/w2v1010100-en"
dictPath = "../ni_data/tweetVec/tweets.dict"
corpusPath = "../ni_data/tweetVec/tweets.mm"

####################################################
if __name__ == "__main__":
    print "Program starts at ", time.asctime()

    (dataDirPath, arg_calDF, arg_calZs, arg_calCluster, arg_dfFileSuff, arg_zsFileSuff, arg_clsFileSuff, arg_trainVec, arg_testVec, arg_leftTW, arg_rightTW, arg_validDaysFlag, arg_outputDaysFlag, arg_tfBurstyFlag, arg_zsDelta, arg_clusterAlgor, arg_Kc, arg_KcStep, arg_Kt) = parseArgs(sys.argv)

    ##############
    # Parameters
    Para_train, Para_test = (arg_trainVec, arg_testVec)
    validDaysFlag, outputDaysFlag = (arg_validDaysFlag, arg_outputDaysFlag)
    (calDF, calZs, calCluster) = (arg_calDF, arg_calZs, arg_calCluster)
    (dfFileSuff, zsFileSuff, clsFileSuff) = arg_dfFileSuff, arg_zsFileSuff, arg_clsFileSuff
    burstyFlag = arg_tfBurstyFlag
    algor = arg_clusterAlgor
    (thred_zscore, topK_c, Kc_step, topK_t) = (arg_zsDelta, arg_Kc, arg_KcStep, arg_Kt)

    if arg_leftTW is None or arg_rightTW is None: timeWindow = None
    else: timeWindow = (-int(arg_leftTW), int(arg_rightTW))
    
    if arg_trainVec is None: Para_train = '-'
    if arg_testVec is None: Para_test = '-'
    if arg_calDF is None: calDF = '-'
    if arg_calZs is None: calZs = '-'
    if arg_calCluster is None: calCluster = "-"
    if arg_dfFileSuff is None: dfFileSuff = ''
    if arg_zsFileSuff is None: zsFileSuff = ''
    if arg_clsFileSuff is None: clsFileSuff = ''
    if calCluster == '1' and arg_clusterAlgor is None: algor = 'dbscan'
    if calCluster == '1' and arg_tfBurstyFlag != '1': burstyFlag = None

    if arg_zsDelta is not None: thred_zscore = float(arg_zsDelta)

    if calCluster != "-" and arg_Kc is None: topK_c = 30
    if calCluster != "-" and arg_Kt is None: topK_t = 5
    if calCluster != "-" and arg_KcStep is None: Kc_step = int(topK_c)
    if arg_Kc is not None: topK_c = int(arg_Kc)
    if arg_Kt is not None: topK_t = int(arg_Kt)
    if arg_KcStep is not None: Kc_step = int(arg_KcStep)

    # fixed para
    thred_radius_dist = 0.4
    default_num_probes_lsh = 20
    Para_dbscan_eps = 0.2
    Para_numClusters = -1
    Para_newsDayWindow = [0]#[-1, 0, +1] #[0]

    Paras = [calDF, calZs, calCluster, Para_train, Para_test, default_num_probes_lsh, thred_radius_dist, thred_zscore, algor, Para_numClusters, Para_dbscan_eps, topK_c, topK_t, Para_newsDayWindow, timeWindow]
    print "**Para setting"
    print "calDF, calZs, calCluster, Para_train, Para_test, default_num_probes_lsh, thred_radius_dist, thred_zscore, algor, Para_numClusters, Para_dbscan_eps, topK_c, topK_t, Para_newsDayWindow, timeWindow"
    print Paras
    ##############


    ######################
    validDays = None
    dataSelect = 1
    if dataSelect == 1:
        devDays = ['06', '07', '08']
        testDays = ['11', '12', '13', '14', '15']
    elif dataSelect == 2:#['18', '19', '20', '21', '22', '26', '27', '28']
        devDays = ['26', '27', '28']
        testDays = ['18', '19', '20', '21', '22']
    elif dataSelect == 3: #[15-31]
        devDays = ['15', '18', '19']
        testDays = ['20', '21', '22', '26', '27', '28']
    elif dataSelect == 9: #[15-31]
        devDays = [str(i).zfill(2) for i in range(1, 32)]
        testDays = []
        #devDays = ['15']#, '16', '17']
        #validDays_fst = ['06', '07', '08', '11', '12', '13', '14', '15']
        #validDays_weekend = ['09', '10', '16', '17', '23', '24', '25']
        validDays = devDays

    if validDays is None:
        if validDaysFlag == 'd': validDays = devDays
        elif validDaysFlag == 't': validDays = testDays
        elif validDaysFlag == 'a': validDays = sorted(devDays + testDays)

    if outputDaysFlag is None or outputDaysFlag == "d": outputDays = devDays
    elif outputDaysFlag == "t": outputDays = testDays
    elif outputDaysFlag == "v": outputDays = validDays

    fileSuf_data = ".65w"# + str(dataSelect)
    dfFilePath = "../ni_data/tweetVec/finance.df" + fileSuf_data + dfFileSuff
    zscoreFilePath = "../ni_data/tweetVec/finance.zs" + fileSuf_data + zsFileSuff
    clusterFilePath = "../ni_data/tweetVec/finance.cluster" + fileSuf_data + clsFileSuff

    ##############
    print "validDays", validDays
    print "outputDays", outputDays


    ##############
    sym_names = snpLoader.loadSnP500(snpFilePath)
    snp_syms = [snpItem[0] for snpItem in sym_names]
    snp_comp = [strUtil.getMainComp(snpItem[1]) for snpItem in sym_names]
    symCompHash = dict(zip(snp_syms, snp_comp))
    ##############


    ##############
    # prepare news vec for eval recall
    #stockNewsVec(stock_newsDir, snp_comp, word2vecModelPath, newsVecPath)
    # load news vec
    newsVecFile = open(newsVecPath, "r")
    print "## news obtained for eval", newsVecPath 
    dayNews = cPickle.load(newsVecFile)
    vecNews = cPickle.load(newsVecFile)
    newsSeqDayHash = cPickle.load(newsVecFile)
    newsSeqComp = cPickle.load(newsVecFile)
    newsVecFile.close()
    #statGoldNews(dayNews)
   ##############

    tweetTexts_all = None
    tweetTexts_all, seqTidHash, seqDayHash, dayTweetNumHash = loadTweetsFromDir(dataDirPath)

    ##############
    dayArr = sorted(dayTweetNumHash.keys()) # ['01', '02', ...]
    if timeWindow is not None:
        dayWindow, dayRelWindow = idxTimeWin(dayTweetNumHash, timeWindow)
        validDayWind = getValidDayWind(validDays, dayArr, dayWindow, dayRelWindow)
        print validDayWind
        print dayRelWindow
    ##############

    ##############
    # testing/using
    if calCluster == "1" or calZs == "1":
    #if calCluster == "1":
        #tweetTexts_all = tweetTexts_all[:300000]
        dataset = None
        if Para_test.find('3') >= 0:
            dataset_w2v = getVec('3', None, None, None, word2vecModelPath, tweetTexts_all+seedTweets)
            seedTweetVecs = dataset_w2v[range(-20, 0), :]
            if dataset is None:
                dataset = dataset_w2v[:-20,:]
            else:
                # concatenate d2v and w2v
                dataset = np.append(dataset, dataset_w2v, axis=1)
                #dataset = np.add(dataset, dataset_w2v)
                dataset_w2v = None # end of using

        if Para_test[0] == '4':
            dataset = texts2TFIDFvecs(tweetTexts_all + seedTweets, dictPath, corpusPath)
            seedTweetVecs = dataset[range(-20, 0), :]
            dataset = dataset[:-20, :]

        dataset = dataset.astype(np.float32)
        print "## Dataset vector obtained. ", time.asctime()

    ##############

    ##############
    # filtering tweets, clustering
    if calCluster == "1":
        dayClusters = []
        #for day in dayArr:
        for day in validDays:
            tweetFCSeqIdArr = [docid for docid, dateItem in seqDayHash.items() if dateItem == day]
            texts_day = [tweetTexts_all[seqid] for seqid in tweetFCSeqIdArr]
            dataset_day = dataset[tweetFCSeqIdArr, :]

            clusterArg = getClusteringArg(algor, Para_dbscan_eps, Para_numClusters, len(tweetFCSeqIdArr))
            print "## Begin clustering in ", day, " #tweet, #vecDim", dataset_day.shape, " algorithm", algor, " clusterArg", clusterArg

            tweetClusters = clusterTweets(algor, texts_day, dataset_day, clusterArg)
            cLabels, tLabels, centroids, docDist = tweetClusters
            print "## Clustering done. ", " #cluster", len(cLabels), time.asctime()

            dayClusters.append((day, tweetClusters))
            print len(dayClusters[0])
        if len(dayClusters) > 0:
            print len(dayClusters[0])
            clusterFile = open(clusterFilePath, "w")
            cPickle.dump(dayClusters, clusterFile)
            clusterFile.close()
            print "## Clustering results stored.", clusterFilePath, time.asctime()
    elif calCluster == "0":
        clusterFile = open(clusterFilePath, "r")
        dayClusters = cPickle.load(clusterFile)
        clusterFile.close()
        print len(dayClusters[0])
        print "## Clustering results loaded.", clusterFilePath, time.asctime()

    if calZs == "1":
        dayOutClusters = []
        for dayClusterItem in dayClusters:
            print len(dayClusterItem)
            #print dayClusterItem[0]
            #print len(dayClusterItem[1]), dayClusterItem[1][0]
            #print len(dayClusterItem[2]), dayClusterItem[2][0]
            #print len(dayClusterItem[3]), dayClusterItem[3]
            #(day, texts_day, dataset_day, tweetClusters) = dayClusterItem
            (day, tweetClusters) = dayClusterItem
            if day not in validDays: continue
            dayInt = int(day)-1

            cLabels, tLabels, centroids, docDist = tweetClusters
            print "## Clustering obtained. ", clusterFilePath, " #cluster", len(cLabels), time.asctime()

            # calculate centroids nnDF/zscore in timeWindow
            if Para_test == "4":
                ngIdxArray = getSim_sparse(day, centroids, dataset, thred_radius_dist, validDayWind[dayInt], dayRelWindow[dayInt])
            else:
                ngIdxArray = getSim_dense(day, centroids, dataset, thred_radius_dist, validDayWind[dayInt], dayRelWindow[dayInt])
            #ngIdxArray, indexedInCluster, clusters = getSim_falconn(dataset, thred_radius_dist, default_num_probes_lsh, None, validDayWind, dayRelWindow)
            simDfArr = getDF(day, ngIdxArray, seqDayHash, timeWindow)
            zscoreArr = getBursty(simDfArr, dayTweetNumHash, day, timeWindow)
            print "## Cluster zscore calculating done.", time.asctime()

            tweetFCSeqIdArr = [docid for docid, dateItem in seqDayHash.items() if dateItem == day]
            texts_day = [tweetTexts_all[seqid] for seqid in tweetFCSeqIdArr]
            dataset_day = dataset[tweetFCSeqIdArr, :]

            clusterFeatures = getClusterFeatures(tweetClusters, texts_day, dataset_day, seedTweetVecs, snp_comp, symCompHash)
            #docDist, cDensity, cTexts, cComps, cDocs_zip, cDistToST = clusterFeatures
            print "## Cluster zscore calculating done.", time.asctime()

            clusterScore, clusterScoreDetail = clusterScoring(tweetClusters, clusterFeatures, zscoreArr)
            print "## Clustering scoring done.", time.asctime()

            sumFlag = 3
            outputClusters = clusterSummary(sumFlag, clusterScore, cLabels, tLabels, dataset_day, topK_c, topK_t)
            print "## Clustering summary done.", time.asctime()

            if 1:
                outputTCluster(outputClusters, texts_day, clusterScoreDetail)

            dayOutClusters.append((day, texts_day, dataset_day, outputClusters))
        clusterOutputFile = open(clusterFilePath+zsFileSuff, "w")
        cPickle.dump(dayOutClusters, clusterOutputFile)
        clusterOutputFile.close()
        print "## Clustering results stored.", clusterFilePath+zsFileSuff, time.asctime()
    else:
        clusterOutputFile = open(clusterFilePath+zsFileSuff, "r")
        dayOutClusters = cPickle.load(clusterOutputFile)
        clusterOutputFile.close()


    ##############
    ## evaluation and output
    evalOutputEvents(dayOutClusters, outputDays, devDays, testDays, topK_c, Kc_step, Para_newsDayWindow, newsSeqDayHash, vecNews, dayNews, newsSeqComp, snp_comp, symCompHash)

    print "Program ends at ", time.asctime()

