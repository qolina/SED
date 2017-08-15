import os
import sys
import math
import time
import cPickle
from collections import Counter
from sklearn.metrics import pairwise

from tweetVec import * #loadTweetsFromDir, texts2TFIDFvecs, trainDoc2Vec, trainDoc2Vec
from getSim import getSim, getSim_falconn, getSim_sparse
from tweetNNFilt import getDF, getBursty, getBursty_tw1, getBursty_tw2, filtering_by_zscore, getBursty_byday, tweetForClustering
from tweetSim import testVec_byLSH
from tweetClustering import clustering, outputTCluster, compInDoc
from evalRecall import evalTClusters, outputEval, evalOutputFAEvents
from statistic import distDistribution, idxTimeWin, getValidDayWind, statGoldNews
from statistic import output_zsDistri_day, stat_nn_performance, stat_wordNum

from word2vec import loadWord2Vec

sys.path.append("../src/util/")
import snpLoader
import stringUtil as strUtil

goldFA = [("28", "[kick kicked] [0-0 0 0];start;off"),
("30", "[ramires chelsea];[goal 1-0 1 0] score;yes"),
("33", "[salomon kalou];run;box mazy"),
("35", "mikel;[yellow card booking] gerrard;foul;agger"),
("37", "[second half 2nd];[kick kicked] off"),
("40", "agger;[booked yellow card] tackle;mikel;challenge"),
("41", "[goal 2-0 2 0];[didier drogba chelsea] score"),
("42", "[frank lampard];[pass assist] ball;drogba"),
("37", "[andy carroll liverpool];[goal 2-1 2 1] score"),
("44", "[first half ht];[half time ht] [1-0 1 0]"),
("46", "[luis suarez];save;cech shot;forces"),
("48", "[andy carroll];line header;cech;over;claim;equalise"),
("50", "[full time final whistle gone] chelsea;champions;congratulations;[2-1 2 1];win")]


seedTweets = []

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
word2vecModelPath = "../ni_data/tweetVec/w2v1010100-en"
dictPath = "../ni_data/tweetVecFa/tweets.dict"
corpusPath = "../ni_data/tweetVecFa/tweets.mm"

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

    if calCluster != "-" and arg_Kc is None: topK_c = 20
    if calCluster != "-" and arg_Kt is None: topK_t = 5
    if calCluster != "-" and arg_KcStep is None: Kc_step = 5
    if arg_Kc is not None: topK_c = int(arg_Kc)
    if arg_Kt is not None: topK_t = int(arg_Kt)
    if arg_KcStep is not None: Kc_step = int(arg_KcStep)
    if Kc_step > topK_c: Kc_step = topK_c

    # fixed para
    thred_radius_dist = 0.4
    default_num_probes_lsh = 20
    Para_dbscan_eps = 0.3
    Para_numClusters = -1
    Para_newsDayWindow = [0]

    Paras = [calDF, calZs, calCluster, Para_train, Para_test, default_num_probes_lsh, thred_radius_dist, thred_zscore, algor, Para_numClusters, Para_dbscan_eps, topK_c, topK_t, Para_newsDayWindow, timeWindow]
    print "**Para setting"
    print "calDF, calZs, calCluster, Para_train, Para_test, default_num_probes_lsh, thred_radius_dist, thred_zscore, algor, Para_numClusters, Para_dbscan_eps, topK_c, topK_t, Para_newsDayWindow, timeWindow"
    print Paras
    ##############


    ######################
    validDays = None
    #devDays = [str(i).zfill(2) for i in [28]]
    devDays = [str(i).zfill(2) for i in [28, 30, 33, 35]]
    testDays = [str(i).zfill(2) for i in [37, 40, 41, 42, 44, 46, 48, 50]]

    if validDays is None:
        if validDaysFlag == 'd': validDays = devDays
        elif validDaysFlag == 't': validDays = testDays
        elif validDaysFlag == 'a': validDays = sorted(devDays + testDays)

    if outputDaysFlag is None or outputDaysFlag == "d": outputDays = devDays
    elif outputDaysFlag == "t": outputDays = testDays
    elif outputDaysFlag == "v": outputDays = validDays

    fileSuf_data = ""# + str(dataSelect)
    dfFilePath = "../ni_data/tweetVecFa/fa.df" + fileSuf_data + dfFileSuff
    zscoreFilePath = "../ni_data/tweetVecFa/fa.zs" + fileSuf_data + zsFileSuff
    clusterFilePath = "../ni_data/tweetVecFa/fa.cluster" + fileSuf_data + clsFileSuff

    ##############
    print "validDays", validDays
    print "outputDays", outputDays


    tweetTexts_all = None
    tweetTexts_all, seqTidHash, seqDayHash, dayTweetNumHash = loadTweetsFromDir(dataDirPath)

    ##############
    dayArr = sorted(dayTweetNumHash.keys()) # ['01', '02', ...]
    if calDF == "1" and timeWindow is not None:
        dayWindow, dayRelWindow = idxTimeWin(dayTweetNumHash, timeWindow)
        validDayWind = getValidDayWind(validDays, dayArr, dayWindow, dayRelWindow)
        print validDayWind
        print dayRelWindow
    ##############

    ##############
    # testing/using
    if calDF == "1" or calCluster == "1":
        dataset = None
        seedTweetVecs = None
        if Para_test.find('3') >= 0:
            dataset_w2v = getVec('3', None, None, len(tweetTexts_all), word2vecModelPath, tweetTexts_all+seedTweets)
            if len(seedTweets) > 0:
                seedTweetVecs = dataset_w2v[range(-len(seedTweets), 0), :]
                if dataset is None:
                    dataset = dataset_w2v[:-len(seedTweets),:]
            else:
                dataset = dataset_w2v

        if Para_test[0] == '4':
            dataset = texts2TFIDFvecs(tweetTexts_all + seedTweets, dictPath, corpusPath)
            seedTweetVecs = dataset[range(-len(seedTweets), 0), :]
            dataset = dataset[:-len(seedTweets), :]

        dataset = dataset.astype(np.float32)
        print "## Dataset vector obtained. ", time.asctime()

    #stat_wordNum(tweetTexts_all, seqDayHash, validDays)
    #stat_nn_performance(dataset, tweetTexts_all)# testing vec's performace by finding nearest neighbor
    #distDistribution(dataset)

    ##############

    ##############
    if calDF == '1':
        # cal NN, do not store NN result, takes too much space. Each day's NN can be calculated in 5 mins
        if Para_test == "4":
            ngIdxArray = getSim_sparse(dataset, thred_radius_dist, validDayWind, dayRelWindow)
            indexedInCluster = None
            clusters = None
        else:
            ngIdxArray, indexedInCluster, clusters = getSim_falconn(dataset, thred_radius_dist, default_num_probes_lsh, None, validDayWind, dayRelWindow)
            #testVec_byLSH(ngIdxArray, tweetTexts_all)

        simDfDayArr = getDF(ngIdxArray, seqDayHash, timeWindow, indexedInCluster, clusters)
        ngIdxArray = None # end of using
        dfFile = open(dfFilePath, "wb")
        cPickle.dump(simDfDayArr, dfFile)
        print "## simDfDayArr writen", dfFilePath, time.asctime()
    elif calDF == '0':
        dfFile = open(dfFilePath, "rb")
        simDfDayArr = cPickle.load(dfFile)
        print "## simDfDayArr loaded", dfFilePath, time.asctime()

    if calZs == '1':
        if len(simDfDayArr) < 75: # stored by day
            if 0:
                # calculate zs only within sub timewindows
                #sub_timewindow = [(-i, i) for i in range(11, 15)]
                #sub_timewindow = [(-i, i) for i in [25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]
                #sub_timewindow = [(-i, i) for i in [15, 0]]
                #sub_timewindow = [(-i, 0) for i in range(1, 15)]
                sub_timewindow = [(-i, 0) for i in [14]]
                for sub_tw in sub_timewindow:
                    sub_zscoreDayArr = getBursty_byday(simDfDayArr, dayTweetNumHash, sub_tw)
                    sub_zsFile = open(zscoreFilePath+str(sub_tw[0]), "wb")
                    cPickle.dump(sub_zscoreDayArr, sub_zsFile)
                    print "## zscoreDayArr stored", sub_zsFile.name, time.asctime()
            zscoreDayArr = getBursty_byday(simDfDayArr, dayTweetNumHash, timeWindow)
        simDfDayArr = None # end of using
        zsFile = open(zscoreFilePath, "wb")
        cPickle.dump(zscoreDayArr, zsFile)
    elif calZs == '0':
        zsFile = open(zscoreFilePath, "rb")
        zscoreDayArr = cPickle.load(zsFile)
        print "## zscoreDayArr obtained", zscoreFilePath, time.asctime()
    #sys.exit(0)

    #output_zsDistri_day(validDays, zscoreDayArr, simDfDayArr, dayTweetNumHash, tweetTexts_all)
    ##############

    ##############
    # filtering tweets, clustering
    if calCluster == "1":
        dayClusters = []
        for day in validDays:
            startNumDay = sum([num for dayItem, num in dayTweetNumHash.items() if int(dayItem)<int(day)])

            if burstyFlag != '1': # choose all tweets
                tweetFCSeqIdArr = [docid for docid, dateItem in seqDayHash.items() if dateItem == day]
            else:
                tweetFCSeqIdArr = tweetForClustering(day, seqDayHash, zscoreDayArr, thred_zscore, startNumDay, dayTweetNumHash)
            if tweetFCSeqIdArr is None: continue

            #########################################################
            # obtain bursty texts, featureVectors
            texts_day = [tweetTexts_all[seqid] for seqid in tweetFCSeqIdArr]
            dataset_day = dataset[tweetFCSeqIdArr, :]

            clusterArg = getClusteringArg(algor, Para_dbscan_eps, Para_numClusters, len(tweetFCSeqIdArr))
            print "## Begin clustering in ", day, " #tweet, #vecDim", dataset_day.shape, " algorithm", algor, " clusterArg", clusterArg
            tweetClusters, clusterScoreDetail = clustering(algor, texts_day, dataset_day, clusterArg, topK_c, topK_t, tweetFCSeqIdArr, seedTweetVecs)

            if 0:
                outputTCluster(tweetClusters, texts_day, clusterScoreDetail)
                break

            dayClusters.append((day, texts_day, dataset_day, tweetClusters))

        if len(dayClusters) > 1:
            clusterFile = open(clusterFilePath, "w")
            cPickle.dump(dayClusters, clusterFile)
    elif calCluster == "0":
        clusterFile = open(clusterFilePath, "r")
        dayClusters = cPickle.load(clusterFile)


    if calCluster != "-":
        ##############
        ## evaluation and output
        evalOutputFAEvents(dayClusters, outputDays, devDays, testDays, topK_c, Kc_step, goldFA)

    print "Program ends at ", time.asctime()

