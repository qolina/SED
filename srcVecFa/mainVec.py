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
from evalRecall import evalOutputFAEvents, loadGold
from statistic import distDistribution, idxTimeWin, getValidDayWind, statGoldNews, zsDistribution
from statistic import output_zsDistri_day, stat_nn_performance, stat_wordNum

from word2vec import loadWord2Vec

sys.path.append("../src/util/")
import snpLoader
import stringUtil as strUtil

#goldFA = [("28", "[kick kicked] [0-0 0 0];start;off"),
#("30", "[ramires chelsea];[goal 1-0 1 0] score;yes"),
#("33", "[salomon kalou];run;box mazy"),
#("35", "mikel;[yellow card booking] gerrard;foul;agger"),
#("37", "[second half 2nd];[kick kicked] off"),
#("40", "agger;[booked yellow card] tackle;mikel;challenge"),
#("41", "[goal 2-0 2 0];[didier drogba chelsea] score"),
#("42", "[frank lampard];[pass assist] ball;drogba"),
#("37", "[andy carroll liverpool];[goal 2-1 2 1] score"),
#("44", "[first half ht];[half time ht] [1-0 1 0]"),
#("46", "[luis suarez];save;cech shot;forces"),
#("48", "[andy carroll];line header;cech;over;claim;equalise"),
#("50", "[full time final whistle gone] chelsea;champions;congratulations;[2-1 2 1];win")]

#goldFA = [
#("137", "[kick kicked]  [0-0 0 0];start;off "),
#("147", "[ramires chelsea];[goal 1-0 1 0]   score;yes   "),
#("162", "[salomon kalou];run;box    mazy     "),
#("174", "mikel;[yellow card booking]    gerrard;foul;agger  "),
#("199", "[second half 2nd];[kick kicked]    off "),
#("182", "agger;[booked yellow card] tackle;mikel;challenge"),
#("205", "[goal 2-0 2 0];[didier drogba chelsea] score   "),
#("206", "[frank lampard];[pass assist]  ball;drogba "),
#("217", "[andy carroll liverpool];[goal 2-1 2 1]    score"),
#("184", "[first half ht];[half time ht] [1-0 1 0]   "),
#("227", "[luis suarez];save;cech    shot;forces "),
#("237", "[andy carroll];line    header;cech;over;claim;equalise"),
#("250", "[full time final whistle gone] chelsea;champions;congratulations;[2-1 2 1];win ")]

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
    arg_calZs = getArg(args, "-zs") #'0', '1' or None
    arg_calCluster = getArg(args, "-cls") #'0', '1' or None

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
        print "[-zs] [-cls] calculate/load/ignore zscore|clustering"
        print "[-zsfs] [-clsfs] file suffix of zscore|clustering"
        print "[-trainvec] -testvec     tweet vector"
        print "[-ltw] 30 [-rtw] 0"
        print "[-tf]    choose to use bursty tweet filtering"
        print "[-delta]     threshold of bursty zscore"
        print "[-cluster]   clustering algorithm"
        print "[-kc] [-kt] [-kcs] topK_c, topK_t, Kc_step"
        sys.exit(0)

    return arg1, arg_calZs, arg_calCluster, arg_zsFileSuff, arg_clsFileSuff, arg_trainVec, arg_testVec, arg_leftTW, arg_rightTW, arg_validDaysFlag, arg_outputDaysFlag, arg_tfBurstyFlag, arg_zsDelta, arg_clusterAlgor, arg_Kc, arg_KcStep, arg_Kt



word2vecModelPath = "../ni_data/tweetVec/w2v1010100-en"
dictPath = "../ni_data/tweetVec/tweets.dict"
corpusPath = "../ni_data/tweetVec/tweets.mm"

####################################################
if __name__ == "__main__":
    print "Program starts at ", time.asctime()

    (dataDirPath, arg_calZs, arg_calCluster, arg_zsFileSuff, arg_clsFileSuff, arg_trainVec, arg_testVec, arg_leftTW, arg_rightTW, arg_validDaysFlag, arg_outputDaysFlag, arg_tfBurstyFlag, arg_zsDelta, arg_clusterAlgor, arg_Kc, arg_KcStep, arg_Kt) = parseArgs(sys.argv)

    ##############
    # Parameters
    Para_train, Para_test = (arg_trainVec, arg_testVec)
    validDaysFlag, outputDaysFlag = (arg_validDaysFlag, arg_outputDaysFlag)
    (calZs, calCluster) = (arg_calZs, arg_calCluster)
    (zsFileSuff, clsFileSuff) = arg_zsFileSuff, arg_clsFileSuff
    burstyFlag = arg_tfBurstyFlag
    algor = arg_clusterAlgor
    (thred_zscore, topK_c, Kc_step, topK_t) = (arg_zsDelta, arg_Kc, arg_KcStep, arg_Kt)

    if arg_leftTW is None or arg_rightTW is None: timeWindow = None
    else: timeWindow = (-int(arg_leftTW), int(arg_rightTW))
    
    if arg_trainVec is None: Para_train = '-'
    if arg_testVec is None: Para_test = '-'
    if arg_calZs is None: calZs = '-'
    if arg_calCluster is None: calCluster = "-"
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
    thred_radius_dist = 0.5
    default_num_probes_lsh = 20
    Para_dbscan_eps = 0.5 #default 0.2
    Para_numClusters = -1

    Paras = [calZs, calCluster, Para_train, Para_test, default_num_probes_lsh, thred_radius_dist, thred_zscore, algor, Para_numClusters, Para_dbscan_eps, topK_c, Kc_step, topK_t, timeWindow]
    print "**Para setting"
    print sys.argv
    print "calZs, calCluster, Para_train, Para_test, default_num_probes_lsh, thred_radius_dist, thred_zscore, algor, Para_numClusters, Para_dbscan_eps, topK_c, Kc_step, topK_t, timeWindow"
    print Paras
    ##############


    ######################
    validDays = None
    ## FA cup
    #devDays = ["137", "147", "162", "174"]
    #testDays = ["199", "182", "205", "206", "217", "184", "227", "237", "250"]
    #["16_16", "16_26", "16_41", "16_53"]["17_18", "17_01", "17_24", "17_25", "17_36", "17_03", "17_46", "17_56", "18_09"]
    # US election
    devDays = ["043", "046", "047", "049", "051", "053", "055", "056", "057", "058"]
    testDays = ["059", "060", "061", "064", "066", "067", "068", "071", "072", "073", "075", "076", "077", "078", "082", "083"]
    #["043", "046", "047", "049", "051", "053", "055", "056", "057", "058", "059", "060", "061", "064", "066", "067", "068", "071", "072", "073", "075", "076", "077", "078", "082", "083"]

    if validDays is None:
        if validDaysFlag == 'd': validDays = devDays
        elif validDaysFlag == 't': validDays = testDays
        elif validDaysFlag == 'a': validDays = sorted(devDays + testDays)

    if outputDaysFlag is None or outputDaysFlag == "d": outputDays = devDays
    elif outputDaysFlag == "t": outputDays = testDays
    elif outputDaysFlag == "v": outputDays = validDays

    fileSuf_data = ""# + str(dataSelect)
    zscoreFilePath = "../ni_data/tweetVecUse/use.zs" + fileSuf_data + zsFileSuff
    clusterFilePath = "../ni_data/tweetVecUse/use.cluster" + fileSuf_data + clsFileSuff

    ##############
    print "validDays", validDays
    print "outputDays", outputDays

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
        #tweetTexts_all = tweetTexts_all[:300000]
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

    ##############

    ########### Stat
    if 0:
        stat_tweets = []
        #for day in devDays:
        #for day in testDays:
        for day in dayArr:
            tweetFCSeqIdArr = [docid for docid, dateItem in seqDayHash.items() if dateItem == day]
            texts_day = [tweetTexts_all[seqid] for seqid in tweetFCSeqIdArr]
            stat_tweets.extend(texts_day)

        stat_words = Counter(" ".join(stat_tweets).split())
        print "In total", len(stat_tweets), len(stat_words)
        sys.exit(0)
    ########### Stat end
    
    
    
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
        if len(dayClusters) > 0:
            clusterFile = open(clusterFilePath, "w")
            cPickle.dump(dayClusters, clusterFile)
            print "## Clustering results stored.", clusterFilePath, time.asctime()
    elif calCluster == "0":
        clusterFile = open(clusterFilePath, "r")
        dayClusters = cPickle.load(clusterFile)
        print "## Clustering results obtained.", clusterFilePath, time.asctime()

    if calZs == "1":
        dayOutClusters = []
        for dayClusterItem in dayClusters:
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

            clusterFeatures = getClusterFeatures(tweetClusters, texts_day, dataset_day, seedTweetVecs, None, None)
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
        print "## Clustering results stored.", clusterFilePath+zsFileSuff, time.asctime()
    else:
        clusterOutputFile = open(clusterFilePath+zsFileSuff, "r")
        dayOutClusters = cPickle.load(clusterOutputFile)
        print "## Clustering results obtained.", clusterFilePath+zsFileSuff, time.asctime()

    ##############
    ## evaluation and output
    ## FAcup
    if 0:
        goldFA = loadGold("../ni_data/fat/FACup_ground_truth_topics/")
        FaTimeWindowMap = dict([("16_16", "137"), ("16_26", "147"), ("16_41", "162"), ("16_53", "174"), ("17_18", "199"), ("17_1", "182") , ("17_24", "205"), ("17_25", "206"), ("17_36", "217"), ("17_3" , "184"), ("17_46", "227"), ("17_56", "237"), ("18_9", "250")])
        goldFA = [(FaTimeWindowMap[tw], goldFA[tw]) for tw in goldFA]
        print goldFA
        evalOutputFAEvents(dayOutClusters, outputDays, devDays, testDays, topK_c, Kc_step, goldFA)

    ## US election
    if 1:
        goldUse = loadGold("../ni_data/use/USElections_ground_truth_topics/")
        UseTimeWindowMap = dict([("0_0", "043"), ("0_30", "046"), ("0_40", "047"), ("1_0", "049"), ("1_20", "051"), ("1_40", "053"), ("2_0", "055"), ("2_10", "056"), ("2_20", "057"), ("2_30", "058"), ("2_40", "059"), ("2_50", "060"), ("3_0", "061"), ("3_30", "064"), ("3_50", "066"), ("4_0", "067"), ("4_10", "068"), ("4_40", "071"), ("4_50", "072"), ("5_0", "073"), ("5_20", "075"), ("5_30", "076"), ("5_40", "077"), ("5_50", "078"), ("6_30", "082"), ("6_40", "083")])
        goldUse = [(UseTimeWindowMap[tw], goldUse[tw]) for tw in goldUse]
        evalOutputFAEvents(dayOutClusters, outputDays, devDays, testDays, topK_c, Kc_step, goldUse)

    print "Program ends at ", time.asctime()
