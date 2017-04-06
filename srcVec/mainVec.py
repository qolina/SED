import os
import sys
import math
import time
import cPickle
from collections import Counter
from sklearn.metrics import pairwise

from tweetVec import * #loadTweetsFromDir, texts2TFIDFvecs, trainDoc2Vec, trainDoc2Vec
from getSim import getSim, getSim_falconn, getSim_sparse
from tweetNNFilt import getDF, getBursty, getBursty_tw1, getBursty_tw2, filtering_by_zscore, getBursty_byday
from tweetSim import testVec_byNN, testVec_byLSH
from tweetClustering import clustering, outputTCluster, compInDoc
from evalRecall import evalTClusters, stockNewsVec, outputEval, dayNewsExtr
from statistic import distDistribution

from word2vec import loadWord2Vec

sys.path.append("../src/util/")
import snpLoader
import stringUtil as strUtil

def idxTimeWin(dayTweetNumHash, timeWindow):
    test = False
    if dayTweetNumHash is None:
        test = True
        dayTweetNumArr = [i for i in range(1, 10)]
    else:
        dayTweetNumArr = sorted(dayTweetNumHash.items(), key = lambda a:a[0])
        dayTweetNumArr = [item[1] for item in dayTweetNumArr]
    numSum = [sum(dayTweetNumArr[:i+1]) for i in range(len(dayTweetNumArr))]
    numSumPre = [sum(dayTweetNumArr[:i]) for i in range(len(dayTweetNumArr))]
    dayWindow = []
    dayRelWindow = []
    for date in range(len(dayTweetNumArr)):
        tw1stDate = max(0, date + timeWindow[0])
        twLstDate = min(date + timeWindow[1], len(dayTweetNumArr)-1)
        start = numSumPre[tw1stDate]
        end = numSum[twLstDate]
        dayWindow.append((start, end))
        rel_start = numSumPre[date] - numSumPre[tw1stDate]
        rel_end = rel_start + dayTweetNumArr[date]
        dayRelWindow.append((rel_start, rel_end))
    if test:
        print "## Tesing index extraction in time window"
        print dayTweetNumArr
        print numSum
        print numSumPre
        print dayWindow
        print dayRelWindow
        print "## End Tesing index extraction in time window"
    return dayWindow, dayRelWindow

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



snpFilePath = "../data/snp500_sutd"
stock_newsDir = '../ni_data/stocknews/'
newsVecPath = "../ni_data/tweetVec/stockNewsVec1_Loose1"
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

fileSuf_data = ".65w"
fileSuffix_r = ".all"
fileSuffix_w = ""

nnFilePath = "../ni_data/tweetVec/finance.nn" + fileSuf_data + fileSuffix_r
dfFilePath = "../ni_data/tweetVec/finance.df" + fileSuf_data + fileSuffix_r
zscoreFilePath = "../ni_data/tweetVec/finance.zs" + fileSuf_data + fileSuffix_r + fileSuffix_w
clusterFilePath = "../ni_data/tweetVec/finance.cluster" + fileSuf_data + fileSuffix_r + fileSuffix_w


####################################################
if __name__ == "__main__":
    print "Program starts at ", time.asctime()

    dataSelect = 3
    if dataSelect == 1:
        devDays = ['06', '07', '08']
        testDays = ['11', '12', '13', '14', '15']
    elif dataSelect == 2:#['18', '19', '20', '21', '22', '26', '27', '28']
        devDays = ['26', '27', '28']
        testDays = ['18', '19', '20', '21', '22']
    elif dataSelect == 3: #[15-31]
        #devDays = [str(i).zfill(2) for i in range(1, 32)]
        #devDays = ['15', '18', '19']
        devDays = ['15']#, '16', '17']
        testDays = ['20', '21', '22', '26', '27', '28', '29']
        

    #validDays_debug = ["02", '03']
    #validDays_fst = ['06', '07', '08', '11', '12', '13', '14', '15']
    #validDays_weekend = ['09', '10', '16', '17', '23', '24', '25']
    validDays = sorted(devDays)# + testDays)
    #validDays = testDays
    outCluster = "d"

    #step = topK_c
    step = 5

    ##############
    # Parameters
    Para_NumT = 600000
    Para_train, Para_test = ('-', '3')
    #timeWindow = (int(fileSuffix_w), abs(int(fileSuffix_w)))
    #timeWindow = (int(fileSuffix_w), 0)
    timeWindow = (-30, 30)
    #timeWindow = None

    # '-': even do not need to load
    # '0': load preCalculated
    # '1': need to calculate
    calDF = '-'
    calNN = calDF
    calZs = '-'
    calCluster = "0"
    trainLSH = False
    thred_radius_dist = 0.4
    thred_zscore = 5.0
    algor = "dbscan"# algor: "kmeans", "affi", "spec", "agg"(ward-hierarchical), "dbscan"
    Para_dbscan_eps = 0.3
    Para_numClusters = -1
    topK_c, topK_t = 30, 5
    default_num_probes_lsh = 20
    Para_newsDayWindow = [0]
    Paras = [Para_NumT, Para_train, Para_test, timeWindow, calNN, calDF, calZs, calCluster, trainLSH, thred_radius_dist, thred_zscore, algor, Para_numClusters, Para_dbscan_eps, topK_c, topK_t, default_num_probes_lsh, Para_newsDayWindow]
    print "**Para setting"
    print "Para_NumT, Para_train, Para_test, timeWindow, calNN, calDF, calZs, calCluster, trainLSH, thred_radius_dist, thred_zscore, algorithm, Para_numClusters, Para_dbscan_eps, topK_c, topK_t, default_num_probes_lsh, Para_newsDayWindow"
    print Paras
    print validDays
    ##############

    ##############
    sym_names = snpLoader.loadSnP500(snpFilePath)
    snp_syms = [snpItem[0] for snpItem in sym_names]
    snp_comp = [strUtil.getMainComp(snpItem[1]) for snpItem in sym_names]
    symCompHash = dict(zip(snp_syms, snp_comp))
    #testSent = "RT @WSJ  Google searches on mobile devices now outnumber those on PCs in 10 countries including the U . S and Japan"
    #print compInDoc(testSent.lower(), snp_comp, symCompHash)
    #sys.exit(0)
    ##############

    ##############
    #w2v = loadWord2Vec(word2vecModelPath)
    #cashtag = [item for item in w2v if item[0]=="$"]
    #snpRatio = [item for item in cashtag if item[1:] in snp_syms]
    #print len(snpRatio)
    #print "\n".join(sorted(snpRatio))
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
    if 0:
        #for item in dayNews[0]:
        #    print item[1]
        print [(did, len(dayNews_day)) for did, dayNews_day in enumerate(dayNews)]
        print sum([len(dayNews_day) for dayNews_day in dayNews[1:]])
        sys.exit(0)
    ##############


    dataDirPath = parseArgs(sys.argv)
    tweetTexts_all = None
    tweetTexts_all, seqTidHash, seqDayHash, dayTweetNumHash = loadTweetsFromDir(dataDirPath)
    #sys.exit(0)
    if Para_NumT != 0 and Para_NumT < 500000:
        tweetTexts_all = tweetTexts_all[:Para_NumT]

    #[(start, end), (start, end), ...]
    # int(date)-1
    dayArr = sorted(dayTweetNumHash.keys()) # ['01', '02', ...]
    dayWindow, dayRelWindow = idxTimeWin(dayTweetNumHash, timeWindow)
    validDayWind = [] #[None]*len(dayWindow)
    for day in dayArr:
        (st, end) = dayWindow[int(day)-1]
        (rel_st, rel_end) = dayRelWindow[int(day)-1]
        if day in validDays:
            validDayWind.append((st, end))
        else:
            validDayWind.append((rel_end-rel_st,))
    #print dayWindow
    #print dayRelWindow
    #print validDayWind

    ##############
    # training
    trainDoc2Vec(Para_train, doc2vecModelPath, largeCorpusPath, l_doc2vecModelPath, tweetTexts_all)

    ##############
    # testing/using
    if calNN == "1" or calCluster == "1":
        dataset = None
        if Para_test[0] in ['0', '1', '2']:
            dataset = getVec(Para_test[0], doc2vecModelPath, l_doc2vecModelPath, len(tweetTexts_all), word2vecModelPath, None)
        if Para_test.find('3') >= 0:
            dataset_w2v = getVec('3', doc2vecModelPath, l_doc2vecModelPath, len(tweetTexts_all), word2vecModelPath, tweetTexts_all+seedTweets)
            #seedTweetVecs = getVec('3', None, None, None, word2vecModelPath, seedTweets)
            seedTweetVecs = dataset_w2v[range(-20, 0), :]
            if dataset is None:
                dataset = dataset_w2v[:-20,:]
            else:
                # concatenate d2v and w2v
                dataset = np.append(dataset, dataset_w2v, axis=1)
                #dataset = np.add(dataset, dataset_w2v)
                dataset_w2v = None # end of using

        if Para_test[0] == '4':
            #dataset = texts2TFIDFvecs(tweetTexts_all, dictPath, corpusPath)
            dataset = texts2TFIDFvecs(tweetTexts_all + seedTweets, dictPath, corpusPath)
            seedTweetVecs = dataset[range(-20, 0), :]
            dataset = dataset[:-20, :]

        dataset = dataset.astype(np.float32)
        print "## Dataset vector obtained. ", time.asctime()

    if 0:
        # output tfidf vec file by day
        for i, vdw in enumerate(validDayWind):
            if len(vdw) == 1: continue
            dataIdx_v = range(vdw[0], vdw[1])
            #tweets.extend([tweetTexts_all[docid] for docid in dataIdx_v])

            dataset_v = dataset[dataIdx_v,:]
            dataFile = open("../ni_data/tweetVec/tweets.tfidf."+str(i), "w")
            cPickle.dump(dataset_v, dataFile)
            dataFile.close()
    if 0:
        # statistic wordnum
        tweets = []
        for docid in seqDayHash:
            if seqDayHash[docid] in validDays:
                tweets.append(tweetTexts_all[docid].lower())
        wordsDict = Counter(" ".join(tweets).split())
        cashtag = [num for word,num in wordsDict.items() if word[0] == "$" and len(word) > 1 and word[1].isalpha()==True]
        print "Stat: tNum", len(tweets)
        print "Stat: wNum", len(wordsDict)
        print "Stat: cashtag", len(cashtag), sum(cashtag)*1.0/len(tweets)
        sys.exit(0)

    ##############
    # testing vec's performace by finding nearest neighbor
    if 0:
        dataset = dataset[:1000, :]
        simMatrix = pairwise.cosine_similarity(dataset)
        nns_fromSim = [sorted(enumerate(simMatrix[i]), key = lambda a:a[1], reverse=True)[:100] for i in range(simMatrix.shape[0])]
        print "## Similarity Matrix obtained at", time.asctime()
        testVec_byNN(nns_fromSim, tweetTexts_all, 10)

    if 0:
        # statistic distance distribution
        leng = 10000
        #dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
        dataset = dataset[:leng, :]
        distDistribution(dataset)
        print time.asctime()
        sys.exit(0)


    ##############

    ##############
    # get sim, cal zscore, clustering
    if calNN == '01':
        if Para_test == "4":
            ngIdxArray = getSim_sparse(dataset, thred_radius_dist, validDayWind, dayRelWindow)
            indexedInCluster = None
            clusters = None
        else:
            ngIdxArray, indexedInCluster, clusters = getSim_falconn(trainLSH, dataset, thred_radius_dist, default_num_probes_lsh, None, validDayWind, dayRelWindow)
            #testVec_byLSH(ngIdxArray, tweetTexts_all)

        nnFile = open(nnFilePath, "wb")
        cPickle.dump(ngIdxArray, nnFile)
        cPickle.dump(indexedInCluster, nnFile)
        cPickle.dump(clusters, nnFile)
    elif calNN == '00':
        nnFile = open(nnFilePath, "rb")
        ngIdxArray = cPickle.load(nnFile)
        ngIdxArray = np.asarray(ngIdxArray)
        indexedInCluster = cPickle.load(nnFile)
        clusters = cPickle.load(nnFile)
        print "## ngIdxArr loaded", nnFilePath, time.asctime(), ngIdxArray.shape,
        if indexedInCluster is not None:
            print len(indexedInCluster), len(clusters)
        print 
    #sys.exit(0)
    ##############

    ##############
    if calDF == '1':
        # cal NN, do not store NN result, takes too much space. Each day's NN can be calculated in 5 mins
        if Para_test == "4":
            ngIdxArray = getSim_sparse(dataset, thred_radius_dist, validDayWind, dayRelWindow)
            indexedInCluster = None
            clusters = None
        else:
            ngIdxArray, indexedInCluster, clusters = getSim_falconn(trainLSH, dataset, thred_radius_dist, default_num_probes_lsh, None, validDayWind, dayRelWindow)
            #testVec_byLSH(ngIdxArray, tweetTexts_all)


        simDfDayArr = getDF(ngIdxArray, seqDayHash, timeWindow, indexedInCluster, clusters)
        ngIdxArray = None # end of using
        dfFile = open(dfFilePath, "wb")
        cPickle.dump(simDfDayArr, dfFile)
    elif calDF == '0':
        dfFile = open(dfFilePath, "rb")
        simDfDayArr = cPickle.load(dfFile)
        print "## simDfDayArr loaded", dfFilePath, time.asctime()

    if calZs == '1':
        if len(simDfDayArr) < 50: # stored by day
            if 0:
                # calculate zs only within sub timewindows
                #sub_timewindow = [(-i, i) for i in range(11, 15)]
                #sub_timewindow = [(-i, i) for i in [25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]
                #sub_timewindow = [(-i, i) for i in [15, 0]]
                sub_timewindow = [(-i, 0) for i in range(1, 15)]
                for sub_tw in sub_timewindow:
                    sub_zscoreDayArr = getBursty_byday(simDfDayArr, dayTweetNumHash, sub_tw)
                    sub_zsFile = open(zscoreFilePath+str(sub_tw[0]), "wb")
                    cPickle.dump(sub_zscoreDayArr, sub_zsFile)
                    print "## zscoreDayArr stored", sub_zsFile.name, time.asctime()
            zscoreDayArr = getBursty_byday(simDfDayArr, dayTweetNumHash, timeWindow)
        else: # stored together
            if timeWindow is None:
                zscoreDayArr = getBursty(simDfDayArr, dayTweetNumHash, None, None)
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
    #sys.exit(0)

    if 0:
        #print clusters
        tc_valid = [seqdayhash[docid[0]] for docid in clusters]
        print "## tweet clusters valid distri in days", counter(tc_valid).most_common()

    if 0:
        #clusters = [item[0] for item in clusters]
        dayArr = sorted(dayTweetNumHash.keys())
        for pDate in dayArr:
            zs_pDate = [(docid, zscoreDay[0][1]) for docid, zscoreDay in enumerate(zscoreDayArr) if zscoreDay is not None if seqDayHash[docid] == pDate]
            uniq_docs = zs_pDate
            #uniq_docs = [(docid, zs) for docid, zs in zs_pDate if docid in clusters]
            print "## Statistic zs in day", pDate, "unique/all", len(uniq_docs), len(zs_pDate)
            if len(uniq_docs) < 10: continue
            #texts = [tweetTexts_all[docid] for docid, zs in uniq_docs]
            #dataset = getVec('3', None, None, None, word2vecModelPath, texts)
            sorted_zs = sorted(uniq_docs, key = lambda a:a[1], reverse=True)
            for docid, zs in sorted_zs[:50]:
                print docid, zs, sorted(simDfDayArr[docid].items(), key = lambda a:a[0]), tweetTexts_all[docid]
            for docid, zs in sorted_zs[-50:]:
                print docid, zs, sorted(simDfDayArr[docid].items(), key = lambda a:a[0]), tweetTexts_all[docid]
        sys.exit(0)
    if 0:
        for pDate in validDays:
            if pDate != "06": continue
            zs_pDate = zscoreDayArr[int(pDate)-1]
            df_pDate = simDfDayArr[int(pDate)-1]
            startNum = sum([num for dayItem, num in dayTweetNumHash.items() if int(dayItem)<int(pDate)])
            zs_pDate = [(docid, zscoreday[0][1]) for docid, zscoreday in enumerate(zs_pDate)]

            zs_distri = Counter([round(item[1], 0) for item in zs_pDate])
            print "## zs valid distri in days", sorted(zs_distri.items(), key = lambda a:a[0])

            texts = [tweetTexts_all[docid+startNum] for docid in range(len(zs_pDate))]
            textCounter = Counter(texts)
            zs_text = [texts.index(text) for text, num in textCounter.items()]
            print "## Statistic zs in day", pDate, len(zs_pDate), " #uniqText", len(textCounter)

            sorted_zs = sorted(zs_pDate, key = lambda a:a[1], reverse=True)
            for docid, zs in sorted_zs:
                if docid not in zs_text: continue
                text = texts[docid]
                print docid, textCounter[text], zs, sorted(df_pDate[docid].items(), key = lambda a:a[0]), text
            #for docid, zs in sorted_zs[-20:]:
            #    print docid, zs, sorted(df_pDate[docid].items(), key= lambda a:a[0]), tweetTexts_all[docid+startNum]


        sys.exit(0)
    ##############

    ##############
    # filtering tweets, clustering
    if calCluster == "1":

        dayClusters = []
        for day in validDays:
            #print "###########################"
            #if day != "21": continue
            if Para_test == "41":
                burstySeqIdArr = [docid for docid, dateItem in seqDayHash.items() if dateItem == day]
            else:
                burstySeqIdArr = filtering_by_zscore(zscoreDayArr, seqDayHash, day, thred_zscore)

                if burstySeqIdArr is None:continue
                if len(zscoreDayArr) < 50:
                    startNum = sum([num for dayItem, num in dayTweetNumHash.items() if int(dayItem)<int(day)])
                    burstySeqIdArr = [startNum+day_seqid for day_seqid in burstySeqIdArr]#[:500]
                    #for sid in burstySeqIdArr[:100]:
                    #    print sid
            print "## Tweet filtering by zscore done.", len(burstySeqIdArr), " out of", dayTweetNumHash[day]

            if Para_NumT != 0 and Para_NumT < 500000:
                burstySeqIdArr = [seqid for seqid in burstySeqIdArr if seqid < Para_NumT]
            if len(burstySeqIdArr) < 50:
                #print "## Too less documents current day", day, len(burstySeqIdArr)
                continue
            if 0:
                # statistic nn number/ratio in bursty tweets
                dateInt = int(day)-1
                simDf_day = simDfDayArr[dateInt]
                if simDf_day is None: continue
                nnNum_all = [sum(simDf_day[seqid-startNum].values()) for seqid in burstySeqIdArr]
                #for sub_tw in [30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
                for sub_day in range(0, 31):
                    sub_tw = (-sub_day, sub_day)
                    tw = [str(int(day)+i).zfill(2) for i in range(sub_tw[0], sub_tw[1]+1) if int(day)+i >0 and int(day)+i <= 31]

                    nnRatioStat = []
                    for seqid in burstySeqIdArr:
                        docid = seqid - startNum
                        nnDayCounter = simDf_day[docid]
                        nnDayCounter = dict([(d, nnDayCounter[d]) for d in tw])
                        #if sub_day == 30:
                        #    print sorted(nnDayCounter.items(), key = lambda a:a[0])
                        nnNum = sum(nnDayCounter.values())
                        nnRatioStat.append(nnNum)

                    stat_flag = "t"
                    #print "## Statistic Avg #nn in timewindow", sub_tw 
                    if stat_flag == "t":
                        nnNumAvg = np.mean(nnRatioStat[:min(1000, len(nnRatioStat))])
                        vtn = np.mean(nnNum_all[:1000])
                        #print "## top 1000", nnNumAvg, vtn, round(nnNumAvg*1.0/vtn, 4)
                    elif stat_flag == "l":
                        nnNumAvg = np.mean(nnRatioStat[-min(1000, len(nnRatioStat)):])
                        vtn = np.mean(nnNum_all[-1000:])
                        #print "## last 1000", nnNumAvg, vtn, round(nnNumAvg*1.0/vtn, 4)
                    elif stat_flag == "a":
                        nnNumAvg = np.mean(nnRatioStat)
                        vtn = np.mean(nnNum_all)
                        #print "## all ", nnNumAvg, vtn, round(nnNumAvg*1.0/vtn, 4)

                    print round(nnNumAvg*1.0/vtn, 4)


            #########################################################
            # obtain bursty texts, featureVectors
            texts_day = [tweetTexts_all[seqid] for seqid in burstySeqIdArr]
            #dataset_day = [dataset[seqid] for seqid in burstySeqIdArr]
            dataset_day = dataset[burstySeqIdArr,:]
            #for docid, text in enumerate(texts_day):
            #    print docid, text 
            #    print list(dataset_day[docid])

            clusterArg = None
            if algor == "dbscan":
                clusterArg = Para_dbscan_eps 
            else:
                if Para_numClusters != -1:
                    clusterArg = Para_numClusters
                else: # for kmeans: numClusters = [50, 100]  double for hierarchical
                    clusterArg = 100
                    if len(burstySeqIdArr) >= 20000:
                        clusterArg = 200
                    if algor == "kmeans":
                        clusterArg /= 2

            print "## Begin clustering in ", day, " #tweet, #vecDim", dataset_day.shape, " algorithm", algor, " clusterArg", clusterArg
            tweetClusters, clusterScoreDetail = clustering(algor, texts_day, dataset_day, clusterArg, topK_c, topK_t, burstySeqIdArr, snp_comp, symCompHash, seedTweetVecs)

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
        for sub_topK_c in range(step, topK_c+1, step):

            dev_Nums = [[], [], [], []] # trueCNums, cNums, matchNNums, nNums 
            test_Nums = [[], [], [], []]
            #for day, texts_day, dataset_day, tweetClusters in dayClusters:
            for cItem in dayClusters:
                if cItem is None: continue
                day, texts_day, dataset_day, tweetClusters = cItem
                if tweetClusters is None: continue
                if outCluster == "d":
                    outputDays = devDays
                elif outCluster == "t":
                    outputDays = testDays
                elif outCluster == "v":
                    outputDays = validDays
                if day not in outputDays: continue

                sub_tweetClusters = tweetClusters[:sub_topK_c]

                # output
                #outputTCluster(sub_tweetClusters, texts_day, clusterScoreDetail)

                newsDayWindow = [int(day)+num for num in Para_newsDayWindow]

                #textNewsDay = dayNewsTripExtr(newsDayWindow)
                #print compTrip_News

                vecNewsDay, textNewsDay, newsSeqCompDay = dayNewsExtr(newsDayWindow, newsSeqDayHash, vecNews, dayNews, newsSeqComp)

                outputDetail = False
                if sub_topK_c == topK_c:
                    outputDetail = True

                if outputDetail:
                    print "## News in day", day
                    for item in textNewsDay:
                        print item
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
            if sum(dev_Nums[1]) > 0:
                print "** Dev exp in topK_c", sub_topK_c
                outputEval(dev_Nums)
            if sum(test_Nums[1]) > 0:
                print "** Test exp in topK_c", sub_topK_c
                outputEval(test_Nums)
            ##############

    print "Program ends at ", time.asctime()
