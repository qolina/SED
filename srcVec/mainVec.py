import sys
import timeit
import cPickle
from collections import Counter
from sklearn.metrics import pairwise

from tweetVec import *
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
    dayWindow = [None]*abs(timeWindow[0])
    for date in sorted(dayTweetNumHash.keys())[abs(timeWindow[0]):]:
        datePre = int(date) + timeWindow[0]
        dateAft = int(date) + timeWindow[1]
        start = sum(dayTweetNumArr[:datePre-1])
        end = sum(dayTweetNumArr[:dateAft])
        dayWindow.append((start, end))
    return dayWindow


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

nnFilePath = "../ni_data/tweetVec/finance.nn.35w"
dfFilePath = "../ni_data/tweetVec/finance.df.35w.tw3"
zscoreFilePath = "../ni_data/tweetVec/finance.zs.35w.tw3"


####################################################
if __name__ == "__main__":
    print "Program starts at ", time.asctime()

    devDays = ['06', '07', '08']
    testDays = ['11', '12', '13', '14', '15']

    ##############
    # Parameters
    Para_NumT = 350000
    Para_train, Para_test = ('-', '3')
    timeWindow = (-3, 3)
    #timeWindow = None

    # '-': even do not need to load
    # '0': load preCalculated
    # '1': need to calculate
    calNN = '-'
    calDF = '-'
    calZs = '0'
    trainLSH = False
    thred_radius_dist = 0.4
    thred_zscore = 3.0
    topK_c, topK_t = 30, 5
    default_num_probes_lsh = 20
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
        ngIdxArray, indexedInCluster, clusters = getSim_falconn(dataset, thred_radius_dist, False, trained_num_probes, None)
        sys.exit(0)
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
    #sys.exit(0)
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
    dev_Nums = [[], [], [], []] # trueCNums, cNums, matchNNums, nNums 
    test_Nums = [[], [], [], []]
    dayArr = sorted(dayTweetNumHash.keys())
    for day in dayArr:
        if day not in devDays:# and day not in testDays:
        #if int(day) > 15:
            continue
        burstySeqIdArr = filtering_by_zscore(zscoreDayArr, seqDayHash, day, thred_zscore)
        print "## Tweet filtering by zscore done.", len(burstySeqIdArr), " out of", dayTweetNumHash[day]
        if len(burstySeqIdArr) < 10:
            #print "## Too less documents current day", day, len(burstySeqIdArr)
            continue

        # obtain bursty texts, featureVectors
        texts_day = [tweetTexts_all[seqid] for seqid in burstySeqIdArr]
        dataset_day = dataset[burstySeqIdArr,:]

        print "## Begin clustering in ", day, " #tweet, #vecDim", dataset_day.shape
        tweetClusters = clustering(texts_day, dataset_day, topK_c, topK_t, burstySeqIdArr, snp_comp, symCompHash)

        # output
        #outputTCluster(tweetClusters, texts_day)

        #newsDayWindow = [int(day)-1, int(day), int(day)+1]
        newsDayWindow = [int(day)]
        newsSeqIdDay = sorted([newsSeqId for newsSeqId, dayInt in newsSeqDayHash.items() if dayInt in newsDayWindow])
        vecNewsDay = vecNews[newsSeqIdDay,:]
        textNewsDay = []
        for item in newsDayWindow:
            textNewsDay.extend(dayNews[item])
        newsSeqCompDay = [newsSeqComp[newsSeqId] for newsSeqId in newsSeqIdDay]
        trueCNum, matchNNum = evalTClusters(tweetClusters, dataset_day, texts_day, vecNewsDay, textNewsDay, newsSeqCompDay, snp_comp, symCompHash)
        if day in devDays:
            dev_Nums[0].append(trueCNum)
            dev_Nums[1].append(len(tweetClusters))
            dev_Nums[2].append(matchNNum)
            dev_Nums[3].append(len(newsSeqCompDay))
        if day in testDays:
            test_Nums[0].append(trueCNum)
            test_Nums[1].append(len(tweetClusters))
            test_Nums[2].append(matchNNum)
            test_Nums[3].append(len(newsSeqCompDay))
    ##############

    ##############
    # output evaluation metrics_recall
    print "** Dev exp"
    outputEval(dev_Nums)
    #print "** Test exp"
    #outputEval(test_Nums)
    ##############

    print "Program ends at ", time.asctime()
