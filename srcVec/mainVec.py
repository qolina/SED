from collections import Counter
from tweetVec import *
from getSim import getSim, getDF, getBursty, getBursty2 
from getSim import filtering_by_zscore, clustering, getSim_falconn
from tweetSim import testVec_byNN

def idxTimeWin(dayTweetNumHash, timeWindow):
    dayTweetNumArr = sorted(dayTweetNumHash.items(), key = lambda a:a[0])
    dayTweetNumArr = [item[1] for item in dayTweetNumArr]
    dayWindow = [None]*timeWindow
    for date in sorted(dayTweetNumHash.keys())[timeWindow:]:
        datePre = int(date) - timeWindow
        dateAft = int(date) + timeWindow
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

####################################################
if __name__ == "__main__":
    print "Program starts at ", time.asctime()

    dataDirPath = parseArgs(sys.argv)
    tweetTexts_all = None
    #tweetTexts_all, seqTidHash, seqDayHash, dayTweetNumHash = loadTweetsFromDir(dataDirPath)
    #tweetTexts_all = tweetTexts_all[:150000]

    Para_train, Para_test = ('1', '3')
    timeWindow = (-2, 1)
    #timeWindow = None

    #[None]*timeWindow.extend([(start, end), (start, end), ...])
    # int(date)-1
    #dayWindow = idxTimeWin(dayTweetNumHash, timeWindow)

    doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model"
    #l_doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model.large2013.finP"
    l_doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model.large2013"
    #l_doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model.large2016.finP"
    #l_doc2vecModelPath = "../ni_data/tweetVec/tweets.doc2vec.model.large2016"
    #largeCorpusPath = os.path.expanduser("~")+"/corpus/tweet_finance_data/tweetCleanText2016"
    largeCorpusPath = os.path.expanduser("~")+"/corpus/tweet_finance_data/tweetCleanText2013.test"
    word2vecModelPath = "../ni_data/tweetVec/w2v1010100-en"

    ##############
    # training
    trainDoc2Vec(Para_train, doc2vecModelPath, largeCorpusPath, l_doc2vecModelPath, tweetTexts_all)
    sys.exit(0)

    ##############
    # testing/using
    if Para_test in ['0', '1', '2']:
        dataset = getVec(Para_test, doc2vecModelPath, l_doc2vecModelPath, len(tweetTexts_all), word2vecModelPath, None)
    elif Para_test == '3':
        dataset = getVec(Para_test, doc2vecModelPath, l_doc2vecModelPath, len(tweetTexts_all), word2vecModelPath, tweetTexts_all)

    #dataset = dataset[:10000, :]
    #testVec_byNN(dataset, tweetTexts_all)
    
    ##############
    # get sim, cal zscore, clustering
    nnFilePath = "../ni_data/tweetVec/finance.nn"
    #nnFilePath = "../ni_data/tweetVec/finance.nn.test"
    zscoreFilePath = "../ni_data/tweetVec/finance.zscore"
    calNN = True # or False: meaning use precal nn
    calZscore = True
    thred_radius_dist = 0.5
    thred_zscore = 1.0
    topK_c, topK_t = 5, 5

    if calNN:
        #ngDistArray, ngIdxArray = getSim(dataset, thred_radius_dist)
        ngIdxArray = getSim_falconn(dataset, thred_radius_dist)
        nnFile = open(nnFilePath, "w")
        #np.savez(nnFile, dist=ngDistArray, idx=ngIdxArray)
        np.savez(nnFile, idx=ngIdxArray)
        nnFile.seek(0)
    else:
        nnFile = open(nnFilePath, "r")
        #ngDistArray, ngIdxArray = np.load(nnFile)['dist'], np.load(nnFile)['idx']
        ngIdxArray = np.load(nnFile)['idx']

    simDfDayArr = getDF(ngIdxArray, seqDayHash, timeWindow, dataset, tweetTexts_all)
    if timeWindow is None:
        zscoreDayArr = getBursty(simDfDayArr, dayTweetNumHash)
    else:
        zscoreDayArr = getBursty2(simDfDayArr, seqDayHash)

    sys.exit(0)
    dayArr = sorted(dayTweetNumHash.keys())
    for day in dayArr:
        #if day not in ["01", "02", "03"]:
        #    continue
        burstySeqIdArr = filtering_by_zscore(zscoreDayArr, seqDayHash, day, thred_zscore)
        print "## Tweet filtering by zscore done.", len(burstySeqIdArr), " out of", dayTweetNumHash[day]
        continue

        # obtain new texts, featureVectors
        texts_day = [tweetTexts_all[seqid] for seqid in burstySeqIdArr]
        dataset_day = dataset[burstySeqIdArr,:]
        print "## Begin clustering. #tweet, #vecDim", dataset_day.shape
        clustering(texts_day, dataset_day, topK_c, topK_t, burstySeqIdArr)


