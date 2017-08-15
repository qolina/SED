import os
import sys
import time
import timeit
import cPickle
from collections import Counter
from sklearn.metrics import pairwise

sys.path.append("../srcVec/")
from tweetVec import * #loadTweetsFromDir, texts2TFIDFvecs, trainDoc2Vec, trainDoc2Vec
#from getSim import getSim, getSim_falconn
#from tweetNNFilt import getDF, getBursty, getBursty_tw1, getBursty_tw2, filtering_by_zscore, getBursty_byday
#from tweetSim import testVec_byNN, testVec_byLSH
#from tweetClustering import clustering, outputTCluster, compInDoc
#from evalRecall import evalTClusters, stockNewsVec, outputEval
#
#from word2vec import loadWord2Vec
#
#sys.path.append("../src/util/")
#import snpLoader
#import stringUtil as strUtil

from wordBursty import wordBursty

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
    tweetTexts_all, seqTidHash, seqDayHash, dayTweetNumHash = loadTweetsFromDir(dataDirPath)

    if 0:
        socialFeaPath = "../ni_data/timeCorrect/tweetSocialFeature01"
        socialFeaHash1 = cPickle.load(open(socialFeaPath, "r"))
        socialFeaPath = "../ni_data/timeCorrect/tweetSocialFeature02"
        socialFeaHash2 = cPickle.load(open(socialFeaPath, "r"))

        socialFeaHash1 = dict([(tid, socialFeaHash1[tid]) for tid in seqTidHash.values() if tid in socialFeaHash1])
        socialFeaHash2 = dict([(tid, socialFeaHash2[tid]) for tid in seqTidHash.values() if tid in socialFeaHash2])

        socialFeaHash1.update(socialFeaHash2)
        outputFile = open("../ni_data/tweetVec/tweetSocialFeature", "w")
        cPickle.dump(socialFeaHash1, outputFile)
        outputFile.close()
        sys.exit(0)

    texts = raw2Texts(tweetTexts_all, True, True, 1)
    texts = [" ".join(words) for words in texts]
    wordBursty(tweetTexts_all, seqDayHash, dayTweetNumHash, seqTidHash)

    print "Program ends at ", time.asctime()
