import re
import os
import sys
import math
import time
import cPickle
import datetime

sys.path.append("/home/yxqin/Scripts/")
import readConll
import hashOperation as hashOp

def rankCooccurList(sym, compName, units, unitAppHash, NUM, textHash, depLinkHash, mweHash):
    debug = False
    cooccurHash = {}
    cooccurHash_filterByDist = {}
    cooccurHash_filterByDepLink = {}

    for item in units:
        cocount = getCooccurCount(sym, item, unitAppHash)
        if debug:
            print sym, item, cocount
        if cocount >= 5:
            cooccurHash[item] = cocount

#    for word in cooccurHash:
#        commonAppList = getCooccurApp(sym, word, unitAppHash)
#        commonAppList_eleList = [textHash.get(tid).lower().split(" ") for tid in commonAppList]
#
#################
#        # filtering by wordDistance
#        wordDist = [abs(wordArr.index(sym)-wordArr.index(word)) for wordArr in commonAppList_eleList]
#        if sum(wordDist)*1.0/len(wordDist) > 5:
#            cooccurHash_filterByDist[word] = 0 #10000+sum(wordDist)*1.0/len(wordDist)
#
#################
#        # filtering by depLink
#        if depLinkHash is None:
#            continue
#        depLinkList = [depLinkHash.get(tid) for tid in commonAppList]
#
#        depLink_score = fea_depLink(sym, compName, word, depLinkList, mweHash, commonAppList)
#        if depLink_score < 0.5 :
#            cooccurHash_filterByDepLink[word] = 0 #10000 + sum(contain_depLink_list)*1.0/len(depLinkList)
#
##    cooccurHash.update(cooccurHash_filterByDist)
##    cooccurHash.update(cooccurHash_filterByDepLink)

    showNum = min(len(cooccurHash), NUM)
    relatedWords = []
    for item in sorted(cooccurHash.items(), key = lambda a:a[1], reverse = True)[:showNum]:
        if item[1] == 0:
            continue
        relatedWords.append(item)
        print "----------"
        print sym, "=", item[0], "\t", len(unitAppHash.get(sym)), "\t", item[1], "\t", item[1]*1.0/len(unitAppHash.get(sym))

        ## cooccur tweets
        commonAppList = getCooccurApp(sym, item[0], unitAppHash)
        commonTextHash = {}
        for tid in commonAppList:
            text = textHash.get(tid).encode("utf-8", 'ignore')
            cumulativeInsert(commonTextHash, text, 1)

        sortedList = sortHash(commonTextHash, 1, True)
        for item in sortedList:
            print "--", item[1], "\t", item[0]

    return relatedWords


def statisticCooccur(btySnPHash, unitAppHash, wordPOSHash, NUM, textHash, depLinkHash, mweHash):
    nonsyms =  sorted([key for key in unitapphash if key not in btysnphash])
    burstNouns =  sorted([key for key in unitAppHash if (wordPOSHash.get(key) is not None) and (wordPOSHash.get(key) in ["N", "^", "Z"])]) # , "CD"
    burstVerbs =  sorted([key for key in unitAppHash if (wordPOSHash.get(key) is not None) and wordPOSHash.get(key) in ["V", "M", "Y"]])
    print "--- Num of bursty verbs, nouns: ", len(burstVerbs), len(burstNouns)
    units = sorted(unitAppHash.keys())

    ## for eval
    relatedCouples_verb = []
    relatedCouples_noun = []

    for sym in sorted(btySnPHash.keys()):
        print "****************************************", sym

        print "-----------------Verbs"
        #keys, features = featuresCal(sym, btySnPHash.get(sym), burstVerbs, unitAppHash, 20, textHash, depLinkHash, mweHash)

        relatedWords = rankCooccurList(sym, btySnPHash.get(sym), burstVerbs, unitAppHash, 20, textHash, depLinkHash, mweHash)
        relatedCouples_verb.extend([(sym, wordItem) for wordItem in relatedWords])


        #print "-----------------Nouns"
        #relatedWords = rankCooccurList(sym, btySnPHash.get(sym), burstNouns, unitAppHash, 50, textHash, None, None)
    print "relatedCouple_verb Num: ", len(relatedCouples_verb)
    #print relatedCouples_verb
    return relatedCouples_verb, relatedCouples_noun

###############################################
if __name__ == "__main__":
    print "###program starts at " + str(time.asctime())

    if len(sys.argv) == 3:
        dataFilePath = sys.argv[1]+"/"
        btySklFileName = sys.argv[2]
    elif len(sys.argv) == 2:
        btySklFileName = sys.argv[1]
        dataFilePath = os.path.split(btySklFileName)[0] + "/"
    else:
        print "Usage: python burstyEntityLinking.py [datafilepath] burstyFeafilepath"
        sys.exit(0)

###############################################
## statistic burst snp companies' cooccurrance with other bursty segments

    relatedCouples_verb, relatedCouples_noun = statisticCooccur(btySnPHash, unitAppHash, wordPOSHash, 20, cleanedTextHash, depLinkHash, mweHash)
    print relatedCouples_verb


    print "###program ends at " + str(time.asctime())



