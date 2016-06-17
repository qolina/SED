import time
import re
import os
import sys
import cPickle
import math

sys.path.append("/home/yxqin/FrED/srcSklcls/")
from estimatePs_smldata import statisticDF
from estimatePs_smldata import statisticDF_fromFile


from verifyBurstyFea import loadEvtseg

from getSnP500 import loadSnP500
from posSeggedFile import posCount

sys.path.append("/home/yxqin/Scripts/")
from readConll import *
from hashOperation import *

def getCooccurCount(key1, key2, unitAppHash):
    if key1 == key2:
        return 0
    appHash1 = unitAppHash.get(key1)
    appHash2 = unitAppHash.get(key2)

    commonApp = [tid for tid in appHash1 if tid in appHash2]
    return len(commonApp)


def getCooccurApp(key1, key2, unitAppHash):
    if key1 == key2:
        return None 
    appHash1 = unitAppHash.get(key1)
    appHash2 = unitAppHash.get(key2)

    commonApp = [tid for tid in appHash1 if tid in appHash2]
    return commonApp

# sym: target entity (snp company: $+abbrv_name)
# word: bursty verb 
# depLinks: dependency link appeared in one sentence
def hasDepLink(sym, word, depLinks):
    sym = sym[1:] # del first letter($)
    comb_of_sym_word = [(sym, word), (word, sym)]

    company_fullname = nameHash.get(sym).split(" ")
    comb_of_full_ele_word = [(item, word) for item in company_fullname]
    comb_of_full_ele_word_inv = [(word, item) for item in company_fullname]

def rankCooccurList(sym, units, unitAppHash, NUM, textHash, depLinkHash):
    cooccurHash = {}
    cooccurHash_filterByDist = {}
    cooccurHash_filterByDepLink = {}

    for item in units:
        cocount = getCooccurCount(sym, item, unitAppHash)
        if cocount >= 5:
            cooccurHash[item] = cocount

    for word in cooccurHash:
        commonAppList = getCooccurApp(sym, word, unitAppHash)
        commonAppList_eleList = [textHash.get(tid).lower().split(" ") for tid in commonAppList]
#        for wordArr in commonAppList_eleList:
#            print wordArr, sym, word,
#            print wordArr.index(word)
        # filtering by wordDistance
        wordDist = [abs(wordArr.index(sym)-wordArr.index(word)) for wordArr in commonAppList_eleList]
        if sum(wordDist)*1.0/len(wordDist) > 5:
            cooccurHash_filterByDist[word] = 0 #10000+sum(wordDist)*1.0/len(wordDist)

        # filtering by depLink
        if depLinkHash is None:
            continue
        depLinkList = [depLinkHash.get(tid) for tid in commonAppList]
#        if word == "meeting":
#            print depLinkList
        contain_depLink = [1 for depLinks in depLinkList if (sym[1:], word) in depLinks or (word, sym[1:]) in depLinks]
        cooccurHash_filterByDepLink[word] = 10000 + sum(contain_depLink)*1.0/len(depLinkList)
        if sum(contain_depLink)*1.0/len(depLinkList) > 0.5 :
            cooccurHash_filterByDepLink[word] = 10000 + sum(contain_depLink)*1.0/len(depLinkList)

#    cooccurHash.update(cooccurHash_filterByDist)
    cooccurHash.update(cooccurHash_filterByDepLink)

    showNum = min(len(cooccurHash), NUM)
    for item in sorted(cooccurHash.items(), key = lambda a:a[1], reverse = True)[:showNum]:
        print "----------"
        print sym, "=", item[0], "\t", len(unitAppHash.get(sym)), "\t", item[1], "\t", item[1]*1.0/len(unitAppHash.get(sym))

#        commonAppList = getCooccurApp(sym, item[0], unitAppHash)
#        commonTextHash = {}
#        for tid in commonAppList:
#            text = textHash.get(tid).encode("utf-8", 'ignore')
#            cumulativeInsert(commonTextHash, text, 1)
#
#        sortedList = sortHash(commonTextHash, 1, True)
#        for item in sortedList:
#            print "--", item[1], "\t", item[0]



def statisticCooccur(btySnPSym, unitAppHash, wordPOSHash, NUM, textHash, depLinkHash):
    nonSyms =  sorted([key for key in unitAppHash if key not in btySnPSym])
    burstNouns =  sorted([key for key in unitAppHash if (wordPOSHash.get(key) is not None) and (wordPOSHash.get(key)[:2] in ["NN"])]) # , "CD"
    burstVerbs =  sorted([key for key in unitAppHash if (wordPOSHash.get(key) is not None) and (wordPOSHash.get(key).startswith("VB"))])
    units = sorted(unitAppHash.keys())
    for sym in sorted(btySnPSym):
        print "****************************************"

        print "-----------------Verbs"
        rankCooccurList(sym, burstVerbs, unitAppHash, 20, textHash, depLinkHash)
        print "-----------------Nouns"
        rankCooccurList(sym, burstNouns, unitAppHash, 50, textHash, None)

def loadNonEngText(textFileName):
    textFile = file(textFileName)
    textHash = {} # tid:text

    lineIdx = 0
    while 1:
        try:
            lineStr = cPickle.load(textFile)
        except EOFError:
            print "End of reading file. total lines: ", lineIdx, textFileName
            break
        lineStr = lineStr.strip()
        lineIdx += 1

        [tweet_id, tweet_text] = lineStr.split("\t")
        tweet_text = delNuminText(tweet_text)

        textHash[tweet_id] = tweet_text
    textFile.close()
    return textHash
        
def delNuminText(tweet_text):
    tweet_text = re.sub("\|", " ", tweet_text)
    words = tweet_text.split(" ")
    #words = [word for word in words if re.search("[0-9]", word) is None]
    words = [word for word in words if re.search("http", word) is None]
    tweet_text = " ".join(words)
    return tweet_text

def loadseggedText(textFileName):

    textHash = {}
    content = open(textFileName, "r").readlines()
    content = [line[:-1].split("\t") for line in content]
    for tweet in content:
        textHash[tweet[0]] = delNuminText(tweet[-1])
        #textHash[tweet[0]] = tweet[-1]
    print "End of reading file. total lines: ", len(content), textFileName
    return textHash

def getDepLink(parsedTextFileName, seggedTextFileName):

    sentences_conll = read_conll_file(parsedTextFileName)
    dep_link_list = get_dep_links(sentences_conll)

    content = open(seggedTextFileName, "r").readlines()
    tweet_ids = [line[:-1].split("\t")[0] for line in content if len(line) > 1]

    depLinkHash = dict([(tweet_ids[i], dep_link_list[i]) for i in range(len(tweet_ids))])

    return depLinkHash



###############################################
if __name__ == "__main__":
    print "###program starts at " + str(time.asctime())

    if len(sys.argv) == 3:
        dataFilePath = sys.argv[1]+"/"
        btySklFileName = sys.argv[2]
    else:
        print "Usage: python verifyBurstyFea.py datafilepath burstyFeafilepath"
        sys.exit(0)

    [btySklHash, unitDFHash, unitInvolvedHash] = loadEvtseg(btySklFileName)

###########
    snpSym = loadSnP500("/home/yxqin/corpus/snp500_201504")
    snpSym = ["$"+line for line in snpSym]
    snpSymStr = "_".join(snpSym)+"_"

###########
    posFilePath = "/home/yxqin/corpus/data_stock201504/segment/" + "pos_segged_tweetCleanText01"
    wordPOSHash = posCount(posFilePath)



###############################################
## statistic burst snp companies' cooccurrance with other bursty segments
    day = btySklFileName[-2:]
    unitAppHash = statisticDF_fromFile(dataFilePath+"segged_tweetCleanText"+day, btySklHash)
    btySnPSym = [key for key in unitAppHash if snpSymStr.find("_" + key.upper() + "_") >= 0]
    wordPOSHash = dict([(word, wordPOSHash.get(word)) for word in unitAppHash])

    nonEngTextFileName = "/home/yxqin/corpus/data_stock201504/nonEng/tweetText"+day
    seggedTextFileName = dataFilePath+"segged_tweetCleanText"+day
    parsedTextFileName = dataFilePath+"../nlpanalysis/tweetText"+day+".predict"

    depLinkHash = getDepLink(parsedTextFileName, seggedTextFileName)

#    nonEngTextHash = loadNonEngText(nonEngTextFileName)
#    statisticCooccur(btySnPSym, unitAppHash, wordPOSHash, 20, nonEngTextHash)

    seggedTextHash = loadseggedText(seggedTextFileName)
    statisticCooccur(btySnPSym, unitAppHash, wordPOSHash, 20, seggedTextHash, depLinkHash)

    sys.exit(0)


