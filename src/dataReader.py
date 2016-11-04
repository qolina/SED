import re
import os
import sys
import math
import time
import datetime

sys.path.append("/home/yxqin/FrED/srcSklcls/")
import estimatePs_smldata as estimatePs
import verifyBurstyFea

sys.path.append("/home/yxqin/Scripts/")
import readConll
import hashOperation as hashOp

from util import snpLoader
from util import posReader
from util import fileReader
from util import stringUtil as strUtil
from util import statisticUtil as statUtil
import featuresCalculation as feaCal

def dataNamesSet(day):
    #snpFilePath = "/home/yxqin/corpus/obtainSNP500/snp500_ranklist_20160801" # not ranked version: snp500_201504
    snpFilePath = "../data/snp500_ranklist_20160801" # not ranked version: snp500_201504
    cleanedTextFileName = dataFilePath+"tweetCleanText"+day
    parsedTextFileName = dataFilePath+"../nlpanalysis/tweetText"+day+".predict"
    fileArr = [snpFilePath, cleanedTextFileName, parsedTextFileName]
    return fileArr


def read(dataFilePath, btyFeaFilePath):
    day = btyFeaFilePath[-2:]
    depLinkHash = None
    mweHash = None

    [snpFilePath, cleanedTextFileName, parsedTextFileName] = dataNamesSet(day)
    
    [btyUnitHash, unitDFHash, unitInvolvedHash] = verifyBurstyFea.loadEvtseg(btyFeaFilePath)
    unitAppHash = estimatePs.statisticDF_fromFile(cleanedTextFileName, btyUnitHash)
    wordPOSHash = posReader.posCount_fromParsed(parsedTextFileName)
    cleanedTextHash = fileReader.loadTweets(cleanedTextFileName)

    # sym_names: (snpSym,snpComp) [abbrv: fullname]
    sym_names = snpLoader.loadSnP500(snpFilePath)

    snpNameHash = dict([(item[0], item[1]) for item in sym_names])
    snpSym = ["$"+item[0] for item in sym_names]
    snpSymStr = "_" + "_".join(snpSym)+"_" #_$sym1_#$sym2_... 

    #btySnPSym = [key for key in unitAppHash if key in snpSym]
    btySnPSym = [key for key in unitAppHash if snpSymStr.find("_"+key+"_") >= 0]
    btySnPHash = dict([(key, snpNameHash.get(key[1:])) for key in btySnPSym]) # $sym:snpName
    btyWordPOSHash = dict([(word, wordPOSHash.get(word)) for word in unitAppHash])

    depLinkHash, mweHash = fileReader.getDepLink_mwe(parsedTextFileName, cleanedTextFileName)

    burstNouns =  sorted([key for key in unitAppHash if (btyWordPOSHash.get(key) is not None) and (btyWordPOSHash.get(key) in ["N", "^", "Z"])]) # , "CD"
    burstVerbs =  sorted([key for key in unitAppHash if (btyWordPOSHash.get(key) is not None) and btyWordPOSHash.get(key) in ["V", "M", "Y"]])
    sv = [(sym, verb) for sym in sorted(btySnPHash.keys()) for verb in burstVerbs]
    sn = [(sym, noun) for sym in sorted(btySnPHash.keys()) for noun in burstNouns]
    svn = [(sym, verb, noun) for (sym, verb) in sv for noun in burstNouns]
    svn_countHash = dict([(item, statUtil.getCooccurCount_tri(item[0], item[1], item[2], unitAppHash)) for item in svn])
#    svn_countStat = hashOp.statisticHash(svn_countHash, [1, 2, 3, 4, 5])
#    print sum(svn_countStat), svn_countStat

    valid_sv = [item for item in sv if statUtil.getCooccurCount(item[0], item[1], unitAppHash) >= 1]
    valid_sn = [item for item in sn if statUtil.getCooccurCount(item[0], item[1], unitAppHash) >= 1]
    valid_svn = [item for item in svn if svn_countHash.get(item) >= 1]
    print "all svn:", len(svn), "valid:", len(valid_svn)

    #######################
    # tbs problem: no svn corresponding to $aapl
#    #apple_svn = [(item, statUtil.getCooccurCount_tri(item[0], item[1], item[2], unitAppHash)) for item in svn if item[0] == "$aapl"]
#    apple_svn = [(item, statUtil.getCooccurCount(item[0], item[1], unitAppHash)) for item in svn if item[0] == "$aapl"]
#    print apple_svn
    #######################

    #######################
    # output valid_svn and its tweets
#    for svn in sorted(valid_svn):
#        print "#######################################################"
#        print svn[0],"\t",svn[1],"\t",svn[2]
#        commonAppList = statUtil.getCooccurApp_tri(svn[0], svn[1], svn[2], unitAppHash)
#        sentHash = {}
#        for tid in commonAppList:
#            hashOp.cumulativeInsert(sentHash, cleanedTextHash.get(tid).lower(), 1)
#        hashOp.output_sortedHash(sentHash, 1, True)
    #######################

    return valid_sv, valid_sn, valid_svn, btySnPHash, unitAppHash, cleanedTextHash, depLinkHash, mweHash



def featuresCal(sv, sn, svn, btySnPHash, unitAppHash, cleanedTextHash, depLinkHash, mweHash):
    features_sv = []
    features_sn = []
    features_svn = []

    for sym, vp in sv:
        compName = btySnPHash.get(sym)
        fea_sv = feaCal.featuresCal_sv(sym, compName, vp, unitAppHash, cleanedTextHash, depLinkHash, mweHash)
        features_sv.append(fea_sv)

    for sym, np in sn:
        compName = btySnPHash.get(sym)
        fea_sn = feaCal.featuresCal_sn(sym, compName, np, unitAppHash, cleanedTextHash, depLinkHash, mweHash)
        features_sn.append(fea_sn)

    for sym, vp, np in svn:
        compName = btySnPHash.get(sym)
        fea_svn = feaCal.featuresCal_svn(sym, compName, vp, np, unitAppHash, cleanedTextHash, depLinkHash, mweHash)
        features_svn.append(fea_svn)

    return features_sv, features_sn, features_svn

##########################################################################
#test/eval functions

##test the precision of mainName extraction
def mainNameEx_eval(snpNameHash):
    thrName = []
    for item in snpNameHash.values():
        mainName = strUtil.getMainComp(item)
        if len(mainName.split()) == 2:
            print item, "----", mainName, "----", mainName.split()[0]
        if len(mainName.split()) > 2:
            thrName.append(item + "----" + mainName + "----" + mainName.split()[0])
    print "\n".join(thrName)

##########################################################################
if __name__ == "__main__":

    if len(sys.argv) == 3:
        dataFilePath = sys.argv[1]+"/"
        btyFeaFilePath = sys.argv[2]
    elif len(sys.argv) == 2:
        btyFeaFilePath = sys.argv[1]
        dataFilePath = os.path.split(btyFeaFilePath)[0] + "/"
    else:
        print "Usage: python dataReader.py [dataFilePath] btyFeaFilePath"
        sys.exit(0)

    [sv, sn, svn, btySnPHash, unitAppHash, cleanedTextHash, depLinkHash, mweHash] = read(dataFilePath, btyFeaFilePath)
    #features_sv, features_sn, features_svn = featuresCal(sv, sn, svn, btySnPHash, unitAppHash, cleanedTextHash, depLinkHash, mweHash)
