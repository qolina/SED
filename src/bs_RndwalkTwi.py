import re
import os
import sys
import math
import time
import datetime
import numpy as np

sys.path.append("/home/yxqin/FrED/srcSklcls/")
import estimatePs_smldata as estimatePs
import verifyBurstyFea

sys.path.append("/home/yxqin/Scripts/")
import readConll
import hashOperation as hashOp

from util import snpLoader
from util import posReader
from util import fileReader
from util import statisticUtil as statUtil


def getRWMatrix(unitAppHash, rwStep):
    ## co-occur matrix
    btyUnits = sorted(unitAppHash.keys())
    btyNum = len(btyUnits)
    cocountArray = np.zeros((btyNum, btyNum))
    rwMatrix = np.zeros((btyNum, btyNum))
    for i in range(btyNum):
        for j in range(btyNum):
            if i != j:
                cocountArray[i, j] = statUtil.getCooccurCount(btyUnits[i], btyUnits[j], unitAppHash)
        if sum(cocountArray[i, :]) != 0:
            for j in range(btyNum):
                rwMatrix[i, j] = cocountArray[i, j]/sum(cocountArray[i, :])

    rwMatrix = np.matrix(rwMatrix)
    rwMatrix = rwMatrix**rwStep

    #print rwMatrix[80:90,90:100]
    return rwMatrix


def dataNamesSet(day):
    #snpFilePath = "/home/yxqin/corpus/obtainSNP500/snp500_ranklist_20160801" # not ranked version: snp500_201504
    snpFilePath = "../data/snp500_ranklist_20160801" # not ranked version: snp500_201504
    cleanedTextFileName = dataFilePath+"tweetCleanText"+day
    parsedTextFileName = dataFilePath+"../nlpanalysis/tweetText"+day+".predict"
    fileArr = [snpFilePath, cleanedTextFileName, parsedTextFileName]
    return fileArr


def read(dataFilePath, btyFeaFilePath):
    day = btyFeaFilePath[-2:]
    [snpFilePath, cleanedTextFileName, parsedTextFileName] = dataNamesSet(day)
    
    [btyUnitHash, unitDFHash, unitInvolvedHash] = verifyBurstyFea.loadEvtseg(btyFeaFilePath)
    unitAppHash = estimatePs.statisticDF_fromFile(cleanedTextFileName, btyUnitHash)
    wordPOSHash = posReader.posCount_fromParsed(parsedTextFileName)
    cleanedTextHash = fileReader.loadTweets(cleanedTextFileName)

    rwStep = 5 
    rwMatrix = getRWMatrix(unitAppHash, rwStep)

    btyUnits = sorted(unitAppHash.keys())
    btyWordPOSHash = dict([(word, wordPOSHash.get(word)) for word in btyUnits])

    # sym_names: (snpSym,snpComp) [abbrv: fullname]
    sym_names = snpLoader.loadSnP500(snpFilePath)
    snpNameHash = dict([(item[0], item[1]) for item in sym_names])
    snpSym = ["$"+item[0] for item in sym_names]
    snpSymStr = "_" + "_".join(snpSym)+"_" #_$sym1_#$sym2_... 


    btySnPSym = [key for key in btyUnits if snpSymStr.find("_"+key+"_") >= 0]
    btySnPHash = dict([(key, snpNameHash.get(key[1:])) for key in btySnPSym]) # $sym:snpName
    burstNouns =  [key for key in btyUnits if (btyWordPOSHash.get(key) is not None) and (btyWordPOSHash.get(key) in ["N", "^", "Z"])] # , "CD"
    burstVerbs =  [key for key in btyUnits if (btyWordPOSHash.get(key) is not None) and btyWordPOSHash.get(key) in ["V", "M", "Y"]]

    print len(btySnPSym), len(burstVerbs), len(burstNouns)
    for sym in btySnPSym:
        print "#########################################"
        sub_id = btyUnits.index(sym)
        burstVerbId = [btyUnits.index(verb) for verb in burstVerbs]
        burstNounId = [btyUnits.index(verb) for verb in burstNouns]
        burstSymVerb = dict([(verb_id, rwMatrix[sub_id, verb_id]) for verb_id in burstVerbId])
        selected_burstSymVerb = hashOp.sortHash(burstSymVerb, 1, True)
        print sym, [btyUnits[item[0]] for item in selected_burstSymVerb[:5]]
        print sym, selected_burstSymVerb[:5]
        #print sym, max(sv_score), min(sv_score), sum(sv_score)/len(sv_score)

        for (verb_id, score) in selected_burstSymVerb:
            burstVerbNoun = dict([(noun_id, rwMatrix[verb_id, noun_id]) for noun_id in burstNounId])
            selected_burstVerbNoun = hashOp.sortHash(burstVerbNoun, 1, True)
            print sym, btyUnits[verb_id], [btyUnits[item[0]] for item in selected_burstVerbNoun[:5]]
            print sym, btyUnits[verb_id], selected_burstVerbNoun[:5]

        #print "******"
        #burstSymVerb = dict([(verb_id, rwMatrix[sub_id, verb_id]) for verb_id in range(len(btyUnits))])
        #selected_burstSymVerb = hashOp.sortHash(burstSymVerb, 1, True)
        #sv_score = [item[1] for item in selected_burstSymVerb]
        #print sv_score
        #print sym, max(sv_score), min(sv_score), sum(sv_score)/len(sv_score), sum(sv_score)

    return 0


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

    read(dataFilePath, btyFeaFilePath)
