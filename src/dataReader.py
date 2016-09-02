import re
import os
import sys
import math
import time
import datetime
from gensim import corpora, models, similarities

sys.path.append("/home/yxqin/FrED/srcSklcls/")
import estimatePs_smldata as estimatePs
import verifyBurstyFea

sys.path.append("/home/yxqin/Scripts/")
import readConll
import hashOperation as hashOp

import snpLoader
import posReader
import fileReader
import stringUtil as strUtil
import statisticUtil as statUtil


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
    snpSymStr = "_".join(snpSym)+"_"

    btySnPSym = [key for key in unitAppHash if snpSymStr.find("_" + key + "_") >= 0]
    btySnPHash = dict([(key, snpNameHash.get(key[1:])) for key in btySnPSym])
    btyWordPOSHash = dict([(word, wordPOSHash.get(word)) for word in unitAppHash])

    #depLinkHash, mweHash = fileReader.getDepLink_mwe(parsedTextFileName, cleanedTextFileName)

    burstNouns =  sorted([key for key in unitAppHash if (btyWordPOSHash.get(key) is not None) and (btyWordPOSHash.get(key) in ["N", "^", "Z"])]) # , "CD"
    burstVerbs =  sorted([key for key in unitAppHash if (btyWordPOSHash.get(key) is not None) and btyWordPOSHash.get(key) in ["V", "M", "Y"]])
    sym_verb_noun = [(sym, verb, noun) for sym in sorted(btySnPHash.keys()) for verb in burstVerbs for noun in burstNouns]
    svn_countHash = dict([(item, statUtil.getCooccurCount_tri(item[0], item[1], item[2], unitAppHash)) for item in sym_verb_noun])
#    svn_countStat = hashOp.statisticHash(svn_countHash, [1, 2, 3, 4, 5])
#    print sum(svn_countStat), svn_countStat

    valid_svn = [item for item in svn_countHash if svn_countHash.get(item) >= 5]
    print "all svn:", len(sym_verb_noun), "valid:", len(valid_svn)
    for svn in sorted(valid_svn):
        print "#######################################################"
        print svn[0],"\t",svn[1],"\t",svn[2]
        commonAppList = statUtil.getCooccurApp_tri(svn[0], svn[1], svn[2], unitAppHash)
        sentHash = {}
        for tid in commonAppList:
            hashOp.cumulativeInsert(sentHash, cleanedTextHash.get(tid).lower(), 1)
        hashOp.output_sortedHash(sentHash, 1, True)



    return valid_svn

def featuresCal(sym, compName, units, unitAppHash, NUM, textHash, depLinkHash, mweHash):
    features = []
    keys = [(sym, item) for item in sorted(units)]

    for (sym, item) in keys:
        commonAppList = statUtil.getCooccurApp(sym, item, unitAppHash)
        commonAppList_sent = [textHash.get(tid).lower() for tid in commonAppList]
        commonAppList_eleList = [sent.split(" ") for sent in commonAppList_sent]
        wordDistArr = [abs(wordArr.index(sym)-wordArr.index(item)) for wordArr in commonAppList_eleList]
        depLinkList = [depLinkHash.get(tid) for tid in commonAppList]

        #### features
        cocount = len(commonAppList)
        if cocount == 0:
            features.append(None)
            continue
        g_test_score = statUtil.g_test(sym, item, unitAppHash, len(textHash))
        wordDist = sum(wordDistArr)*1.0/len(wordDistArr)
        depLink_sym = fea_depLink(sym, "", item, depLinkList, None, commonAppList)
        depLink_fullname = fea_depLink(sym, compName, item, depLinkList, None, commonAppList)
        depLink_mwe = fea_depLink(sym, compName, item, depLinkList, mweHash, commonAppList)

        compNum, sym_start_ratio, sim_in_docs = fea_multiComp(sym, compName, commonAppList_sent)

        #### add features into list
        featureList = [cocount, g_test_score, wordDist, depLink_sym, depLink_fullname, depLink_mwe]
        featureList.extend([compNum, sym_start_ratio, sim_in_docs])

        features.append(featureList)
    return keys, features


##########################################################################
#string util functions

## similarity of sentences in commonTweets of sym and item
## how many $companies exists in sentences
## whether the sentences starts with sym
def fea_multiComp(sym, compName, commonAppList_sent):
    companies = [strUtil.comps_in_a_sent(sent) for sent in commonAppList_sent]
    compNums = [len(comps) for comps in companies]
    compNum = sum(compNums)*1.0/len(compNums)
    
    sym_start_sentences = [strUtil.sym_start(sent, sym, compName) for sent in commonAppList_sent]
    sym_start_ratio = sum(sym_start_sentences)/float(len(sym_start_sentences))

    dictionary = corpora.Dictionary([sent.split() for sent in commonAppList_sent])
    corpus = [dictionary.doc2bow(sent.split()) for sent in commonAppList_sent]
    tfidf = models.TfidfModel(corpus)
    indexedSents = similarities.Similarity(None, corpus, num_features=len(dictionary), shardsize=32768)

    sim_in_docs = 0
    for sims_doc in indexedSents:
        sim_in_docs += sum(sims_doc)
    sim_in_docs /= float(len(commonAppList_sent)**2)

    return compNum, sym_start_ratio, sim_in_docs


def fea_depLink(sym, compName, word, depLinkList, mweHash, commonAppList):

    compWords = []
    ##### only sym
    compWords.append(sym)
    ##### fullname of company
    compWords.extend(compName.split(" "))
    ##### mwe considered
    cmpInmwe_hash = {}
    if mweHash is not None:
        for sent_id in range(len(commonAppList)):
            tid = commonAppList[sent_id]
            if tid in mweHash: # current sent contain mwe
                cmpInmwe_hash[sent_id] = [mwe for mwe in mweHash.get(tid) if hashOp.hasSameKey_le1_inlist(mwe, compWords) >= 1]

    contain_depLink_list = [1 for sent_id in range(len(commonAppList)) if strUtil.contain_depLink_sent(compWords, word, depLinkList[sent_id], cmpInmwe_hash.get(sent_id))]
#    print "sum(contain_depLink_list)", sum(contain_depLink_list)
    depLink_score = sum(contain_depLink_list)*1.0/len(depLinkList)
    
    return depLink_score


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

    read(dataFilePath, btyFeaFilePath)
