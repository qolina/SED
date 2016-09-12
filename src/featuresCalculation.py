import re
import os
import sys
import math
import time
import datetime

from gensim import corpora, models, similarities

from util import stringUtil as strUtil
from util import statisticUtil as statUtil


def featuresCal_sv(sym, compName, item, unitAppHash, textHash, depLinkHash, mweHash):
    commonAppList = statUtil.getCooccurApp(sym, item, unitAppHash)
    commonAppList_sent = [textHash.get(tid).lower() for tid in commonAppList]
    commonAppList_eleList = [sent.split(" ") for sent in commonAppList_sent]
    wordDistArr = [abs(wordArr.index(sym)-wordArr.index(item)) for wordArr in commonAppList_eleList]
    depLinkList = [depLinkHash.get(tid) for tid in commonAppList]


    #### features
    cocount = len(commonAppList)
    if cocount == 0:
        return None
    g_test_score = statUtil.g_test(sym, item, unitAppHash, len(textHash))
    wordDist = sum(wordDistArr)*1.0/len(wordDistArr)
    depLink_sym = fea_depLink(sym, "", item, depLinkList, None, commonAppList)
    depLink_fullname = fea_depLink(sym, compName, item, depLinkList, None, commonAppList)
    depLink_mwe = fea_depLink(sym, compName, item, depLinkList, mweHash, commonAppList)

    compNum, sym_start_ratio, sim_in_docs = fea_multiComp(sym, compName, commonAppList_sent)

    #### add features into list
    featureList = [cocount, g_test_score, wordDist, depLink_sym, depLink_fullname, depLink_mwe]
    featureList.extend([compNum, sym_start_ratio, sim_in_docs])
    return featureList


def featuresCal_sn(sym, compName, item, unitAppHash, textHash, depLinkHash, mweHash):
    commonAppList = statUtil.getCooccurApp(sym, item, unitAppHash)
    commonAppList_sent = [textHash.get(tid).lower() for tid in commonAppList]
    commonAppList_eleList = [sent.split(" ") for sent in commonAppList_sent]
    wordDistArr = [abs(wordArr.index(sym)-wordArr.index(item)) for wordArr in commonAppList_eleList]
    depLinkList = [depLinkHash.get(tid) for tid in commonAppList]

    #### features
    cocount = len(commonAppList)
    if cocount == 0:
        return None
    wordDist = sum(wordDistArr)*1.0/len(wordDistArr)
    g_test_score = statUtil.g_test(sym, item, unitAppHash, len(textHash))

    compNum, sym_start_ratio, sim_in_docs = fea_multiComp(sym, compName, commonAppList_sent)
    featureList = [cocount, g_test_score, wordDist, depLink_sym, depLink_fullname, depLink_mwe]
    featureList.extend([compNum, sym_start_ratio, sim_in_docs])
    return featureList

def featuresCal_svn(sym, compName, vitem, nitem, unitAppHash, textHash, depLinkHash, mweHash):
    commonAppList = statUtil.getCooccurApp(sym, item, unitAppHash)
    commonAppList_sent = [textHash.get(tid).lower() for tid in commonAppList]
    commonAppList_eleList = [sent.split(" ") for sent in commonAppList_sent]
    wordDistArr = [abs(wordArr.index(sym)-wordArr.index(item)) for wordArr in commonAppList_eleList]
    depLinkList = [depLinkHash.get(tid) for tid in commonAppList]

    #### features
    cocount = len(commonAppList)
    if cocount == 0:
        return None
    wordDist = sum(wordDistArr)*1.0/len(wordDistArr)
    g_test_score = statUtil.g_test(sym, item, unitAppHash, len(textHash))

    compNum, sym_start_ratio, sim_in_docs = fea_multiComp(sym, compName, commonAppList_sent)
    featureList = [cocount, g_test_score, wordDist, depLink_sym, depLink_fullname, depLink_mwe]
    featureList.extend([compNum, sym_start_ratio, sim_in_docs])
    return featureList

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


