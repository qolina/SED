import re
import os
import sys
import math
import time
import cPickle
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


def dataNamesSet(day):
    snpFilePath = "/home/yxqin/corpus/obtainSNP500/snp500_ranklist_20160801" # not ranked version: snp500_201504
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
    cleanedTextHash = loadTweets(cleanedTextFileName)

    # sym_names: (snpSym,snpComp) [abbrv: fullname]
    sym_names = snpLoader.loadSnP500(snpFilePath)

    snpNameHash = dict([(item[0], item[1]) for item in sym_names])
    snpSym = ["$"+item[0] for item in sym_names]
    snpSymStr = "_".join(snpSym)+"_"

    btySnPSym = [key for key in unitAppHash if snpSymStr.find("_" + key + "_") >= 0]
    btySnPHash = dict([(key, snpNameHash.get(key[1:])) for key in btySnPSym])
    btyWordPOSHash = dict([(word, wordPOSHash.get(word)) for word in unitAppHash])

    #depLinkHash, mweHash = getDepLink_mwe(parsedTextFileName, cleanedTextFileName)

    burstNouns =  sorted([key for key in unitAppHash if (btyWordPOSHash.get(key) is not None) and (btyWordPOSHash.get(key) in ["N", "^", "Z"])]) # , "CD"
    burstVerbs =  sorted([key for key in unitAppHash if (btyWordPOSHash.get(key) is not None) and btyWordPOSHash.get(key) in ["V", "M", "Y"]])
    sym_verb_noun = [(sym, verb, noun) for sym in sorted(btySnPHash.keys()) for verb in burstVerbs for noun in burstNouns]
    print len(sym_verb_noun), sym_verb_noun[10]
    valid_sym_verb_noun = []

def featuresCal(sym, compName, units, unitAppHash, NUM, textHash, depLinkHash, mweHash):
    features = []
    keys = [(sym, item) for item in sorted(units)]

    for (sym, item) in keys:
        commonAppList = getCooccurApp(sym, item, unitAppHash)
        commonAppList_sent = [textHash.get(tid).lower() for tid in commonAppList]
        commonAppList_eleList = [sent.split(" ") for sent in commonAppList_sent]
        wordDistArr = [abs(wordArr.index(sym)-wordArr.index(item)) for wordArr in commonAppList_eleList]
        depLinkList = [depLinkHash.get(tid) for tid in commonAppList]

        #### features
        cocount = len(commonAppList)
        if cocount == 0:
            features.append(None)
            continue
        g_test_score = g_test(sym, item, unitAppHash, len(textHash))
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
#other data reading functions
def loadTweets(textFileName):
    textHash = {}
    content = open(textFileName, "r").readlines()
    content = [line[:-1].split("\t") for line in content]
    for tweet in content:
        textHash[tweet[0]] = delNuminText(tweet[-1])
    print "##End of reading file.[segged text file] total lines: ", len(content), textFileName
    return textHash

def getDepLink_mwe(parsedTextFileName, textFileName):
    sentences_conll = read_conll_file(parsedTextFileName)
    print "##End of reading file.[parsed text file for depLink_mwe]  total sents: ", len(sentences_conll), parsedTextFileName

    dep_link_list = get_dep_links(sentences_conll)
    mwes_sents_hash = get_mwes(sentences_conll)

    content = open(textFileName, "r").readlines()
    tweet_ids = [line[:-1].split("\t")[0] for line in content if len(line) > 1]

    depLinkHash = dict([(tweet_ids[i], dep_link_list[i]) for i in range(len(tweet_ids))])
    mweHash = dict([(tweet_ids[i], mwes_sents_hash[i]) for i in mwes_sents_hash])

    return depLinkHash, mweHash

### not used currently
def loadnonengtext(textfilename):
    textfile = file(textfilename)
    texthash = {} # tid:text

    lineidx = 0
    while 1:
        try:
            linestr = cpickle.load(textfile)
        except eoferror:
            print "##end of reading file.[noneng text file] total lines: ", lineidx, textfilename
            break
        linestr = linestr.strip()
        lineidx += 1

        [tweet_id, tweet_text] = linestr.split("\t")
        tweet_text = delnumintext(tweet_text)

        texthash[tweet_id] = tweet_text
    textfile.close()
    return texthash
 

##########################################################################
#math_util functions
def getCooccurCount(key1, key2, key3, unitAppHash):

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

def g_test(sym, item, unitAppHash, Tweets_day):
    appHash_s = unitAppHash.get(sym)
    appHash_u = unitAppHash.get(item)
    appHash_su = {}
    appHash_su.update(appHash_s)
    appHash_su.update(appHash_u)

    Osu = getCooccurCount(sym, item, unitAppHash)
    Os_u = len(appHash_s) - Osu
    O_su = len(appHash_u) - Osu
    O_s_u = Tweets_day - len(appHash_su)

    oArr = [Osu, Os_u, O_su, O_s_u]
    oArr = [val*1.0/Tweets_day for val in oArr]
    e = 1.0/len(oArr)

#    print oArr
    tempScore = [val*math.log(val/e) for val in oArr]
    g_test_score = sum(tempScore)
    return g_test_score


##########################################################################
#string util functions
def delNuminText(tweet_text):
    tweet_text = re.sub("\|", " ", tweet_text) # could read segged text
    words = tweet_text.split(" ")
    #words = [word for word in words if re.search("[0-9]", word) is None]
    words = [word for word in words if re.search("http", word) is None]
    tweet_text = " ".join(words)
    return tweet_text


### extract mainName out of company fullname
def getMainComp(compName):
    mainName = re.sub(r"\b(?:corporation|limited|company|group|inc|corp|ltd|svc.gp.|int'l|the|plc|cos|co)\b[.]?", "", compName)
    return mainName.strip(",&")

def comps_in_a_sent(sentence):
    comps = [1 for word in sentence.split() if word.startswith("$")]
    return comps

def sym_start(sentence, sym, compName):
    if sentence.startswith(sym):
        return 1

    firstWord_comp = compName.split(" ")[0]
    if sentence.startswith(firstWord_comp):
        return 1
    return 0

# sym: target entity (snp company: $+abbrv_name)
# word: bursty verb 
# depLinks: dependency link appeared in one sentence
def hasDepLink(sym, word, depLinks):
    sym = sym[1:] # del first letter($)
    comb_of_sym_word = [(sym, word), (word, sym)]

    company_fullname = nameHash.get(sym).split(" ")
    comb_of_full_ele_word = [(item, word) for item in company_fullname]
    comb_of_full_ele_word_inv = [(word, item) for item in company_fullname]


def contain_depLink_sent(compWords, word, depLinks, mwes):
    #### consider mwes
    if mwes is not None:
        result = [compWords.extend(mwe) for mwe in mwes]
    compWords = set(compWords)

    wordpair = [(cmpword, word) for cmpword in compWords]
    wordpair_inv = [(word, cmpword) for cmpword in compWords]
    wordpair.extend(wordpair_inv)
#    print wordpair

    if hashOp.hasSameKey_le1_inlist(depLinks, wordpair) > 0:
        return True
    return False

## similarity of sentences in commonTweets of sym and item
## how many $companies exists in sentences
## whether the sentences starts with sym
def fea_multiComp(sym, compName, commonAppList_sent):
    companies = [comps_in_a_sent(sent) for sent in commonAppList_sent]
    compNums = [len(comps) for comps in companies]
    compNum = sum(compNums)*1.0/len(compNums)
    
    sym_start_sentences = [sym_start(sent, sym, compName) for sent in commonAppList_sent]
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

    contain_depLink_list = [1 for sent_id in range(len(commonAppList)) if contain_depLink_sent(compWords, word, depLinkList[sent_id], cmpInmwe_hash.get(sent_id))]
#    print "sum(contain_depLink_list)", sum(contain_depLink_list)
    depLink_score = sum(contain_depLink_list)*1.0/len(depLinkList)
    
    return depLink_score



##########################################################################
#test/eval functions

##test the precision of mainName extraction
def mainNameEx_eval(snpNameHash):
    thrName = []
    for item in snpNameHash.values():
        mainName = getMainComp(item)
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
