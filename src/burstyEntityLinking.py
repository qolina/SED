import time
import re
import os
import sys
import cPickle
import math
import datetime
from gensim import corpora, models, similarities
from nltk.stem import WordNetLemmatizer

sys.path.append("/home/yxqin/FrED/srcSklcls/")
from estimatePs_smldata import statisticDF
from estimatePs_smldata import statisticDF_fromFile


from verifyBurstyFea import loadEvtseg

from getSnP500 import loadSnP500
from posSeggedFile import posCount
from posSeggedFile import posCount_fromParsed

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


def contain_depLink_sent(compWords, word, depLinks, mwes):
    #### consider mwes
    if mwes is not None:
        result = [compWords.extend(mwe) for mwe in mwes]
    compWords = set(compWords)

    wordpair = [(cmpword, word) for cmpword in compWords]
    wordpair_inv = [(word, cmpword) for cmpword in compWords]
    wordpair.extend(wordpair_inv)
#    print wordpair

    if hasSameKey_le1_inlist(depLinks, wordpair) > 0:
        return True
    return False

def loadNonEngText(textFileName):
    textFile = file(textFileName)
    textHash = {} # tid:text

    lineIdx = 0
    while 1:
        try:
            lineStr = cPickle.load(textFile)
        except EOFError:
            print "##End of reading file.[nonEng text file] total lines: ", lineIdx, textFileName
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
                cmpInmwe_hash[sent_id] = [mwe for mwe in mweHash.get(tid) if hasSameKey_le1_inlist(mwe, compWords) >= 1]

    contain_depLink_list = [1 for sent_id in range(len(commonAppList)) if contain_depLink_sent(compWords, word, depLinkList[sent_id], cmpInmwe_hash.get(sent_id))]
#    print "sum(contain_depLink_list)", sum(contain_depLink_list)
    depLink_score = sum(contain_depLink_list)*1.0/len(depLinkList)
    
    return depLink_score


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

def getGold_manual(manualGoldPath, currDate):
    contents = open(manualGoldPath, "r").readlines()   #+str(currDate)
    trueVerbs_couples_cmp = [line.strip().split("\t")[-2] for line in contents]
    trueVerbs_couples_verb = [line.strip().split("\t")[-1] for line in contents]
    trueVerbs_couples = zip(trueVerbs_couples_cmp, trueVerbs_couples_verb)
    return trueVerbs_couples

def getLemma(word):
    wordnet_lemmatizer = WordNetLemmatizer()
    try:
        word = wordnet_lemmatizer.lemmatize(word).encode("utf-8")
    except Exception:
        return word
    return word

def docCleanning(contentArr):
    docStr = "".join(contentArr).lower()
    docStr = re.sub("\n", " ", docStr)
    docStr = re.sub("\s+", " ", docStr)
    words = [getLemma(word) for word in docStr.strip().split()]
    return " ".join(words)

def getNews(newsDirPath, egDate):
    dirPath = newsDirPath + str(egDate) + "/"
    newsArr = [open(dirPath+nameItem, "r").readlines() for nameItem in os.listdir(dirPath)]
    newsArr = [docCleanning(newsDoc) for newsDoc in newsArr]

    return newsArr

def getSim_couple_news(couple, newsArr, newsHeadline):
    (comp, verb) = couple
    mainName = getMainComp(comp)
    mainName = mainName.split()[0] # first meaningfull word in compName

    compContain_news = [news for news in newsArr if mainName in news.split()]
    #compContain_newsHeadline = [newsHeadline.index(headline) for headline in newsHeadline if mainName in headline.split()]
    compContain_newsHeadline = [newsHeadline.index(headline) for headline in newsHeadline if mainName in headline]
    print "****************************************"
    print "--", comp, "\t", verb, "\t#", mainName, len(compContain_news), len(compContain_newsHeadline)
    if len(compContain_newsHeadline) == 0:
        return compContain_newsHeadline
    print "\n".join([newsHeadline[idx] for idx in compContain_newsHeadline])

    #verbContain_news = [news for news in compContain_news if verb in news]
    verbContain_news = [newsArr[head_idx] for head_idx in compContain_newsHeadline if verb in newsArr[head_idx].split()]
    #return len(verbContain_news)
    return verbContain_news


def trueByNews(relatedCouples, btySnPHash, newsDirPath, currDate):

    newsArr = getNews(newsDirPath, currDate - datetime.timedelta(1))
    newsArr.extend(getNews(newsDirPath, currDate))
    newsHeadline = [news[:news.find(" -- ")].strip("--") for news in newsArr]
    print "##End of loading stock news.", len(newsArr)
    #print newsArr[0]

    #trueCouples = [(sym, verb) for (sym, verb) in relatedCouples if len(getSim_couple_news((btySnPHash.get(sym), wordnet_lemmatizer.lemmatize(verb[0])), newsArr, newsHeadline)) > 0]
    trueCouples = [ ]
    for (sym, verb) in relatedCouples:
        sim = getSim_couple_news((btySnPHash.get(sym), getLemma(verb[0])), newsArr, newsHeadline)
        if len(sim) > 0:
            trueCouples.append((sym, verb[0]))
            print "--- news couple:", sym, "\t", verb[0], len(sim)
            print "\n".join(sim[:min(len(sim), 10)])
    return trueCouples

def evalVerbs(relatedCouples, btySnPHash, manualGoldPath, newsDirPath, currDate):
    sysVerbs_couples = [(item[0], item[1][0]) for item in relatedCouples]

    #trueVerbs_couples = getGold_manual(manualGoldPath, currDate)
    trueVerbs_couples = trueByNews(relatedCouples, btySnPHash, newsDirPath, currDate)
    print trueVerbs_couples

    truePositive_verbsNum = hasSameKey_le1_inlist(trueVerbs_couples, sysVerbs_couples)

    pre = truePositive_verbsNum*100.0/len(sysVerbs_couples)
    rec = truePositive_verbsNum*100.0/len(trueVerbs_couples)
    f1 = pre*rec/(pre+rec)
    print truePositive_verbsNum, len(sysVerbs_couples), len(trueVerbs_couples)
    print "Pre: ", pre
    print "Rec: ", rec
    print "F1 : ", f1


def statisticCooccur(btySnPHash, unitAppHash, wordPOSHash, NUM, textHash, depLinkHash, mweHash):
    nonSyms =  sorted([key for key in unitAppHash if key not in btySnPHash])
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

def getMainComp(compName):
    #mainName = re.sub(r"(?:(.))\b(?:corporation|limited|company|group|inc|corp|ltd|svc.gp.|int'l|the|plc|cos|co)\b[.]?", "", compName)
    mainName = re.sub(r"\b(?:corporation|limited|company|group|inc|corp|ltd|svc.gp.|int'l|the|plc|cos|co)\b[.]?", "", compName)
    return mainName.strip(",&")

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

    [btySklHash, unitDFHash, unitInvolvedHash] = loadEvtseg(btySklFileName)

    day = btySklFileName[-2:]
    currDate = datetime.date(2015, 5, int(day))

    nonEngTextFileName = "/home/yxqin/corpus/data_stock201504/nonEng/tweetText"+day
    seggedTextFileName = dataFilePath+"segged_tweetCleanText"+day
    cleanedTextFileName = dataFilePath+"tweetCleanText"+day
    parsedTextFileName = dataFilePath+"../nlpanalysis/tweetText"+day+".predict"
#    posFilePath = "/home/yxqin/corpus/data_stock201504/segment/" + "pos_segged_tweetCleanText01"

###########
    # sym_names: (snpSym,snpComp) [abbrv: fullname]
    sym_names = loadSnP500("/home/yxqin/corpus/obtainSNP500/snp500_ranklist_20160801")
    snpNameHash = dict([(item[0], item[1]) for item in sym_names])
    snpSym = ["$"+item[0] for item in sym_names]
    snpSymStr = "_".join(snpSym)+"_"

####### test the precision of mainName extraction
#    thrName = []
#    for item in snpNameHash.values():
#        mainName = getMainComp(item)
#        if len(mainName.split()) == 2:
#            print item, "----", mainName, "----", mainName.split()[0]
#        if len(mainName.split()) > 2:
#            thrName.append(item + "----" + mainName + "----" + mainName.split()[0])
#    print "\n".join(thrName)
#    sys.exit(0)

###############################################
## statistic burst snp companies' cooccurrance with other bursty segments
    unitAppHash = statisticDF_fromFile(cleanedTextFileName, btySklHash)

    #wordPOSHash = posCount(posFilePath)
    wordPOSHash = posCount_fromParsed(parsedTextFileName)
    depLinkHash, mweHash = getDepLink_mwe(parsedTextFileName, cleanedTextFileName)

    btySnPSym = [key for key in unitAppHash if snpSymStr.find("_" + key + "_") >= 0]
    btySnPHash = dict([(key, snpNameHash.get(key[1:])) for key in btySnPSym])
    wordPOSHash = dict([(word, wordPOSHash.get(word)) for word in unitAppHash])

#    nonEngTextHash = loadNonEngText(nonEngTextFileName)
#    statisticCooccur(btySnPHash, unitAppHash, wordPOSHash, 20, nonEngTextHash)

#    seggedTextHash = loadseggedText(seggedTextFileName)
#    statisticCooccur(btySnPHash, unitAppHash, wordPOSHash, 20, seggedTextHash, depLinkHash, mweHash)

    cleanedTextHash = loadseggedText(cleanedTextFileName)
    relatedCouples_verb, relatedCouples_noun = statisticCooccur(btySnPHash, unitAppHash, wordPOSHash, 20, cleanedTextHash, depLinkHash, mweHash)
    print relatedCouples_verb

    manualGoldPath = "/home/yxqin/sed_expData/relatedVerbs.true"
    newsDirPath = "/home/yxqin/corpus/stockNews_2015/reuters/"
    #evalVerbs(relatedCouples_verb, btySnPHash, manualGoldPath, newsDirPath, currDate)

    print "###program ends at " + str(time.asctime())
