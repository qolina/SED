import re
import os
import sys
import math
import time
import cPickle
import datetime
from nltk.stem import WordNetLemmatizer

sys.path.append("/home/yxqin/Scripts/")
import hashOperation as hashOp


##########################################################################
## string util functions
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


##########################################################################
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

def getGold_manual(manualGoldPath, currDate):
    contents = open(manualGoldPath, "r").readlines()   #+str(currDate)
    trueVerbs_couples_cmp = [line.strip().split("\t")[-2] for line in contents]
    trueVerbs_couples_verb = [line.strip().split("\t")[-1] for line in contents]
    trueVerbs_couples = zip(trueVerbs_couples_cmp, trueVerbs_couples_verb)
    return trueVerbs_couples


def evalVerbs(relatedCouples, btySnPHash, manualGoldPath, newsDirPath, currDate):
    sysVerbs_couples = [(item[0], item[1][0]) for item in relatedCouples]

    #trueVerbs_couples = getGold_manual(manualGoldPath, currDate)
    trueVerbs_couples = trueByNews(relatedCouples, btySnPHash, newsDirPath, currDate)
    print trueVerbs_couples

    truePositive_verbsNum = hashOp.hasSameKey_le1_inlist(trueVerbs_couples, sysVerbs_couples)

    pre = truePositive_verbsNum*100.0/len(sysVerbs_couples)
    rec = truePositive_verbsNum*100.0/len(trueVerbs_couples)
    f1 = pre*rec/(pre+rec)
    print truePositive_verbsNum, len(sysVerbs_couples), len(trueVerbs_couples)
    print "Pre: ", pre
    print "Rec: ", rec
    print "F1 : ", f1

##########################################################################
if __name__ == "__main__":

    day = btyFeaFilePath[-2:]
    currDate = datetime.date(2015, 5, int(day))

    manualGoldPath = "/home/yxqin/sed_expData/relatedVerbs.true"
    newsDirPath = "/home/yxqin/corpus/stockNews_2015/reuters/"
    #evalVerbs(relatedCouples_verb, btySnPHash, manualGoldPath, newsDirPath, currDate)
