import os
import sys
import time
import math
import cPickle
from collections import Counter

sys.path.append("../srcVec/")
from evalRecall import tClusterMatchNews_content, dayNewsExtr, outputEval
from tweetClustering import compInCluster, topCompInCluster
from tweetVec import raw2Texts

sys.path.append("../src/util/")
import snpLoader
import stringUtil as strUtil


snpFilePath = "../data/snp500_sutd"
stock_newsDir = '../ni_data/stocknews/'
newsVecPath = "../ni_data/tweetVec/stockNewsVec1_Loose1"
socialFeaPath = "../ni_data/tweetVec/tweetSocialFeature"

def wordBursty(tweetTexts_ori, seqDayHash, dayTweetNumHash, seqTidHash):
    tweetTexts_all = raw2Texts(tweetTexts_ori, True, True, 1)
    tweetTexts_all = [" ".join(words) for words in tweetTexts_all]
    
    socialFeaHash = cPickle.load(open(socialFeaPath, "r"))

    k = 13
    topK_c = 20
    topK_t = 5
    Para_newsDayWindow = [0]

    dataSelect = 3
    outCluster = "d"

    if dataSelect == 1:
        devDays = ['06', '07', '08']
        testDays = ['11', '12', '13', '14', '15']
    elif dataSelect == 2:#['18', '19', '20', '21', '22', '26', '27', '28']
        devDays = ['26', '27', '28']
        testDays = ['18', '19', '20', '21', '22']
    elif dataSelect == 3: #[15-31]
        devDays = ['15', '18', '19']
        testDays = ['20', '21', '22', '26', '27', '28']

    validDays = sorted(devDays + testDays)
    ###################
    if outCluster == "d":
        outputDays = devDays
    elif outCluster == "t":
        outputDays = testDays
    elif outCluster == "v":
        outputDays = validDays


    ##############
    sym_names = snpLoader.loadSnP500(snpFilePath)
    snp_syms = [snpItem[0] for snpItem in sym_names]
    snp_comp = [strUtil.getMainComp(snpItem[1]) for snpItem in sym_names]
    symCompHash = dict(zip(snp_syms, snp_comp))
    ##############

    ##############
    # load news vec
    newsVecFile = open(newsVecPath, "r")
    dayNews = cPickle.load(newsVecFile)
    vecNews = cPickle.load(newsVecFile)
    newsSeqDayHash = cPickle.load(newsVecFile)
    newsSeqComp = cPickle.load(newsVecFile)
    ##############

    ###################
    anchorContent = open(os.path.expanduser("~") + "/Twevent/Tools/anchorProbFile_unig", "r")
    anchorContent = [line.strip().split() for line in anchorContent if len(line.strip().split()) == 2]
    anchorProbs = dict([(item[1], float(item[0])) for item in anchorContent])
    print "## anchor prob loaded done.", len(anchorProbs)
    ###################


    ###################
    wordDict, wordIdDict, wordProb, wordDF = calWordProb(tweetTexts_all)
    print "## word tokenize done. #word", len(wordDict)

    ###################
    wordZs = calWordZs(wordProb, wordDF, seqDayHash, dayTweetNumHash)
    print "## word zscore cal done. #word", len(wordZs)
    dateArr = sorted(dayTweetNumHash.keys())

    dev_Nums = [[], [], [], []] # trueCNums, cNums, matchNNums, nNums 
    test_Nums = [[], [], [], []]

    dayClusters = []
    ###################
    for date in outputDays:
        ###################
        top_w = int(math.sqrt(dayTweetNumHash[date]))
        wordIds_date = [(wordId,zsDict[date]) for wordId, zsDict in wordZs.items() if date in zsDict]
        burstyWords = sorted(wordIds_date, key = lambda a:a[1], reverse = True)[:top_w]
        print "## bursty words done.", len(burstyWords), " in date", date

        ###################
        burstyWordTextVec, burstyWordDF = calWordTextVec(date, burstyWords, tweetTexts_all, wordDF, seqDayHash, wordIdDict, seqTidHash, socialFeaHash)
        clusters, clusterEdges, indexedIn, pairSims, sortedWordIds = knnCluster(burstyWordTextVec, burstyWordDF, k, wordDict)
        

        ###################
        burstyWordText = []
        for dfs in burstyWordDF:
            texts = []
            for df in dfs:
                for docid in df:
                    texts.append(tweetTexts_ori[docid].lower())
            burstyWordText.append(texts)
        #burstyWordText = [[tweetTexts_ori[docid].lower() for docid in df] for dfs in burstyWordDF for df in dfs]
        clusterScores = clusterScoring(clusters, clusterEdges, pairSims, sortedWordIds, wordDict, topK_c, anchorProbs)
        dayClusters.append((date, clusterScores, clusters, sortedWordIds, burstyWordText, burstyWords))


    for citem in dayClusters:
        date, clusterScores, clusters, sortedWordIds, burstyWordText, burstyWords = citem
        if date not in outputDays: continue

        print "## Output details of Clusters in day", date

        newsDayWindow = [int(date)+num for num in Para_newsDayWindow]
        vecNewsDay, textNewsDay, newsSeqCompDay = dayNewsExtr(newsDayWindow, newsSeqDayHash, vecNews, dayNews, newsSeqComp)


        trueCNum, matchNNum = evalWClusters(clusterScores, clusters, burstyWords, wordDict, burstyWordText, True, topK_t, textNewsDay, newsSeqCompDay, snp_comp, symCompHash)
        if date in devDays:
            dev_Nums[0].append(trueCNum)
            dev_Nums[1].append(len(clusterScores))
            dev_Nums[2].append(matchNNum)
            dev_Nums[3].append(len(textNewsDay))
        if date in testDays:
            test_Nums[0].append(trueCNum)
            test_Nums[1].append(len(clusterScores))
            test_Nums[2].append(matchNNum)
            test_Nums[3].append(len(textNewsDay))

    ##############
    # output evaluation metrics_recall
    if sum(dev_Nums[1]) > 0:
        print "** Dev exp in topK_c", topK_c
        outputEval(dev_Nums)
    if sum(test_Nums[1]) > 0:
        print "** Test exp in topK_c", topK_c
        outputEval(test_Nums)


def calWordProb(tweetTexts_all):
    words = set([word for text in tweetTexts_all for word in text.lower().split()])
    wordDict = dict(enumerate(sorted(list(words))))# wordId:word
    wordIdDict = dict([(word, wid) for wid, word in wordDict.items()])

    wordDF = dict([(i, set()) for i in range(len(wordIdDict))])
    for seqid, text in enumerate(tweetTexts_all):
        words = text.lower().split()
        wordIds = [wordIdDict[w] for w in words]
        for wid in wordIds:
            wordDF[wid].add(seqid)

    docNUM = len(tweetTexts_all)
    wordProb = {} # wordProb: dict (wordId, prob)
    wordProb = dict([(wid, float(len(df))/docNUM) for wid, df in wordDF.items()])
    return wordDict, wordIdDict, wordProb, wordDF

def calWordZs(wordProb, wordDF, seqDayHash, dayTweetNumHash):
    wordZs = {} # (wordId, zscores)   zscores: (date, zs)
    for wordId in wordDF:
        df = wordDF[wordId]
        df_date = [seqDayHash[docid] for docid in df]
        df_date_counter = Counter(df_date)

        zscoreDict = {}
        for date, df_day in df_date_counter.items():
            prob = wordProb[wordId]
            mu = dayTweetNumHash[date] * prob
            sigma = math.sqrt(mu*(1.0-prob))
            zscore = round((df_day*1.0-mu)/sigma, 4)
            zscoreDict[date] = zscore

        wordZs[wordId] = zscoreDict
    return wordZs

def calWordTextVec(date, burstyWords, tweetTexts_all, wordDF, seqDayHash, wordIdDict, seqTidHash, socialFeaHash):
    seqid_currDay = [docid for docid in range(len(tweetTexts_all)) if seqDayHash[docid] == date]
    #text_currDay = [tweetTexts_all[docid] for docid in seqid_currDay]
    #uniqWords_currDay = Counter(" ".join(text_currDay).lower().split())
    hour_currDay = [socialFeaHash.get(seqTidHash[docid]).get("Time") for docid in range(len(tweetTexts_all))]
    subTW_currDay = [int(hour)/2 for hour in hour_currDay]

    burstyWordDF = []
    burstyWordTextVec = []
    for wordId, zs in burstyWords:
        df_currWord = [docid for docid in wordDF[wordId] if seqDayHash[docid] == date]

        dfs = []
        vecs = []
        for stw in range(12):
            vec = []
            subTW_DF = [docid for docid in df_currWord if subTW_currDay[docid] == stw]
            if len(subTW_DF) > 1:
                sub_text = [tweetTexts_all[docid] for docid in subTW_DF]
                words_uniq = Counter(" ".join(sub_text).lower().split())
                wordIds_uniq = [(wordIdDict[word],tf) for word, tf in words_uniq.items()]
                vec = [(wid, float(tf)/len(wordDF[wid])) for wid, tf in wordIds_uniq]
            vecs.append(vec)
            dfs.append(subTW_DF)
        burstyWordDF.append(dfs)
        burstyWordTextVec.append((wordId, vecs))
    return burstyWordTextVec, burstyWordDF


# previous version without considering sub-timewindow splitted wordText
def calWordTextVec_v1(date, burstyWords, tweetTexts_all, wordDF, seqDayHash, wordIdDict):
    burstyWordDF = []
    burstyWordTextVec = []
    for wordId, zs in burstyWords:
        df_currDay = [docid for docid in wordDF[wordId] if seqDayHash[docid] == date]
        text_currDay = [tweetTexts_all[docid] for docid in df_currDay]
        burstyWordDF.append(df_currDay)
        text_currDay = " ".join(text_currDay).lower()

        words_uniq = Counter(text_currDay.split())
        wordIds_uniq = [(wordIdDict[word],tf) for word, tf in words_uniq.items()]

        vec = [(wid, float(tf)/len(wordDF[wid])) for wid, tf in wordIds_uniq]
        burstyWordTextVec.append((wordId, vec))
    return burstyWordTextVec, burstyWordDF

def cosine(source,target):
    numerator=sum([source[word]*target[word] for word in source if word in target])
    sourceLen=math.sqrt(sum([value*value for value in source.values()]))
    targetLen=math.sqrt(sum([value*value for value in target.values()]))
    denominator=sourceLen*targetLen
    if denominator==0:
        return 0
    else:
        return numerator/denominator


def knnCluster(burstyWordTextVec, burstyWordDF, k, wordDict):
    sortedWordIds = [wid for wid, vecs in burstyWordTextVec]
    sortedWords = [wordDict[wid] for wid in sortedWordIds]
    #print sorted(sortedWords)

    #print sorted([wordDict[wid] for wid, item in burstyWordTextVec])

    pairSims = []
    knn = []
    for sid, wordItem in enumerate(burstyWordTextVec):
        wid, vecs = wordItem
        dfss = burstyWordDF[sid]
        sims = []
        for si in range(sid+1, len(burstyWordTextVec)):
            wi, ves = burstyWordTextVec[si]
            dfs = burstyWordDF[si]

            ## similarity
            dfnum1 = sum([len(dfitem) for dfitem in dfss])
            dfnum2 = sum([len(dfitem) for dfitem in dfs])
            sim = 0.0
            for stw in range(12):
                vec = vecs[stw]
                ve = ves[stw]
                sub_dfnum1 = len(dfss[stw])
                sub_dfnum2 = len(dfs[stw])

                if sub_dfnum1 > 0 and sub_dfnum2 > 0:
                    sub_sim = cosine(dict(vec), dict(ve)) * (sub_dfnum1*1.0/dfnum1) * (sub_dfnum2*1.0/dfnum2)
                    sim += sub_sim

            sims.append((si, sim))
        pairSims.append(sims)

        sims_small = [(small, sim) for small in range(sid) for large, sim in pairSims[small] if large == sid]
        sims_small.extend(sims)
        topK = sorted(sims_small, key = lambda a:a[1], reverse=True)[:k]
        knn.append([item[0] for item in topK])

    if 0:
        for sid, nn in enumerate(knn):
            print sid, sortedWordIds[sid], sortedWords[sid], [(i, sortedWords[i]) for i in nn]

    indexedIn = {}
    clusters = []
    clusterEdges = {} # (clusterId, edges)  edges: (sid1, sid2)
    for sid, nn in enumerate(knn):
        for si in nn:
            if si in indexedIn: continue
            if sid not in knn[si]: continue
            # si, sid in same cluster
            if sid in indexedIn: # sid already in one cluster
                indexedIn[si] = indexedIn[sid]
                clusters[indexedIn[sid]].append(si)
                clusterEdges[indexedIn[sid]].append((sid, si))
            else: # sid, si assign a new cluster
                indexedIn[sid] = len(clusters)
                indexedIn[si] = len(clusters)
                clusters.append([sid, si])
                clusterEdges[indexedIn[sid]] = [(sid, si)]
        if sid not in indexedIn: # sid in a cluster alone
            indexedIn[sid] = len(clusters)
            clusters.append([sid])
            clusterEdges[indexedIn[sid]] = []
    
    return clusters, clusterEdges, indexedIn, pairSims, sortedWordIds

def clusterScoring(clusters, clusterEdges, pairSims, sortedWordIds, wordDict, topK_c, anchorProbs):
    sortedWords = [wordDict[wid] for wid in sortedWordIds]

    clusterScores = []
    for cid, seqWords in enumerate(clusters):
        words = [sortedWords[sid] for sid in seqWords]
        wikiProb = [anchorProbs.get(word) for word in words if word in anchorProbs]
        wikiProb = sum(wikiProb)/len(words)

        edges = clusterEdges[cid]
        edgeSims = [sim for sid1, sid2 in edges for si,sim in pairSims[min(sid1, sid2)] if si == max(sid1, sid2)]
        edgeSim = sum(edgeSims)/len(words)

        score = wikiProb*edgeSim
        clusterScores.append((cid, score))

    sortedClusters = sorted(clusterScores, key = lambda a:a[1], reverse=True)[:min(topK_c, len(clusters))]
    return sortedClusters


def evalWClusters(clusterScores, clusters, burstyWords, wordDict, burstyWordText, outputDetail, topK_t, textNewsDay, newsSeqCompDay, snp_comp, symCompHash):
    trueCluster = []
    matchedNews = []

    wordAppTweet = [Counter(burstyWordText[sid]).most_common()[0] for sid in range(len(burstyWords))]
    for outIdx, cItem in enumerate(clusterScores):
        clabel, cscore = cItem
        cwordseqIds = clusters[clabel]
        cWordZs = [(sid, burstyWords[sid][1]) for sid in cwordseqIds]
        sortedWordZsIn = sorted(cWordZs, key = lambda a:a[1], reverse=True)[:min(topK_t, len(cwordseqIds))]
        cAppTweet = Counter("\n".join(["\n".join(burstyWordText[sid]) for sid in cwordseqIds]).split("\n")).most_common()

        tNumIn = sum([num for text, num in cAppTweet])
        compsIn = compInCluster(cAppTweet, snp_comp, symCompHash, True, False)
        compsIn = topCompInCluster(compsIn, tNumIn, 2)
        matchedNews_c = tClusterMatchNews_content(newsSeqCompDay, textNewsDay, cAppTweet, compsIn)

        if outputDetail:
            print "############################"
            print "1-** cluster", outIdx, clabel, cscore, ", #words", len(cwordseqIds), ", #tweet", tNumIn, compsIn
            #for item in cAppTweet[:topK_t]:
            #    print 0, item[1], "\t", item[0]
            #print "\t".join(words)
            for sid, zs in sortedWordZsIn:
                print wordDict[burstyWords[sid][0]], wordAppTweet[sid][1], "\t", wordAppTweet[sid][0]
        if matchedNews_c is None: continue

        trueCluster.append(clabel)
        matchedNews.extend(matchedNews_c)

        if outputDetail:
            #sortedNews = sorted(simToNews_byComp, key = lambda a:a[1], reverse = True)
            sortedNews = Counter(matchedNews_c).most_common()
            for idx, sim in sortedNews:
                print '-MNews', idx, sim, textNewsDay[idx]

    matchedNews = Counter(matchedNews)
    return len(trueCluster), len(matchedNews)


