import os
import sys
import time
import timeit
import math
from collections import Counter

import numpy as np

#######################
# statistic tweetSimDf by day
# tweetSimDfDayArr: [nnDay_Counter_seqid0, seq1, ...]
# nnDay_Counter_seqid: (day, tweet_nn_num)
def getDF(ngIdxArray, seqDayHash, timeWindow, dataset, tweetTexts_all, indexedInCluster, clusters):
    tweetSimDfDayArr = []
    for docid, nnIdxs in enumerate(ngIdxArray):
        if nnIdxs is None: 
            sameDocId = clusters[indexedInCluster[docid]][0]
            nnIdxs = ngIdxArray[sameDocId]
            #print docid, sameDocId, len(nnIdxs)
            #continue
        nnDays = [seqDayHash.get(seqid) for seqid in nnIdxs]
        if timeWindow is not None:
            date = int(seqDayHash.get(docid))
            if (date > 0-timeWindow[0]) and (date <= 31-timeWindow[1]):
                date_inTimeWin = [str(item).zfill(2) for item in range(date+timeWindow[0], date+timeWindow[1]+1)]
                nnDays = [item for item in nnDays if item in date_inTimeWin]
            else:
                nnDays = None
        if nnDays is not None:
            nnDay_count = Counter(nnDays)
        tweetSimDfDayArr.append(nnDay_count)
    print "## Tweets simDF by day obtained at", time.asctime()
    return tweetSimDfDayArr

# zscoreDayArr = [zscore_seqid0, seq1, ...]
def getBursty_tw1(simDfDayArr, seqDayHash):
    zscoreDayArr = []
    statisticDfsNegDiff = [Counter() for i in range(32)]
    statisticDf = [Counter() for i in range(32)]
    for docid, nnDayCounter in enumerate(simDfDayArr):
        if nnDayCounter is None or len(nnDayCounter) < 3:
            zscoreDayArr.append(None)
            continue
        date = seqDayHash.get(docid)
        dfs = np.asarray(nnDayCounter.values(), dtype=np.float32)
        df_currentDay = nnDayCounter[date]
        mu = np.mean(dfs)
        sigma = np.std(dfs)
        zscore = 0.0
        if df_currentDay != mu:
            zscore = round((df_currentDay-mu)/sigma, 4)

        dfs_sorted = sorted(nnDayCounter.items(), key = lambda a:a[0])
        dfs_diff = [(int(day)-int(date), df-df_currentDay) for day, df in dfs_sorted]
        for window, diff in dfs_diff:
            if diff < 0:
                statisticDfsNegDiff[int(date)][window] += 1
            statisticDf[int(date)][window] += 1

        if docid in range(80030, 80090) or docid == 46909: #docid in range(100000, 100100) or
            print "###############"
            print "## doc", docid, "\t date", date, "\t df_day", 
            print df_currentDay, "\t mu", mu, "\t sigma", sigma, "\t zscore", zscore
            print dfs_sorted
            print dfs_diff

        zscoreDayArr.append([(date, zscore)])
    print "## Tweets zscore in time window obtained at", time.asctime()
    for i in range(1, 32):
        dfDiffInWin = sorted(statisticDf[i].items(), key = lambda a:a[0])
        dfNegDiffInWin = sorted(statisticDfsNegDiff[i].items(), key = lambda a:a[0])
        if len(dfDiffInWin) < 1:
            continue
        ratio = [round(num*1.0/statisticDf[i][day], 4) for day, num in dfNegDiffInWin]
        print "************Date", i
        print dfDiffInWin
        print dfNegDiffInWin
        print ratio
    return zscoreDayArr

def getBursty_tw2(simDfDayArr, seqDayHash, dayTweetNumHash):
    zscoreDayArr = []
    for docid, nnDayCounter in enumerate(simDfDayArr):
        if nnDayCounter is None or len(nnDayCounter) < 3:
            zscoreDayArr.append(None)
            continue
        TweetNum_all_tw = sum([dayTweetNumHash[day] for day in dayTweetNumHash if day in nnDayCounter])
        docSimDF_all = sum(nnDayCounter.values())
        est_prob = docSimDF_all*1.0/TweetNum_all_tw
        zscoreDay = []
        date = seqDayHash.get(docid)
        df_currentDay = nnDayCounter[date]
        TweetNum_day = dayTweetNumHash[date]
        if df_currentDay < 1:
            zscore = -99.0
        else:
            mu = est_prob * TweetNum_day
            sigma = math.sqrt(mu*(1-est_prob))
            zscore = round((df_currentDay*1.0-mu)/sigma, 4)
        #print docid, date, df_currentDay, mu, est_prob, sigma, zscore
        zscoreDay.append((date, zscore))

        #if docid in range(50, 70) or docid in range(150, 170):
        #if docid in range(100000, 100050):
        #    print "#################################"
        #    print nnDayCounter.most_common()
        #    print sorted(zscoreDay, key = lambda a:a[1], reverse=True)
        zscoreDayArr.append(zscoreDay)
    print "## Tweets zscore by day obtained at", time.asctime()
    return zscoreDayArr


# zscoreDayArr: [zscoreDay_seqid0, seq1, ...]
# zscoreDay_seqid: [(day, zscore), (day, zscore)]
def getBursty(simDfDayArr, dayTweetNumHash):
    TweetNum_all = sum(dayTweetNumHash.values())
    zscoreDayArr = []
    for docid, nnDayCounter in enumerate(simDfDayArr):
        docSimDF_all = sum(nnDayCounter.values())
        est_prob = docSimDF_all*1.0/TweetNum_all
        zscoreDay = []
        zscoreTest = []
        for day, simDf in nnDayCounter.items():
            if simDf < 1:
                continue
            TweetNum_day = dayTweetNumHash[day]
            mu = est_prob * TweetNum_day
            sigma = math.sqrt(mu*(1-est_prob))
            #print docid, day, simDf, mu, est_prob, sigma
            zscore = round((simDf*1.0-mu)/sigma, 4)
            zscoreDay.append((day, zscore))
            #if zscore > 5.0:
            #    zscoreTest.append(math.floor(zscore))
        #if docid in range(50, 70) or docid in range(150, 170):
        #if len(zscoreTest) > 0:
        if docid in range(100000, 102000):
            print "#################################"
            print nnDayCounter.most_common()
            print sorted(zscoreDay, key = lambda a:a[1], reverse=True)
        zscoreDayArr.append(zscoreDay)
    print "## Tweets zscore by day obtained at", time.asctime()
    return zscoreDayArr


# choose docs appear in specific time window (day)
def filtering_by_zscore(zscoreDayArr, seqDayHash, day, thred_zscore):
    burstySeqIdArr = []
    zscores = []
    for docid, zscoreDay in enumerate(zscoreDayArr):
        if seqDayHash[docid] != day:
            continue
        if zscoreDay is None:
            continue
        zscore = None
        if len(zscoreDay) == 1:
            if zscoreDay[0][0] == day:
                zscore = zscoreDay[0][1] # zscoreDay from getBursty_tw
        else:
            print zscoreDay
            zscore = dict(zscoreDay).get(day) # zscoreDay from getBursty
        if zscore is None:
            continue
        #zscores.append(round(zscore, 1))
        zscores.append(math.floor(zscore))
        if zscore > thred_zscore:
            burstySeqIdArr.append(docid)
    print "## Tweets filtering by zscore ", thred_zscore, " in day ", day, " obtained at", time.asctime()
    #print "## statistic of zscore", Counter(zscores).most_common()
    return burstySeqIdArr
