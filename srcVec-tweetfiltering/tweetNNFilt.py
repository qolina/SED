import os
import sys
import time
import timeit
import math
from collections import Counter
from statistic import zsDistribution

import numpy as np

#######################
# statistic tweetSimDf by day
# tweetSimDfDayArr: [nnDay_Counter_seqid0, seq1, ...]
# nnDay_Counter_seqid: (day, tweet_nn_num)
def getDF(ngIdxArray, seqDayHash, timeWindow, indexedInCluster, clusters):
    if ngIdxArray.shape[0] < 50:
        print "## Processing simDF by day", ngIdxArray.shape[0]
        nnByDay = True
    else:
        tweetSimDfDayArr = calDF(ngIdxArray, seqDayHash, timeWindow, indexedInCluster, clusters)
        print "## Tweets simDF by day obtained at", time.asctime()
        return tweetSimDfDayArr

    ## nnByDay is True
    tweetSimDfDayArr = []
    for dayInt, ngIdxArray_day in enumerate(ngIdxArray):
        if ngIdxArray_day is None:
            tweetSimDfDayArr.append(None)
            continue
        print "############ simDF nnIdx in day", dayInt
        simDfDayArr_day = calDF(ngIdxArray_day, seqDayHash, None, None, None)
        tweetSimDfDayArr.append(simDfDayArr_day)
    return tweetSimDfDayArr

def calDF(ngIdxArray, seqDayHash, timeWindow, indexedInCluster, clusters):
    simDfDayArr = []
    for docid, nnIdxs in enumerate(ngIdxArray):
        if nnIdxs is None:
            if indexedInCluster is not None:
                sameDocId = clusters[indexedInCluster[docid]][0]
                nnIdxs = ngIdxArray[sameDocId]
                #print docid, sameDocId, len(nnIdxs)
                #continue
            else:
                simDfDayArr.append(None)
                continue
        nnDays = [seqDayHash.get(seqid) for seqid in list(nnIdxs)]

        if timeWindow is not None:
            date = int(seqDayHash.get(docid))
            if (date > 0-timeWindow[0]) and (date <= 31-timeWindow[1]):
                date_inTimeWin = [str(item).zfill(2) for item in range(date+timeWindow[0], date+timeWindow[1]+1)]
                nnDays = [item for item in nnDays if item in date_inTimeWin]
            else:
                nnDays = None
        if nnDays is not None:
            nnDay_count = Counter(nnDays)
        simDfDayArr.append(nnDay_count)
    return simDfDayArr

# zscoreDayArr = [zscore_seqid0, seq1, ...]
# zscore = (df-mean)/std  --> bad score
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
    print "## Tweets zscore by day [tw2: li zscore + tw] obtained at", time.asctime()
    return zscoreDayArr


# zscoreDayArr: [zscoreDay_seqid0, seq1, ...]
# zscoreDay_seqid: [(day, zscore), (day, zscore)]
def getBursty(simDfDayArr, dayTweetNumHash, tDate, timeWindow):
    if timeWindow is None:
        TweetNum_all = sum(dayTweetNumHash.values())
    else:
        tw = [str(int(tDate)+i).zfill(2) for i in range(timeWindow[0], timeWindow[1]+1)]
        TweetNum_all = sum([n for d, n in dayTweetNumHash.items() if d in tw])
    zscoreDayArr = []
    for docid, nnDayCounter in enumerate(simDfDayArr):
        statArr = []

        if timeWindow is not None:
            nnDayCounter = dict([(d, nnDayCounter[d]) for d in tw])

        docSimDF_all = sum(nnDayCounter.values())
        est_prob = docSimDF_all*1.0/TweetNum_all
        zscoreDay = []
        for day, simDf in nnDayCounter.items():
            if tDate is not None and tDate != day: continue
            if simDf < 1:
                continue
            TweetNum_day = dayTweetNumHash[day]
            mu = est_prob * TweetNum_day
            sigma = math.sqrt(mu*(1-est_prob))
            #print docid, day, simDf, mu, est_prob, sigma
            zscore = round((simDf*1.0-mu)/sigma, 4)
            zscoreDay.append((day, zscore))
            if tDate == day:
                statArr.extend([simDf, mu, sigma, zscore])
        statArr.extend([est_prob, docSimDF_all, dayTweetNumHash[tDate], TweetNum_all])

        if 0 and tDate == "06":
            print "#################################"
            print sorted(nnDayCounter.items(), key = lambda a:a[0])
            print statArr
            print sorted(zscoreDay, key = lambda a:a[1], reverse=True)
        zscoreDayArr.append(zscoreDay)
    print "## Tweets zscore by day [li zscore] obtained at", time.asctime()
    return zscoreDayArr

def getBursty_byday(simDfDayArr, dayTweetNumHash, timeWindow):
    print "## Processing zs by day", len(simDfDayArr)
    zscoreDayArr = []
    for dateInt, simDf_day in enumerate(simDfDayArr):
        if simDf_day is None:
            zscoreDayArr.append(None)
            continue
        zs_day = getBursty(simDf_day, dayTweetNumHash, str(dateInt+1).zfill(2), timeWindow)
        zscoreDayArr.append(zs_day)
    return zscoreDayArr

# choose docs appear in specific time window (day)
def filtering_by_zscore(zscoreDayArr, seqDayHash, day, thred_zscore):
    burstySeqIdArr = []
    zscores_day = []
    zscoresStat = []
    if len(zscoreDayArr) < 50:
        zscoreDayArr_day = enumerate(zscoreDayArr[int(day)-1])
    else:
        #zscoreDayArr_day = zscoreDayArr
        zscoreDayArr_day = [(docid, zscoreDay) for docid, zscoreDay in enumerate(zscoreDayArr_day) if seqDayHash[docid] == day]
    if zscoreDayArr_day is None: return None, None
    if thred_zscore <= -99:
        return [item[0] for item in zscoreDayArr_day], None

    for docid, zscoreDay in zscoreDayArr_day:
        if zscoreDay is None: continue
        zscore = None
        if len(zscoreDay) == 1:
            if zscoreDay[0][0] == day:
                zscore = zscoreDay[0][1] # zscoreDay from getBursty_tw
        else:
            print "Error zscore", zscoreDay
            zscore = dict(zscoreDay).get(day) # zscoreDay from getBursty
        if zscore is None: continue
        zscores_day.append((docid, zscore))

        #zscoresStat.append(round(zscore, 1))
        zscoresStat.append(math.floor(zscore))

    sorted_zscores_day = sorted(zscores_day, key = lambda a:a[1], reverse=True)
    if 0:
        tNum_thred = 3000 #len(zscores_day)/3
        tNum_bound = (3000, 10000)

        burstySeqIdArr = [docid for docid, zs in sorted_zscores_day if zs > thred_zscore]
        if thred_zscore > 0 and len(burstySeqIdArr) > tNum_bound[1]:
            burstySeqIdArr = burstySeqIdArr[:min(len(burstySeqIdArr), tNum_bound[1])]
        #elif thred_zscore > 0 and len(burstySeqIdArr) < tNum_bound[0]:
        #    burstySeqIdArr = [docid for docid, zs in sorted_zscores_day[:tNum_bound[0]]]
    elif 1:
        if thred_zscore < 1.0:
            thred_tNum = min(3000, thred_zscore * len(zscores_day))
        else:
            thred_tNum = int(thred_zscore)
        burstySeqIdArr = [docid for docid, zs in sorted_zscores_day[:thred_tNum]]

    print "## Tweets filtering by zscore ", thred_zscore, " in day ", day, " obtained at", time.asctime()

    return burstySeqIdArr, zscoresStat


def tweetForClustering(day, seqDayHash, zscoreDayArr, thred_zscore, startNumDay, dayTweetNumHash):
    burstySeqIdArr, zscoresStat = filtering_by_zscore(zscoreDayArr, seqDayHash, day, thred_zscore)
    if burstySeqIdArr is None: return None
    if len(zscoreDayArr) < 50: # need to add one startNum
        tweetFCSeqIdArr = [startNumDay+day_seqid for day_seqid in burstySeqIdArr]

    if len(tweetFCSeqIdArr) < 10:
        print "## Too less documents current day", day, len(tweetFCSeqIdArr)
        return None

    print "## Tweet filtering done.", len(tweetFCSeqIdArr), " out of", dayTweetNumHash[day], time.asctime()
    return tweetFCSeqIdArr, zscoresStat


