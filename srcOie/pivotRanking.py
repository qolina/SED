# serves as sub model for OllieTwi
# ranking pivot triples for each company

import editdistance as levDis
from nltk.stem.porter import *
global stemmer = PorterStemmer()

def pivorRanking(snp_symname, snpTripleHash, snpTweetHash, tweetArr):
    for snpId in snpTripleHash:
        sym, compName = snp_symname[snpId]
        triples = snpTripleHash[snpId]
        tweetIds = snpTweetHash[snpId]
        tweets = [tweetArr[tid] for tid in tweetIds]

        for triple in triples:
            tweetSupportScore = tweetSupportTriple(triple, tweets)
            pivotCoherenceScore = tripleSupportTriple(triple, triples)
            pivotBurstiness = tripleBursty(triple, tweets)

def tweetSupportTriple(triple, tweets):
    stringTriple = " ".join(triple)
    stemmedArrTriple = [stemmer.stem(item) for item in stringTriple.split()]
    stemmedArrTweet = [stemmer.stem(item) for item in tweet.split()]
    # in TverskySim, tweet served as prototype, small alpha
    alpha = 0.2
    simScores = [TverskySim(stemmedArrTriple, stemmedArrTweet, alpha) for tweet in tweets]
    #simScores = [max(len(stringTriple), len(tweet)) - LevenshteinDis(stringTriple, tweet) for tweet in tweets]

    return sum(simScores)


def tripleSupportTriple(triple, triples):
    simScores = [tripleSim(triple, triItem) for triItem in triples if triItem != triple]
    return sum(simScores)

def tri2stemmedArr(triple):
    return [stemmer.stem(item) for item in " ".join(triple).split()]

def tripleSim(tri1, tri2):
    combinedIndex = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
    combinedTriPair = [(tri1[idx[0]:idx[1]], tri2[idx[0]:idx[1]]) for idx in combinedIndex]
    combinedTriSims = [JaccardSim(tri2stemmedArr(triPair[1]), tri2stemmedArr(triPair[1])) for triPair in combinedTriPair]

    return sum(combinedTriSims)

def tripleBursty(triple, tweets):
    return 0.0


# Jaccard Coefficient
# jc(s1, s2) = |s1 & s2| / |s1|s2|
def JaccardSim(firstArr, secondArr):
    joinList = set(firstArr) & set(secondArr)
    unionList = set(firstArr) | set(secondArr)
    return len(joinList)*1.0/len(unionList)

# Tversky index
# |X&Y| / (|X&Y| + a*|X-Y| + b*|Y-X|)
# suitable assign:  a + b = 1
def TverskySim(firstArr, secondArr, alpha):
    joinList = set(firstArr) & set(secondArr)
    fstNoSndList = set(firstArr).difference(set(secondArr))
    sndNoFstList = set(secondArr).difference(set(firstArr))
    return len(joinList)/ (len(joinList) + alpha*len(fstNoSndList) + (1-alpha)*len(sndNoFstList))

# Levenshtein Distance  : a slow version
def LevenshteinDis(str1, str2):
    i = len(str1)
    j = len(str2)

    if min(i, j) == 0:
        return max(i, j)
    else:
        dis1 = LevenshteinDis(str1[:-1], str2)
        dis2 = LevenshteinDis(str1, str2[:-1])
        dis3 = LevenshteinDis(str1[:-1], str2[:-1])
        if str1[i-1] != str2[j-1]:
            return min(dis1+1, dis2+1, dis3+1)
        else:
            return min(dis1+1, dis2+1, dis3)

##########################################
### main
if __name__ == "__main__":
    # testing for levenshteindis - a fast python package
    print levDis.eval("banana", "bahama")
