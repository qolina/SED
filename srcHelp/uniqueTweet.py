import sys

sys.path.append("/home/yxqin/Scripts/")
from hashOperation import *

from getSnP500 import loadSnP500
from burstyEntityLinking import comps_in_a_sent

def getUniqueContent(filename):
    content = file(filename, "r").readlines()
    content = [line.strip().lower().split("\t") for line in content]
    contentHash = {}
    for arr in content:
        tid = arr[0]
        tweet = arr[1]
        if tweet in contentHash:
            contentHash[tweet].append(tid)
        else:
            contentHash[tweet] = [tid]

    print "## ", len(contentHash), " unique tweets are obtained. out of raw tweets: ", len(content)
    return contentHash

##################################
#main
if __name__ == "__main__":
    K = 10 # num of companies

    sym_names = loadSnP500("/home/yxqin/corpus/obtainSNP500/snp500_ranklist_20160801")
    snpSym = ["$"+item[0] for item in sym_names][:K]

    contentHash = getUniqueContent(sys.argv[1])

    rankedTweets = [tweet for tweet in contentHash if hasSameKey_le1_inlist(tweet.split(), snpSym) > 0]
    
    print "## ", len(rankedTweets), " unique tweets with snp", K, "comp are obtained."
    rankedTweets.sort()
    print "\n".join(rankedTweets)
