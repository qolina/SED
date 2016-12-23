#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import re
import sys
import commands
import string
import cPickle

sys.path.append("../../Scripts/")
from tweetStrOperation import tweWordsArr_delAllSpecial
from hashOperation import updateAppHash
from hashOperation import output_sortedHash
from hashOperation import sortHash

sys.path.append("../src/")
from util import snpLoader
from util import stringUtil as strUtil

import labelGold

#################
## preprocess: make a tmp dir
def makeTempDir(stock_tmpDir):
    if os.path.exists(stock_tmpDir):
        for tmpfile in os.listdir(stock_tmpDir):
            os.remove(stock_tmpDir + tmpfile)
    else:
        os.mkdir(stock_tmpDir)


# snp_syms = [sym, sym...]
# snp_symname = [(sym, name), (sym, name),...]
def load_snp_triples(snp_symname, oie_filename, conf_score):
    content = open(oie_filename, "r").read().lower()
    tweet_triple_Arr = content.strip().split("\n\n")
    tweetArr = []
    triplesArr = []
    snpTweetHash = {} # snpId: appHash    appHash-> tweetID:#count
    # statistic snpTweetHash
    for tweetID in range(len(tweet_triple_Arr)):
        arr = tweet_triple_Arr[tweetID].strip().split("\n")
        tripleLines = [line for line in arr[1:] if re.match("0\.[\d]+: \(", line)]
        triples = [labelGold.load_triple(line) for line in tripleLines]

        tweetArr.append(arr[0])
        triplesArr.append(triples)

        for snpId in range(len(snp_symname)):
            sym,name = snp_symname[snpId]
            if "$"+sym in arr[0].split():
                updateAppHash(snpTweetHash, snpId, tweetID, 1)

    # statistic snpTripleHash
    snpTripleHash = {} # snpId: triples
    for snpId in snpTweetHash:
        sym, compName = snp_symname[snpId]
        if len(snpTweetHash[snpId]) < 5:
            continue
        triples = []
        for tid in snpTweetHash[snpId].keys():
            triples.extend(triplesArr[tid])
        #triples = [triple for triple in triples if triple[0] > conf_score and sym in " ".join([triple[1][0], triple[1][2]]).split()] # sym in SO
        #triples = [triple for triple in triples if triple[0] > conf_score and sym == triple[1][0]] # sym = S
        triples = [triple for triple in triples if triple[0] > conf_score and (sym in " ".join(triple[1]) or compName in " ".join(triple[1]))] # sym, comp in SVO
        if len(triples) > 0:
            snpTripleHash[snpId] = triples

    # statistic #triple
    #snpTripleNumHash = dict([(sym, len(snpTripleHash[sym])) for sym in snpTripleHash])
    #print "Total #triples", sum(snpTripleNumHash.values())
    #sortedList = sortHash(snpTripleNumHash, 1, True)
    #for sym, num in sortedList:
    #    print sym, num
    #    triples = snpTripleHash[sym]
    #    triples = ["###".join(triple[1]) for triple in triples]
    #    print "\n".join(sorted(list(set(triples))))
    
    return snpTripleHash, snpTweetHash, tweetArr


tweetDataDir = '../ni_data/word/'
snpFilePath = "../data/snp500_ranklist_20160801"
###############################################################
if __name__ == "__main__":
    sym_names = snpLoader.loadSnP500(snpFilePath)
    #print "## End of reading file. [snp500 file][with rank]  snp companies: ", len(sym_names), "eg:", sym_names[0], snpFilePath

    ############ prepare file to be OIE analysed
    tweet_tmpDir = tweetDataDir + 'tmp/'
#    makeTempDir(tweet_tmpDir)

    for tweetFileName in sorted(os.listdir(tweetDataDir)):
        if not tweetFileName.startswith("tweetCleanText"): continue
        dayStr = "2015-05-" + tweetFileName[-2:]
        if dayStr != "2015-05-01": continue

        content = open(tweetDataDir + tweetFileName, "r").readlines()
        #printable = set(string.printable)
        #content = filter(lambda x:x in printable, content)

        content = [line.split("\t")[1][:-1].split() for line in content]
        content = [" ".join(tweWordsArr_delAllSpecial(wordsArr)) for wordsArr in content]

        line_tweetFileName = tweet_tmpDir + "tweets_"+dayStr
        output_file = open(line_tweetFileName, "w")
        output_file.write("\n".join(content))
        output_file.close()
        #print "## File written. (temporary lined tweet data file)", line_tweetFileName

        ####### Apply oie tool
        oie_filename = tweet_tmpDir + "oie-" + dayStr
        ## ollie
        ollie_commandline = "java -jar ../tools/ollie/ollie-app-latest.jar " + line_tweetFileName + " > " + oie_filename
        #os.system(ollie_commandline)
        os.remove(line_tweetFileName)

        #print "## File obtained. (tweet oie output)", oie_filename
        #print "## File removed. (temporary lined tweet data file)", line_tweetFileName

        snp_syms = [snpItem[0] for snpItem in sym_names]
        snp_comp = [strUtil.getMainComp(snpItem[1]) for snpItem in sym_names]
        conf_score = 0.8

        snpTripleHash, snpTweetHash, tweetArr = load_snp_triples(zip(snp_syms, snp_comp), oie_filename, conf_score)

        ## read comp-related triples
        #triples = labelGold.load_oie_triples(oie_filename)
        #sym_triples = set([sym+"--"+"###".join(triple) for sym in snp_syms for (score, triple) in triples if sym == triple[0] and score > conf_score])
        #comp_triples = set([comp+"--"+"###".join(triple) for comp in snp_comp for (score, triple) in triples if " "+comp+" " in " ".join(["", triple[0], triple[2], ""]) and score > conf_score])

        # debug
#        sym_triples_struct_debug = set([sym+"\t"+str(score)+"\t"+"###".join(triple) for sym in snp_syms for (score, triple) in triples if sym == triple[0] and score > conf_score])
#        comp_triples_struct_debug = set([comp+"\t"+str(score)+"\t"+"###".join(triple) for comp in snp_comp for (score, triple) in triples if " "+comp+" " in " ".join(["", triple[0], triple[2], ""]) and score > conf_score])
#        print "## #sym_tri, #comp_tri", len(sym_triples_struct_debug), len(comp_triples_struct_debug)
#        print "\n".join(sorted(list(sym_triples_struct_debug)))
#        print "\n".join(sorted(list(comp_triples_struct_debug)))


        ## save to file
#        snp_trifile = open(tweetDataDir + "snp_triple_"+dayStr, "w")
#        cPickle.dump(sym_triples, snp_trifile)
#        cPickle.dump(comp_triples, snp_trifile)
#        print "## File saved (tweets triples file)", snp_trifile.name, "#sym_tri, #comp_tri", len(sym_triples), len(comp_triples)

