#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import re
import sys
import commands
import string
import cPickle

sys.path.append("/home/yxqin/Scripts/")
from tweetStrOperation import tweWordsArr_delAllSpecial

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

tweetDataDir = '../ni_data/word/'
snpFilePath = "../data/snp500_ranklist_20160801"
###############################################################
if __name__ == "__main__":
    sym_names = snpLoader.loadSnP500(snpFilePath)

    ############ prepare file to be OIE analysed
    tweet_tmpDir = tweetDataDir + 'tmp/'
#    makeTempDir(tweet_tmpDir)

    for tweetFileName in sorted(os.listdir(tweetDataDir)):
        if not tweetFileName.startswith("tweetCleanText"): continue
        dayStr = "2015-05-" + tweetFileName[-2:]
        if dayStr == "2015-05-01": continue

        content = open(tweetDataDir + tweetFileName, "r").readlines()
        #printable = set(string.printable)
        #content = filter(lambda x:x in printable, content)

        content = [line.split("\t")[1][:-1].split() for line in content]
        content = [" ".join(tweWordsArr_delAllSpecial(wordsArr)) for wordsArr in content]

        line_tweetFileName = tweet_tmpDir + "tweets_"+dayStr
        output_file = open(line_tweetFileName, "w")
        output_file.write("\n".join(content)) 
        output_file.close()
        print "## File written. (temporary lined tweet data file)", line_tweetFileName

        ####### Apply oie tool
        oie_filename = tweet_tmpDir + "oie-" + dayStr 
        ## ollie
        ollie_commandline = "java -jar ../tools/ollie/ollie-app-latest.jar " + line_tweetFileName + " > " + oie_filename
        os.system(ollie_commandline)
        os.remove(line_tweetFileName)

        print "## File obtained. (tweet oie output)", oie_filename
        print "## File removed. (temporary lined tweet data file)", line_tweetFileName

        ## read triples
        triples = labelGold.load_oie_triples(oie_filename)
        snp_syms = [snpItem[0] for snpItem in sym_names]
        snp_comp = [strUtil.getMainComp(snpItem[1]) for snpItem in sym_names]
        sym_triples = set(["###".join(triple) for sym in snp_syms for (score, triple) in triples if sym == triple[0] and score > 0.5])
        comp_triples = set(["###".join(triple) for comp in snp_comp for (score, triple) in triples if " "+comp+" " in " ".join(["", triple[0], triple[2], ""]) and score > 0.5])

#        sym_triples_struct_debug = [(sym, score, triple) for sym in snp_syms for (score, triple) in triples if sym == triple[0] and score > 0.5]
#        comp_triples_struct_debug = [(comp, score, triple) for comp in snp_comp for (score, triple) in triples if " "+comp+" " in " ".join(["", triple[0], triple[2], ""]) and score > 0.5]
#        print "## #sym_tri, #comp_tri", len(sym_triples_struct_debug), len(comp_triples_struct_debug)
#        for item in comp_triples_struct_debug:
#            print item[0], item[1], item[2]


        ## save to file
        snp_trifile = open(tweetDataDir + "snp_triple_"+dayStr, "w")
        cPickle.dump(sym_triples, snp_trifile)
        cPickle.dump(comp_triples, snp_trifile)
        print "## File saved (tweets triples file)", snp_trifile.name, "#sym_tri, #comp_tri", len(sym_triples), len(comp_triples)

