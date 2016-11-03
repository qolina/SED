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


#################
## preprocess: make a tmp dir
def makeTempDir(stock_tmpDir):
    if os.path.exists(stock_tmpDir):
        for tmpfile in os.listdir(stock_tmpDir):
            os.remove(stock_tmpDir + tmpfile)
    else:
        os.mkdir(stock_tmpDir)

def load_reverb_triples(filename):
    content = open(filename, "r").readlines()
    triples = []
    for line in content:
        triples.extend(line.split("\t")[1][:-1].split(" "))
    triples = [strTriple.replace("_", " ").split("|") for strTriple in triples]
    return triples


tweetDataDir = '../ni_data/skl/'
snpFilePath = "../data/snp500_ranklist_20160801"
###############################################################
if __name__ == "__main__":
    sym_names = snpLoader.loadSnP500(snpFilePath)

    for reverbFileName in sorted(os.listdir(tweetDataDir)):
        if not reverbFileName.startswith("relSkl_2015-05-"): continue
        dayStr = "2015-05-" + reverbFileName[-2:]
        #if dayStr != "2015-05-01": continue
        #tweWordsArr_delAllSpecial(wordsArr)

        ## read triples
        triples = load_reverb_triples(tweetDataDir + reverbFileName)
        snp_syms = [snpItem[0] for snpItem in sym_names]
        snp_comp = [strUtil.getMainComp(snpItem[1]) for snpItem in sym_names]
        sym_triples = set(["###".join(triple) for sym in snp_syms for triple in triples if sym == triple[0]])
        comp_triples = set(["###".join(triple) for comp in snp_comp for triple in triples if " "+comp+" " in " ".join(["", triple[0], triple[2], ""])])

#        sym_triples_struct_debug = [(sym, triple) for sym in snp_syms for triple in triples if sym == triple[0]]
#        comp_triples_struct_debug = [(comp, triple) for comp in snp_comp for triple in triples if " "+comp+" " in " ".join(["", triple[0], triple[2], ""])]
#        print "## #sym_tri, #comp_tri", len(sym_triples_struct_debug), len(comp_triples_struct_debug)
#        for item in sym_triples_struct_debug:
#            print item[0], item[1]


        ## save to file
        snp_trifile = open(tweetDataDir + "snp_triple_"+dayStr, "w")
        cPickle.dump(sym_triples, snp_trifile)
        cPickle.dump(comp_triples, snp_trifile)
        print "## File saved (tweets triples file)", snp_trifile.name, "#sym_tri, #comp_tri", len(sym_triples), len(comp_triples)

