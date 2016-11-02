#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
import re
import sys
import commands
import subprocess
import string
import cPickle

from nltk.tokenize import sent_tokenize

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

stock_newsDir = '../ni_data/stocknews/'
snpFilePath = "../data/snp500_ranklist_20160801"
###############################################################
if __name__ == "__main__":
    sym_names = snpLoader.loadSnP500(snpFilePath)

    ############ prepare file to be OIE analysed
    stock_tmpDir = stock_newsDir + 'tmp/'
#    makeTempDir(stock_tmpDir)

    for dayDir in sorted(os.listdir(stock_newsDir)):
        #if dayDir < "2015-05-14": continue
        if dayDir.endswith("tmp"): continue
        #if dayDir == "2015-04-30": continue
        dayContent = []

        for newsfile in sorted(os.listdir(stock_newsDir + dayDir)):
            #print newsfile
            content = open(stock_newsDir + dayDir + "/" + newsfile, "r").read()
            printable = set(string.printable)
            content = filter(lambda x:x in printable, content)

            content = [line.strip() for line in content.split("\n")]
            content = [line.replace("* ", ". ") for line in content if len(line) > 1]

            if len(content) <= 5:
                continue

            dayContent.append(content[0][3:])
            #dayContent.append(content[3]) # filename for debug

            ##### delete special words in the first line of content
            line = content[4]
            content[4] = delHeadofContent(line)

            sents = sent_tokenize(" ".join(content[4:]))
            dayContent.extend(sents)
            
        #content = [str(idx+1)+"\t"+dayContent[idx] for idx in range(len(dayContent)) if len(dayContent[idx]) < 500] # for tool clausie

        line_newsfilename = stock_tmpDir + dayDir
        output_file = open(line_newsfilename, "w")
        output_file.write("\n".join(dayContent)) 
        output_file.close()
        print "## File written. (temporary lined newsfile)", line_newsfilename

        ####### Apply oie tool
        oie_filename = stock_tmpDir + "oie-" + dayDir 

        ## clausie [exceptions]
        #clausie_commandline = "java -jar ../clausie/clausie.jar -vlf " + line_newsfilename + " -o " + oie_filename
        #clausie_commandline = "sh ../clausie/clausie.sh -vlf " + line_newsfilename + " -o " + oie_filename

        ## ollie
        ollie_commandline = "java -jar ../ollie/ollie-app-latest.jar " + line_newsfilename + " > " + oie_filename
        os.system(ollie_commandline)
        os.remove(line_newsfilename)

        print "## File obtained. (news oie output)", oie_filename
        print "## File removed. (temporary lined newsfile)", line_newsfilename


        ## read triples
        triples = load_oie_triples(oie_filename)
        snp_syms = [snpItem[0] for snpItem in sym_names]
        snp_comp = [strUtil.getMainComp(snpItem[1]) for snpItem in sym_names]
        sym_triples = set(["###".join(triple) for sym in snp_syms for (score, triple) in triples if sym == triple[0] and score > 0.5])
        comp_triples = set(["###".join(triple) for comp in snp_comp for (score, triple) in triples if " "+comp+" " in " ".join(["", triple[0], triple[2], ""]) and score > 0.5])


        ## save to file
        snp_trifile = open(stock_newsDir + "snp_triple_"+dayDir, "w")
        cPickle.dump(sym_triples, snp_trifile)
        cPickle.dump(comp_triples, snp_trifile)
        print "## File saved (news triples file)", snp_trifile.name, "#sym_tri, #comp_tri", len(sym_triples), len(comp_triples)

