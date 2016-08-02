# func: load content from tweet.json file. read each tweet into 
# tweetContent(original text) 
# and tweetStructure 

import os
import sys
import re
import time
import json
import cPickle


sys.path.append("/home/yxqin/Scripts/")
import lang

from Tweet import *
from tweetStrOperation import *
from hashOperation import *
from strOperation import *


# load companies in snp500
def loadSnP500(snpfilename):
    companies = open(snpfilename, "r").readlines()
    companies = [line.strip().lower() for line in companies]
    syms = [line.split("\t")[0] for line in companies]
    names = [line.split("\t")[1] for line in companies]
    sym_names = zip(syms, names)
    print "## End of reading file. [snp500 file][with rank]  snp companies: ", len(syms), "eg:", sym_names[0], snpfilename
    return sym_names


def loadTweetFromFile(jsonFileName, snpfilename, outFileName_tweetText, outFileName_tweetStruct):
    # debug format
    loadDataDebug = True

    mayContentHash = {} # tid:tweet

    snpSym = None
    if snpfilename is not None:
        snpNameHash = loadSnP500(snpfilename)
        snpSym = snpNameHash.keys()
        snpSym.sort()

    snpHash = {}

    # for statistics when debugging
    statisticArr = [0, 0] # non-eng tweets, encode-error tweets
    
    # should output result or not
    outputFlag_text = False
    outputFlag_struct = False
    if outFileName_tweetText is not None:
        outputFlag_text = True
        out_textFile = file(outFileName_tweetText, "w")
    if outFileName_tweetStruct is not None: # binary output
        outputFlag_struct = True
        out_structFile = file(outFileName_tweetStruct, "wb")

    jsonFile = file(jsonFileName)
    firstLine = jsonFile.readline()
    jsonContents = jsonFile.readlines()
    print "File loaded done. start processing", len(jsonContents), time.asctime()
    textOutArr = []
    structOutArr = []

    lineIdx = 0
### read file option 1 (line by line)
#    while 1:
#        lineStr = jsonFile.readline()
#        if not lineStr:
#            print "End of file. total lines: ", lineIdx
#            break
### read file option 2 (read all lines)
    for lineStr in jsonContents:
        lineIdx += 1

        if lineIdx % 10000 == 0:
#            if outputFlag_text:
#                for item in textOutArr:
#                    cPickle.dump(item, out_textFile)
#            if outputFlag_struct:
#                for item in structOutArr:
#                    cPickle.dump(item, out_structFile)

            print "Lines processed (stored): ", lineIdx, " at ", time.asctime()
#            del structOutArr[:]
#            del textOutArr[:]

        lineStr = lineStr[:-1]
        if len(lineStr) < 20:
            continue
#        lineStr = re.sub(r'\\\\', r"\\", lineStr)


        # compile into json format
        try:
            jsonObj = json.loads(lineStr)
        except ValueError as errInfo:
            if loadDataDebug:
                print "Non-json format! ", lineIdx, lineStr
            continue

        # create tweet and user instance for current jsonObj
        currTweet = getTweet(jsonObj)
        if currTweet is None: # lack of id_str
            if loadDataDebug:
                print "Null tweet (no id_str)", lineIdx, str(jsonObj)
            continue

        currUser = getUser(jsonObj)
        if currUser is None: # lack of user or user's id_str
            if loadDataDebug:
                print "Null user (no usr of usr's id_str)" + str(jsonObj)
            continue

        currTweet.user_id_str = currUser.id_str # assign tweet's user_id_str

        date = time.strftime("%b", readTime_fromTweet(currTweet.created_at))

        if date == "May":
            mayContentHash[currTweet.id_str] = currTweet.text.lower()
        continue

# filter out non-Eng Tweet
#        if not isENTweet(currTweet):
#            if loadDataDebug:
##                print "non-english"
##                print currTweet.id_str, currTweet.text
#                statisticArr[0] += 1
#            continue


        #print currTweet.id_str, "\t", currTweet.text.encode("utf-8", 'ignore')
        syms = [symStruct['text'] for symStruct in currTweet.symbols]
        if len(syms) == 0:
            cumulativeInsert(snpHash, "-", 1)
        containSym = [sym for sym in syms if sym.upper() in snpSym]
        if len(containSym) == 0:
#            print "_".join(syms), "\t", currTweet.id_str, "\t", currTweet.text.encode("utf-8", 'ignore')

            for sym in syms:
                cumulativeInsert(snpHash, sym, 1)

        # output
        # time filtering ,keep tweets between (20130101-20130115)
#        baseTime = readTime_fromTweet(currTweet.created_at)
#        baseDate = time.strftime("%Y-%m-%d", baseTime)
#        if baseDate.startsWith("2012-12-31"):
#            continue

        if outputFlag_struct:
#            cPickle.dump(currTweet, out_structFile)
            structOutArr.append(currTweet)
        if outputFlag_text:
            try:
                # -> leads to 250k encode error tweets when output to file directly
#                out_textFile.write(currTweet.id_str + " " + currTweet.text.encode("utf-8", 'ignore') + "\n")
                # checked  no \t in text
#                cPickle.dump(currTweet.id_str + "\t" + currTweet.text, out_textFile)
                textOutArr.append(currTweet.id_str + "\t" + currTweet.text)
#                print currTweet.text
            except Exception as errInfo:
                if loadDataDebug:
#                    print "encode error"
                    statisticArr[1] += 1
                continue

#        if lineIdx > 1000000:
#            print "Lines processed: ", lineIdx, " at ", time.asctime()
#            break

    if outputFlag_text:
        for item in textOutArr:
            cPickle.dump(item, out_textFile)
    if outputFlag_struct:
        for item in structOutArr:
            cPickle.dump(item, out_structFile)
    print "End of file. total lines: ", lineIdx

    jsonFile.close()
    if outputFlag_struct:
        out_structFile.close()
    if outputFlag_text:
        out_textFile.close()
    
    if loadDataDebug:
        print "Statictis of non-eng, encode-error tweets", statisticArr


    print "##################################"
    print sum(snpHash.values())
    print len(snpHash)
    output_sortedHash(snpHash, 0, False)
    print "##################################"
    output_sortedHash(snpHash, 1, True)

    return mayContentHash


def addCash2Text(oriText, text):
    if oriText is None:
        return text
    oriWords = oriText.split(" ")
    words = text.split("|")
    newWords = []
    for word in words:
        if "$"+word in oriWords:
            newWords.append("$"+word)
        else:
            newWords.append(word)

    return "|".join(newWords)


def addCash2seggedFile(mayContentHash):
    dirPath = "/home/yxqin/corpus/data_stock201504/segment/noCash/"
    outDirPath = "/home/yxqin/corpus/data_stock201504/segment/"
    for day in [str(i).zfill(2) for i in range(1, 32)]:
        infile = file(dirPath + "segged_tweetCleanText"+day)
        print "Processing", infile.name
        outputFile = file(outDirPath + "segged_tweetCleanText"+day, "w")
        outputContent = []

        content = infile.readlines()
        for line in content:
            arr = line.split("\t")

            oriText = mayContentHash.get(arr[0])
            text = arr[2][:-1]

            newText = addCash2Text(oriText, text)
            newArr = arr[0:2]
            newArr.append(newText)
            outputContent.append("\t".join(newArr))

        outputFile.write("\n".join(outputContent))

        infile.close()
        outputFile.close()

def getArg(args, flag):
    arg = None
    if flag in args:
        arg = args[args.index(flag)+1]
    return arg

def parseArgs(args):
    jsonFileName = getArg(args, "-json")
    if jsonFileName is None:
        sys.exit(0)

    snpFileName = getArg(args, "-snp")
    outFileName_tweetText = getArg(args, "-textOut")
    outFileName_tweetStruct = getArg(args, "-structOut")
    return jsonFileName, snpFileName, outFileName_tweetText, outFileName_tweetStruct

if __name__ == "__main__":
    print "Usage: python getsnp500.py -json tweet.jason.file -snp snpfilename [-textOut tweetTextFilename -structOut tweetStructureFilename]"
    print "       (eg. -json twitter-20130101.txt -snp ~/corpus/snp500_201504 -textOut tweetText-20130101.data -structOut tweetStructure-20130101.data)"

    print "Program starts at time:" + str(time.asctime())

    [jsonFileName, snpfilename, outFileName_tweetText, outFileName_tweetStruct] = parseArgs(sys.argv)

#####################
#    loadTweetFromFile(jsonFileName, snpfilename, outFileName_tweetText, outFileName_tweetStruct)
#####################

#####################
#    mayContentHash = loadTweetFromFile(jsonFileName, snpfilename, outFileName_tweetText, outFileName_tweetStruct)
#    addCash2seggedFile(mayContentHash)
#####################


    print "Program ends at time:" + str(time.asctime())
