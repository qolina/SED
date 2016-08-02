import time
import re
import sys

sys.path.append("/home/yxqin/Scripts")
from hashOperation import *

def prepareSegged2POS():
    dirPath = "/home/yxqin/corpus/data_stock201504/segment/"
    for day in [str(i).zfill(2) for i in range(1, 32)]:
        infile = file(dirPath + "segged_tweetCleanText"+day)
        outputFile = file(dirPath + "topos_segged_tweetCleanText"+day, "w")
        print "Processing", infile.name, " outputFile", outputFile.name
        outputContent = []

        content = infile.readlines()
        for line in content:
            arr = line.split("\t")

            text = arr[2][:-1]

            newText = re.sub("\|", " ", text)
            outputContent.append(newText)

        outputFile.write("\n".join(outputContent))

        infile.close()
        outputFile.close()
        break


def posCount_fromParsed(parseFilePath):
    content = open(parseFilePath, "r").readlines()
    content = [line[:-1].split("\t") for line in content if len(line) > 1]
    
    wordPOSHash = {}
    for wordArr in content:
        word = wordArr[1]
        pos = wordArr[3]

        updateAppHash(wordPOSHash, word, pos, 1)

    for word in wordPOSHash:
        if word[0] == "$":
            wordPOSHash[word] = "^"
        else:
            sortedList = sorted(wordPOSHash[word].items(), key = lambda a:a[1], reverse=True)
            wordPOSHash[word] = sortedList[0][0]
 
    print "## End of reading file.[parsed text file]  unitNum: ", len(wordPOSHash), "Eg", wordPOSHash.items()[:2], parseFilePath
    return wordPOSHash


def posCount(posFilePath):
    wordPOSHash = {}
    posFile = file(posFilePath)
    content = posFile.readlines()
    for line in content:
        words = line[:-1].split(" ")
        for item in words:
            word = item[:item.rfind("_")]
            pos = item[item.rfind("_")+1:]

            updateAppHash(wordPOSHash, word, pos, 1)

    for word in wordPOSHash:
        if word[0] == "$":
            wordPOSHash[word] = "NNP"
        else:
            sortedList = sorted(wordPOSHash[word].items(), key = lambda a:a[1], reverse=True)
            wordPOSHash[word] = sortedList[0][0]
    return wordPOSHash



if __name__ == "__main__":
    print "Program starts at time:" + str(time.asctime())
#####################
#    prepareSegged2POS()
#####################

#####################

    posFilePath = "/home/yxqin/corpus/data_stock201504/segment/" + "pos_segged_tweetCleanText01"
    wordPOSHash = posCount(posFilePath)

#####################



    print "Program ends at time:" + str(time.asctime())
