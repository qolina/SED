
import time
import re
import sys

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


