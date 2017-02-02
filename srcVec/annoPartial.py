import sys

from precision import getAnnoLbl

if __name__ == "__main__":
    print "Usage: python annoPartial.py 2bLblFilename, lblFilename"
    tblblFilename = sys.argv[1]
    lblFilename = sys.argv[2]
    print sys.argv[1:3]

    clusterLabels = getAnnoLbl(lblFilename)
    fstTweLabels = [(firstTweet[:firstTweet.find(" ")], lbl) for firstTweet, lbl in clusterLabels]
    fstTweLabels = dict(fstTweLabels)

    tbContent = open(tblblFilename, "r").readlines()

    processedTBContent = []
    for lineIdx, line in enumerate(tbContent):
        if line.startswith("1-**"):
            firstTweet = tbContent[lineIdx+1]
            tIdx = firstTweet[:firstTweet.find(" ")]
            lbl = fstTweLabels.get(tIdx)
            if lbl is None:
                lbl = "-1"
            elif lbl == 0:
                lbl = ""
            else:
                lbl = str(lbl)

            processedTBContent.append(lbl+line[1:])
        else:
            processedTBContent.append(line)

    outFile = open(tblblFilename+".al", "w")
    outFile.write("".join(processedTBContent))
    outFile.close()
