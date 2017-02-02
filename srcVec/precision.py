import sys

def match2Lbl(line):
    if line.startswith("1-**"):
        return 1
    elif line.startswith("-**"):
        return 0

def getAnnoLbl(filename):
    annotatedFile = open(filename, 'r')
    content = annotatedFile.readlines()
    clusterLabels = [(content[lineIdx+1], match2Lbl(line)) for lineIdx, line in enumerate(content) if line[:4] in ["1-**", "-** "]]
    return clusterLabels

def caltopP(annotatedLabels, preK, days, topK_c):
    precision = []
    for para in preK:
        trueLbl_day = [sum(annotatedLabels[topK_c*i:topK_c*(i+1)][:para]) for i in range(days)]
        precision.append(round(float(sum(trueLbl_day))*100/(days*para), 2))
    print "pre@top", preK
    print "pre", precision

if __name__ == "__main__":
    print "Usage: python precision.py annotatedFilename topK_c"
    print sys.argv[1:3]

    filename = sys.argv[1]
    topK_c = int(sys.argv[2])
    clusterLabels = getAnnoLbl(filename)
    annotatedLabels = [label for firstTweet, label in clusterLabels]

    #days = 5 # in test
    #preK = [5, 10]

    days = 3 # in dev
    preK = [5, 10, 15, 20, 25, 30]

    caltopP(annotatedLabels, preK, days, topK_c)
