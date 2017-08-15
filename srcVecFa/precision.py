import sys

def match2Lbl(line):
    if line.startswith("1-**"):
        return 1
    elif line.startswith("-**"):
        return 0

def getAnnoLbl(filename):
    annotatedFile = open(filename, 'r')
    content = annotatedFile.readlines()
    print "######## pre recall by news"
    print "".join(content[-3:-1])
    recall_news = float(content[-2].split()[-1].strip()) 

    clusterLabels = [(content[lineIdx+1], match2Lbl(line)) for lineIdx, line in enumerate(content) if line[:4] in ["1-**", "-** "]]
    return clusterLabels, recall_news

def caltopP(annotatedLabels, preK, days, topK_c, recall_news):
    pre_byDay = []

    precision = []
    for para in preK:
        trueLbl_day = [sum(annotatedLabels[topK_c*i:topK_c*(i+1)][:para]) for i in range(days)]
        pre_byDay.append([("%.2f" %(item*100.0/para)) for item in trueLbl_day])
        precision.append(round(float(sum(trueLbl_day))*100/(days*para), 2))
    print "Pre_byDay @topK", preK
    for item in pre_byDay:
        print item

    print "pre@top", preK
    print "pre", precision
    p = precision[-1]
    print "f1", 2.0*p*recall_news/(p+recall_news)

if __name__ == "__main__":
    print "Usage: python precision.py annotatedFilename topK_c"
    print sys.argv[1:3]

    filename = sys.argv[1]
    topK_c = int(sys.argv[2])
    Kc_step = 1
    clusterLabels, recall_news = getAnnoLbl(filename)
    annotatedLabels = [label for firstTweet, label in clusterLabels]

    preK = range(Kc_step, topK_c+1, Kc_step)
    days = len(annotatedLabels)/topK_c

    caltopP(annotatedLabels, preK, days, topK_c, recall_news)
