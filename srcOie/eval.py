import sys
import cPickle
import datetime

sys.path.append("../src/")
from util import snpLoader
from util import stringUtil as strUtil

def loadGoldTriple_humanAnn(goldfilename):
    content = open(goldfilename, "r").readlines()
    labels = [line[:-1][:line.find("- ")] for line in content]
    #labels = [abs(int(label)) for label in labels]
    triples = [line[:-1][line.find("- ")+2:] for line in content]

    true_triples = [triples[idx] for idx in range(len(triples)) if labels[idx] == "1"]
    li_triples = [triples[idx] for idx in range(len(triples)) if labels[idx] == "-1"]
    wrong_triples = [triples[idx] for idx in range(len(triples)) if labels[idx] == "0"]
    return true_triples, li_triples, wrong_triples

def loadTweetTriple_humanAnn(goldfilename):
    content = open(goldfilename, "r").readlines()
    labels = [line[0] for line in content]
    #labels = [abs(int(label)) for label in labels]
    triples = [line[2:-1].split("\t")[-1].strip() for line in content]

    true_triples = [triples[idx] for idx in range(len(triples)) if labels[idx] == "2"]
    li_triples = [triples[idx] for idx in range(len(triples)) if labels[idx] == "1"]
    wrong_triples = [triples[idx] for idx in range(len(triples)) if labels[idx] == "0"]
    return true_triples, li_triples, wrong_triples


def loadSysTriple_strVer(systemOutFilename):
    content = open(systemOutFilename, "r").readlines()
    content = [line[:-1] for line in content]
    return content

def strTriple_commonList(tripleList1, tripleList2):
    commonList = set(tripleList1) & set(tripleList2)
    return commonList

def loadSysTriple(systemOutFilename):
    sys_triples = []
    systemOutFile = open(systemOutFilename, "r")
    sym_triples = cPickle.load(systemOutFile)
    comp_triples = cPickle.load(systemOutFile)
    sys_triples.extend(sym_triples)
    sys_triples.extend(comp_triples)
    print "sysTriple eg:", sys_triples[0]
    sys_triples = [line.split("--") for line in sys_triples]
    sys_triples = [(item[0], item[1].split("###")) for item in sys_triples]

    return sys_triples

def triple_commonList(tripleList1, tripleList2):
    commonList = [(tri1, tri2) for tri1 in tripleList1 for tri2 in tripleList2 if same_triple(tri1, tri2)]
    return commonList

# arg both in format (compname, triple)
# triple in format [arg1, verb, arg2]
def same_triple(gold_triple, sys_triple):
    gold_compName = gold_triple[0]
    if gold_compName in snpHash:
        gold_compName = snpHash[gold_compName]
    sys_compName = sys_triple[0]
    if sys_compName in snpHash:
        sys_compName = snpHash[sys_compName]

    if gold_compName != sys_compName: # refer to different company
        return False
    gold_wordArr = " ".join(gold_triple[1]).split()
    sys_wordArr = " ".join(sys_triple[1]).split()
    commonWords = set(gold_wordArr) & set(sys_wordArr)
    unionWords = set(gold_wordArr) | set(sys_wordArr)

    #if len(commonWords)*1.0/len(gold_wordArr) > 0.4: # old coWord

    # jaccard coefficient
    # jc(s1, s2) = |s1 & s2| / |s1|s2|
    if len(commonWords)*1.0/len(unionWords) > 0.3:
        return True
    return False

##########################################
#print "Usage: python eval.py goldfile systemfile"
#print "eg: ../data/snp/snp_triple_in1st_2015-05-01 ../ni_data/word/snp_triple_2015-05-01"

snpFilePath = "../data/snp500_ranklist_20160801"
sym_names = snpLoader.loadSnP500(snpFilePath)
#print "## End of reading file. [snp500 file][with rank]  snp companies: ", len(sym_names), "eg:", sym_names[0], snpFilePath
global snpHash
snpHash = dict([(snpItem[0], strUtil.getMainComp(snpItem[1])) for snpItem in sym_names])


goldfilename = sys.argv[1]
sysfilename = sys.argv[2]

#########################
# string version system_output
#true_triples, li_triples, wrong_triples = loadGoldTriple_humanAnn(goldfilename)
#true_triples, li_triples, wrong_triples = loadTweetTriple_humanAnn(goldfilename)
#sys_triples = loadSysTriple_strVer(sysfilename)
#print sys_triples[0], true_triples[0]
#true_sysTriples = strTriple_commonList(true_triples, sys_triples)
#li_sysTriples = strTriple_commonList(li_triples, sys_triples)
#wrong_sysTriples = strTriple_commonList(wrong_triples, sys_triples)
#########################


#########################
#true_sysTriples = triple_commonList(true_triples, sys_triples)
#li_sysTriples = triple_commonList(li_triples, sys_triples)
#wrong_sysTriples = triple_commonList(wrong_triples, sys_triples)
#########################


#########################
#statisticGold = [len(true_triples), len(li_triples), len(wrong_triples)]
#statisticArr = [len(true_sysTriples), len(li_sysTriples), len(wrong_sysTriples)]
#print "#goldOut, #goldTrue, #goldLI, #goldWrong", sum(statisticGold), statisticGold
#print "#sysOut, #sysLabelled, #sysTrue, #sysLI, #sysWrong", len(sys_triples), sum(statisticArr), statisticArr
#pr = sum(statisticArr[:2])*100.0 / sum(statisticArr)
#re = sum(statisticArr[:2])*100.0 / sum(statisticGold[:2])
#########################


#########################
# struct version system_output -> bad result
# eval between gold news and tweets
dayStr = goldfilename[-10:]
dayBefore = str(datetime.date(int(dayStr[:4]), int(dayStr[-5:-3]), int(dayStr[-2:])) - datetime.timedelta(1))
gold_triples = loadSysTriple(goldfilename)
gold_triples.extend(loadSysTriple(goldfilename[:-10]+dayBefore))

sys_triples = loadSysTriple(sysfilename)
true_sysTriples = triple_commonList(gold_triples, sys_triples)
#print "true sysTri eg:", true_sysTriples[0][0], true_sysTriples[0][1]
for item in true_sysTriples:
    print item[0][0],"\t" ,"###".join(item[0][1]),"\t", "###".join(item[1][1])

print [len(true_sysTriples), len(sys_triples), len(gold_triples)]
pr = len(true_sysTriples)*100.0/len(sys_triples)
re = len(true_sysTriples)*100.0/len(gold_triples)
#########################


print pr, re, pr*re*2/(pr+re)

#########################
#for debug
# output not matched to annotated triples in strVer
#for triple in sys_triples:
#    if triple in true_triples:
#        print "2-" + triple
#    elif triple in li_triples:
#        print "1-" + triple
#    elif triple in wrong_triples:
#        print "0-" + triple
#    else:
#        print "-" + triple
#########################
