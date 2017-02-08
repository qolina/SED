import sys
import time

from pyclausie import ClausIE

print time.asctime()

#sents = open(sys.argv[1],"r").readlines()
#sents = [line.split("\t")[1][:-1] for line in sents]

#sents = ["The so-called 13th month payment to pensioners has been target of some euro zone finance ministers whose countries hav less generous systems but are lending to Greece as part of a 24 billion euro EU/IMF bailout."]

sents = ["Indeed, Goldman reckons a move by EM funds to a neutra China allocation would result in inflows of up to $26 billion."]

cl = ClausIE.get_instance("/home/yxqin/nlpTools/clausie/clausie.jar")

for triple in cl.extract_triples(sents):
    print triple

# process a sentence once does NOT work!
print time.asctime()
