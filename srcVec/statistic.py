
from collections import Counter
from sklearn.metrics import pairwise

def distDistribution(dataset):
    distmatrix = pairwise.euclidean_distances(dataset, dataset)
    #distmatrix = pairwise.cosine_distances(dataset, dataset)
    ds = distmatrix.flatten()
    ds = [round(item, 1) for item in ds]

    ds = Counter(ds)
    num = sum([item[1] for item in ds.items()])
    print [(item[0],round(float(item[1])/num, 3)) for item in sorted(ds.items(), key = lambda a:a[0])]
    print sorted(ds.items(), key = lambda a:a[0])


