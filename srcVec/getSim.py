#!/usr/bin/python
# -*- coding: UTF-8 -*-

## function
## given a numpy array like dataset
## calculate its similarity matrix

import os
import sys
import time
import timeit
import math
from collections import Counter

import numpy as np
from gensim import models, similarities
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors, LSHForest, KDTree
from scipy.spatial.distance import cosine, sqeuclidean
import falconn

sys.path.append(os.path.expanduser("~") + "/Scripts/")
from hashOperation import statisticHash

def getSim(dataset, thred_radius_dist):
    #dataset = dataset[:10000,]

    #######################
    # nearest neighbor method 1
    ## too slow for scipy calculate nearest neighbors
    #simMatrix = pairwise.cosine_similarity(dataset)
    #distMatrix = pairwise.cosine_distances(dataset)
    #nns_fromSim = [sorted(enumerate(distMatrix[i]), key = lambda a:a[1])[:20] for i in range(distMatrix.shape[0])]
    #print "## Similarity Matrix obtained at", time.asctime()

    #######################
    # nearest neighbor method 2
    #thred_n_neighbors = 10
    # choosing best
    #nnModels = [NearestNeighbors(radius=thred_radius_dist),
    #            NearestNeighbors(n_neighbors=thred_n_neighbors, algorithm='auto'), 
    #            NearestNeighbors(n_neighbors=thred_n_neighbors, algorithm='ball_tree'),
    #            NearestNeighbors(n_neighbors=thred_n_neighbors, algorithm='kd_tree'),
    #            LSHForest(random_state=30, n_neighbors=thred_n_neighbors)
    #            ]
    #for modelIdx in range(len(nnModels)):
    #    if modelIdx in [0, 2, 3, 4]: # choose auto
    #        continue
    #    nnModel = nnModels[modelIdx]
    #    nnModel.fit(dataset)
    #    if modelIdx == 0:
    #        ngDistArray, ngIdxArray = nnModel.radius_neighbors()
    #    elif modelIdx == 4:
    #        ngDistArray, ngIdxArray = nnModel.kneighbors(dataset)
    #    else:
    #        ngDistArray, ngIdxArray = nnModel.kneighbors()
    #    print ngDistArray.shape, ngIdxArray.shape
    #    print "## Nearest neighbor with model ", modelIdx, " obtained at", time.asctime()
    #    eval_sklearn_nnmodel(nns_fromSim, ngIdxArray)

    # using
    nnModel = NearestNeighbors(radius=thred_radius_dist)
    #nnModel = LSHForest(radius=thred_radius_dist)
    nnModel.fit(dataset)
    ngDistArray, ngIdxArray = nnModel.radius_neighbors(dataset)
    #nnModel = KDTree(dataset, leaf_size=1.5*dataset.shape[0], metric="euclidean")
    #ngIdxArray = nnModel.query_radius(dataset, thred_radius_dist)
    print ngDistArray.shape, ngIdxArray.shape
    print "## Nearest neighbor with radius ", thred_radius_dist, " obtained at", time.asctime()
    return ngDistArray, ngIdxArray

# nns_fromSim used for eval lsh performance when training
def getSim_falconn(dataset, thred_radius_dist, trainLSH, trained_num_probes, nns_fromSim):
    dataset = prepData_forLsh(dataset)
    num_setup_threads = 20
    para = getPara_forLsh(dataset)
    para.num_setup_threads = num_setup_threads
    #para.l = 10 # num of hash tables
    #para.k = 5 # num of hash funcs per table
    nnModel = getLshIndex(para, dataset)

    # mainly train num_probes
    if trainLSH:
        print "## default l, k, prob", para.l, para.k, nnModel.get_num_probes()

        #distMatrix = pairwise.cosine_distances(dataset)
        #nns_fromSim = [sorted(enumerate(distMatrix[i]), key = lambda a:a[1])[:10] for i in range(distMatrix.shape[0])]
        #print "## nn by cosine sim obtained.", time.asctime()

        prob_cands = range(para.l, 200, 100)
        best_num_probes, max_jc = para.l, 0.0
        for num_probes in prob_cands:
            t1 = timeit.default_timer()
            ngIdxArray, indexedInCluster, clusters = getLshNN(dataset, nnModel, thred_radius_dist, num_probes)
            t2 = timeit.default_timer()
            print "## probs", num_probes, "\t time", round(t2-t1, 2)
            jc = eval_sklearn_nnmodel(nns_fromSim, ngIdxArray)
            if jc > max_jc:
                max_jc = jc
                best_num_probes = num_probes
        return best_num_probes

    ngIdxArray, indexedInCluster, clusters = getLshNN_op1(dataset, nnModel, thred_radius_dist, trained_num_probes)
    print "## Nearest neighbor [Falconn_lsh] with radius ", thred_radius_dist, ngIdxArray.shape, " obtained at", time.asctime()
    return ngIdxArray, indexedInCluster, clusters

def prepData_forLsh(dataset):
    dataset = dataset.astype(np.float32)
    #dataset -= np.mean(dataset, axis=0)
    dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
    return dataset

def getPara_forLsh(dataset):
    num_points, dim = dataset.shape
    para = falconn.get_default_parameters(num_points, dim)
    return para

def getLshIndex(para, dataset):
    nnModel = falconn.LSHIndex(para)
    nnModel.setup(dataset)
    print "## sim falconn data setup", time.asctime()
    return nnModel

def getLshNN_op1(dataset, nnModel, thred_radius_dist, trained_num_probes):
    ngIdxList= []
    indexedInCluster = {}
    clusters = []
    for dataidx in range(dataset.shape[0]):
        if dataidx in indexedInCluster: 
            nn_keys = None
        else:
            indexedInCluster[dataidx] = len(clusters)
            clusters.append([dataidx])

            nnModel.set_num_probes(trained_num_probes)
            # nn_keys: (id1, id2, ...)
            nn_keys = nnModel.find_near_neighbors(dataset[dataidx,:], thred_radius_dist)

            for key in nn_keys:
                if cosine(dataset[dataidx,:], dataset[key, :]) < 0.1:
                    indexedInCluster[key] = indexedInCluster[dataidx]

        ngIdxList.append(nn_keys)
        if (dataidx+1) % 10000 == 0:
            print "## completed", dataidx+1, len(clusters), time.asctime()
    ngIdxList = np.asarray(ngIdxList)
    return ngIdxList, indexedInCluster, clusters

def getLshNN_op2(dataset, nnModel, thred_radius_dist, trained_num_probes):
    ngIdxList= []
    print dataset.shape
    for dataidx in range(dataset.shape[0]):
        query_vec = dataset[dataidx,:]
        nnModel.set_num_probes(trained_num_probes)
        cand = nnModel.get_unique_candidates(query_vec)
        cand = [idx for idx in cand if idx>dataidx-130000 and idx<dataidx+130000]

        #distMatrix = pairwise.cosine_distances(dataset[cand, :], query_vec)
        #query_vec = np.asarray([query_vec])
        #distMatrix = pairwise.pairwise_distances(dataset[cand, :], query_vec, metric='sqeuclidean', n_jobs=5)
        #distMatrix = pairwise.euclidean_distances(dataset[cand, :], query_vec, squared=True)
        #distMatrix = np.dot(dataset[cand, :], query_vec)
        #distMatrix = [sqeuclidean(dataset[idx,:], query_vec) for idx in cand]
        # nn_keys: (id1, id2, ...)
        #nn_keys = [idx for idx, dist in zip(cand, distMatrix) if dist**2 <= thred_radius_dist]
        nn_keys = nnModel.find_near_neighbors(dataset[dataidx,:], thred_radius_dist)

        ngIdxList.append(nn_keys)
        if (dataidx+1) % 1000 == 0:
            print "## completed", dataidx+1, time.asctime()
    ngIdxList = np.asarray(ngIdxList)
    return ngIdxList
