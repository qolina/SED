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
import random
import cPickle
from collections import Counter

import numpy as np
from gensim import models, similarities
from sklearn.metrics import pairwise
from sklearn.neighbors import NearestNeighbors, LSHForest, KDTree
from scipy.spatial.distance import cosine, sqeuclidean
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse import csr_matrix
import falconn

sys.path.append(os.path.expanduser("~") + "/Scripts/")
from hashOperation import statisticHash

from statistic import distDistribution
def getSim(dataset, thred_radius_dist):
    dataset = dataset[:200000,]

    # using
    if 0:
        nnModel = NearestNeighbors(radius=thred_radius_dist, algorithm='brute', metric='minkowski', p=2, n_jobs=1)
        #nnModel = LSHForest(radius=thred_radius_dist)
        nnModel.fit(dataset)
        #ngDistArray, ngIdxArray = nnModel.radius_neighbors(dataset)
        ngIdxArray = nnModel.radius_neighbors(dataset, return_distance=False)
    else:
        nnModel = KDTree(dataset, leaf_size=1.5*dataset.shape[0], metric="euclidean")
        ngIdxArray = nnModel.query_radius(dataset, thred_radius_dist)

    #ngIdxArray = np.asarray(ngIdxArray)
    print ngIdxArray.shape
    print "## Nearest neighbor with radius ", thred_radius_dist, " obtained at", time.asctime()
    return ngIdxArray

def trainLSH(para, dataset, thred_radius_dist):
    nnModel = getLshIndex(para, dataset)
    print "## default l, k, prob", para.l, para.k, nnModel.get_num_probes()

    #distMatrix = pairwise.cosine_distances(dataset)
    #nns_fromSim = [sorted(enumerate(distMatrix[i]), key = lambda a:a[1])[:10] for i in range(distMatrix.shape[0])]
    #print "## nn by cosine sim obtained.", time.asctime()

    prob_cands = range(para.l, 200, 100)
    best_num_probes, max_jc = para.l, 0.0
    for num_probes in prob_cands:
        t1 = timeit.default_timer()
        ngIdxArray, indexedInCluster, clusters = getLshNN_original(dataset, nnModel, thred_radius_dist, num_probes)
        t2 = timeit.default_timer()
        print "## probs", num_probes, "\t time", round(t2-t1, 2)
        jc = eval_sklearn_nnmodel(nns_fromSim, ngIdxArray)
        if jc > max_jc:
            max_jc = jc
            best_num_probes = num_probes
    return best_num_probes

# nns_fromSim used for eval lsh performance when training
def getSim_falconn(dataset, thred_radius_dist, trained_num_probes, nns_fromSim, validDayWind, dayLeftWindow):
    dataset = prepData_forLsh(dataset)

    num_setup_threads = 10
    para = getPara_forLsh(dataset.shape)
    para.num_setup_threads = num_setup_threads
    #para.l = 10 # num of hash tables
    #para.k = 5 # num of hash funcs per table

    #######################################
    # mainly train num_probes
    if 0:
        trained_num_probes = trainLSH(para, dataset, thred_radius_dist)
        return trained_num_probes
    #######################################

    #op = "1"
    op = "ori_valid"
    thred_sameTweetDist = 0.2
    if op == "1":
        ngIdxArray, indexedInCluster, clusters = getLshNN_op1(dataset, nnModel, thred_radius_dist, trained_num_probes, thred_sameTweetDist)
        print "## Nearest neighbor [Falconn_lsh_op1] with radius ", thred_radius_dist, ngIdxArray.shape, " obtained at", time.asctime()
        return ngIdxArray, indexedInCluster, clusters
    elif op == "2":
        ngIdxArray = getLshNN_op2(dataset, nnModel, thred_radius_dist, trained_num_probes, thred_sameTweetDist)
        print "## Nearest neighbor [Falconn_lsh_op2] with radius ", thred_radius_dist, ngIdxArray.shape, " obtained at", time.asctime()
    elif op == "ori":
        ngIdxArray = getLshNN_original(dataset, nnModel, thred_radius_dist, trained_num_probes)
        print "## Nearest neighbor [Falconn_lsh_ori] with radius ", thred_radius_dist, ngIdxArray.shape, " obtained at", time.asctime()
    elif op == "ori_valid":
        ngIdxArray, ngByDays = getLshNN_ori_valid(para, validDayWind, dayLeftWindow, dataset, thred_radius_dist, trained_num_probes)
        print "## Nearest neighbor [Falconn_lsh_ori_valid] with radius ", thred_radius_dist, ngIdxArray.shape, " obtained at", time.asctime()
    return ngIdxArray, None, None

def prepData_forLsh(dataset):
    dataset = dataset.astype(np.float32)
    #dataset -= np.mean(dataset, axis=0)
    dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
    return dataset

def getPara_forLsh(datasetShape):
    num_points, dim = datasetShape
    para = falconn.get_default_parameters(num_points, dim)
    para.distance_function = "euclidean_squared" # vanilla eu
    return para

def getLshIndex(para, dataset):
    nnModel = falconn.LSHIndex(para)
    nnModel.setup(dataset)
    print "## sim falconn data setup done. data", dataset.shape, time.asctime()
    return nnModel

# clusters: list of lists [[docid_c1], [docid_c2]]
# indexedInCluster: dict  docid:clusterIdx_in_clusters
def getLshNN_op1(dataset, nnModel, thred_radius_dist, trained_num_probes, thred_sameTweetDist):
    ngIdxList= []
    indexedInCluster = {}
    clusters = []
    for dataidx in range(dataset.shape[0]):
        if dataidx in indexedInCluster:
            nn_keys = None
        else:
            clusterIdx = len(clusters)
            indexedInCluster[dataidx] = clusterIdx
            clusters.append([dataidx])

            nnModel.set_num_probes(trained_num_probes)
            # nn_keys: (id1, id2, ...)
            nn_keys = nnModel.find_near_neighbors(dataset[dataidx,:], thred_radius_dist)

            nn_dists = [(idx, key) for idx, key in enumerate(nn_keys) if key > dataidx-130000 and key < dataidx+130000 and sqeuclidean(dataset[dataidx,:], dataset[key,:]) < thred_sameTweetDist]
            #nn_dists = [(idx, key) for idx, key in enumerate(nn_keys) if sqeuclidean(dataset[dataidx,:], dataset[key,:]) < 0.2]
            #print len(nn_keys), len(nn_dists), nn_dists[:min(10, len(nn_dists))], nn_dists[-min(10, len(nn_dists)):]

            for idx, key in nn_dists:
                indexedInCluster[key] = clusterIdx

        ngIdxList.append(nn_keys)
        if (dataidx+1) % 10000 == 0:
            print "## completed", dataidx+1, len(clusters), time.asctime()
    ngIdxList = np.asarray(ngIdxList)
    return ngIdxList, indexedInCluster, clusters

def getLshNN_op2(dataset, nnModel, thred_radius_dist, trained_num_probes):
    ngIdxList= []
    for dataidx in range(dataset.shape[0]):
        query_vec = dataset[dataidx,:]
        nnModel.set_num_probes(trained_num_probes)
        cand = nnModel.get_unique_sorted_candidates(query_vec)
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

def getLshNN_original(datasetP, nnModel, thred_radius_dist, trained_num_probes):
    ngIdxList= []
    for dataidx in range(datasetP.shape[0]):
        # nn_keys: (id1, id2, ...)
        nn_keys = nnModel.find_near_neighbors(datasetP[dataidx,:], thred_radius_dist)

        ngIdxList.append(np.asarray(nn_keys, dtype=np.int32))

        if (dataidx+1) % 10000 == 0:
            print "## completed", dataidx+1, time.asctime()

    ngIdxList = np.asarray(ngIdxList)
    return ngIdxList


def getLshNN_ori_valid(para, validDayWind, dayLeftWindow, dataset, thred_radius_dist, trained_num_probes):
    print "## Begin calculating lsh sim.", dataset.shape
    ngIdxArray = []
    dataNum = range(dataset.shape[0])
    for vdw, rel_dw in zip(validDayWind, dayLeftWindow):
        #print vdw, rel_dw
        if len(vdw) == 1:
            ngIdxArray.append(None)
            continue
        dataset_vdw = dataset[range(vdw[0], vdw[1]),:]
        nnModel = getLshIndex(para, dataset_vdw)

        ngIdxArray_day = []
        for dataidx in range(rel_dw[0], rel_dw[1]):
            nn_keys = nnModel.find_near_neighbors(dataset_vdw[dataidx], thred_radius_dist)
            ngIdxArray_day.append(np.asarray(nn_keys, dtype=np.int32))

            if (dataidx) % 10000 == 0:
                print "## nn cal completed", dataidx+1, time.asctime()

        ngIdxArray_day = np.asarray(ngIdxArray_day) + vdw[0]
        ngIdxArray.append(ngIdxArray_day)

        if 0:
            # statistic averaged_nn_ratio for influence of time window in nn

            #randIdx = random.sample(range(rel_dw[0], rel_dw[1]), 10000)
            randIdx = range(rel_dw[0], rel_dw[1])
            ngIdxList_stat = []
            for dataidx in randIdx:
                nn_keys = nnModel.find_near_neighbors(dataset_vdw[dataidx], thred_radius_dist)
                ngIdxList_stat.append(len(nn_keys))
            print "## Statistic: Avg #nn", np.mean(ngIdxList_stat), round(np.mean(ngIdxList_stat)*1.0/len(dataIdx_v), 4)
            continue

    byDays = True
    return np.asarray(ngIdxArray), byDays


def getSim_sparse(dataset, thred_radius_dist, validDayWind, dayLeftWindow):
    print "## Begin calculating tfidf sim.", dataset.shape
    ngIdxArray = []
    for vdw, rel_dw in zip(validDayWind, dayLeftWindow):
        if len(vdw) == 1:
            ngIdxArray.append(None)
            continue
        dataset_vdw = dataset[range(vdw[0], vdw[1]),:]

        #distArr = euclidean_distances(dataset_vdw[0], dataset_vdw)
        #distArr = euclidean_distances(dataset_vdw[range(rel_dw[0], rel_dw[1]), :], dataset_vdw)
        ngIdxArray_day = []
        for dataidx in range(rel_dw[0], rel_dw[1]):
            distArr = euclidean_distances(dataset_vdw[dataidx], dataset_vdw)
            nn_keys = [i for i, eu in enumerate(distArr[0]) if eu <= thred_radius_dist]
            ngIdxArray_day.append(np.asarray(nn_keys, dtype=np.int32))

            if (dataidx) % 10000 == 0:
                print "## nn cal completed", dataidx+1, time.asctime()

        ngIdxArray_day = np.asarray(ngIdxArray_day) + vdw[0]
        ngIdxArray.append(ngIdxArray_day)

    return np.asarray(ngIdxArray)
