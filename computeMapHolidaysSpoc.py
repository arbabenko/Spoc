import numpy as np
import sys
sys.path.append('/home/arbabenko/libs/yael_v438')
sys.path.append('/home/arbabenko/libs/yael_v438/yael')

from xvecReadWrite import *
from yael import ynumpy
import commands
import os
from sklearn.decomposition import PCA
import scipy.spatial.distance as distance
from sklearn.preprocessing import normalize, scale
import cPickle as pickle

if __name__ == '__main__':
    decafsFile = sys.argv[1]
    dim = int(sys.argv[3])
    decafsCount = 1491
    namesFile = sys.argv[2]
    premFile = open(decafsFile, "rb")
    featureMapSide = 37
    featuresFile = open(decafsFile, "rb")
    pooledFeatures = np.zeros((decafsCount, dim), dtype='float32')
    for i in xrange(decafsCount):
        print i
        features = readXVecsFromOpenedFile(featuresFile, dim, featureMapSide * featureMapSide, 'fvecs')
        features = features.transpose(1,0).reshape(dim, featureMapSide, featureMapSide).copy()
        pooledFeatures[i,:] = np.sum(np.sum(features[:,:,:], axis=-1), axis=-1)
    pooledFeatures = normalize(pooledFeatures)
    filePca = open('./pcaFlickr.dat', 'rb')
    (avg, sing, pcamat) = pickle.load(filePca)
    pooledFeatures -= avg
    spocs = np.dot(pooledFeatures, pcamat.T)
    spocs /= sing
    spocs = normalize(spocs)
    nameFile = open(namesFile, 'r')
    idToName = nameFile.readlines()
    nameToId = {}
    for i in xrange(len(idToName)):
        nameToId[idToName[i].strip()] = i
    resultStream = open('resHol.dat', 'w')
    queries = np.zeros((500, spocs.shape[1]), dtype='float32')
    qids = np.zeros((500), dtype='int32')
    for i in xrange(500):
        queryName = str(100000 + i * 100) + '.jpg'
        qid = nameToId[queryName]
        queries[i,:] = spocs[qid,:]
        qids[i] = qid
    dist = distance.cdist(queries, spocs, 'euclidean')
    dist = dist.T
    for i in xrange(500):
        resultStream.write(idToName[qids[i]].strip() + ' ')
        answer = np.argsort(dist[:,i])
        for answer_id in xrange(len(answer)):
            resultStream.write(str(answer_id) + ' ' + idToName[answer[answer_id]].strip() + ' ')
        resultStream.write('\n')
    resultStream.close()
    output = commands.getoutput('python ./holidays_map.py ./resHol.dat')
    print output

