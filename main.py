# -*- coding:utf-8 -*-

from pyAudioAnalysis import audioFeatureExtraction
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.mixture import GaussianMixture
from optparse import OptionParser
from scipy import spatial
from scipy.stats import entropy
import scipy.io as sio
import numpy as np
import cPickle
from time import sleep
import sys
from bhattacharyyaGaussian import *


def ZCR(signal, Fs, Win, Step):
    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = np.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (np.abs(signal)).max()
    signal = (signal - DC) / (MAX + 0.0000000001)

    N = len(signal)  # total number of samples
    curPos = 0
    countFrames = 0

    stFeatures_ZCR = []

    while (curPos + Win - 1 < N):
        countFrames += 1
        x = signal[curPos: curPos + Win]
        curPos = curPos + Step
        stFeatures_ZCR.append(audioFeatureExtraction.stZCR(x))

    return stFeatures_ZCR


def main():
    '''
    db = cPickle.load(open('casia_db_array.p', 'rb'))

    # extract features

    win_size = 0.04
    step = 0.01
    Feature_CZRate = []
    for (x, Fs) in db:
        F = ZCR(x, Fs, win_size * Fs, step * Fs)
        Feature_CZRate.append(list(
            (np.mean(F),
             np.std(F),
             np.max(F),
             np.min(F))))
    Feature_CZRate = np.array(Feature_CZRate)
    cPickle.dump(Feature_CZRate, open('Feature_CZRate.p', 'wb'))
    '''
    Feature_CZRate = cPickle.load(open('Feature_CZRate.p', 'rb'))

    estimator = GaussianMixture(n_components=8, covariance_type='diag',
                                max_iter=100)
    estimator.fit(Feature_CZRate)
    weights = estimator.weights_
    means = estimator.means_
    covariances = estimator.covariances_
    BhaDistance = np.mat(np.zeros([8, 8]))
    for i in range(8):
        for j in range(8):
            BhaDistance[i, j] = calBhaDistanceGaussian(means[i], means[j],
                                                       np.diag(covariances[i]), np.diag(covariances[j]))
    BhaDistance = np.array(BhaDistance)
    np.save('BhaDistance.npy',BhaDistance)


if __name__ == '__main__':
    main()
