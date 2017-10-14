import math
import scipy
from cpython cimport bool
from scipy.stats import norm
cimport numpy
import numpy


cdef float bns(
    numpy.ndarray[numpy.float64_t, ndim=2] X,
    numpy.ndarray[numpy.int64_t, ndim=1] unit_distribution,
    int feature_index,
    int sample_index, 
    int true_index,
    bool verbose):
    # X is either of term-frequency matrix per label or document-frequency per label

    cdef int false_index
    if true_index == 0:
        false_index = 1
    elif true_index == 1:
        false_index = 0
    else:
        raise Exception('true index must be either of 0 or 1')

    # trueラベルで出現した回数
    # tp is frequency of features in the specified positive label
    cdef float tp = X[true_index, feature_index]
    # trueラベルで出現しなかった回数
    # fp is frequency of NON-features(expect specified feature) in the specified positive label
    cdef float fp = unit_distribution[true_index] - tp

    # negativeラベルで出現した回数
    # fn is frequency of features in the specified negative label
    cdef float fn = X[false_index, feature_index]
    # negativeラベルで出現しなかった回数
    # fp is frequency of NON-features(expect specified feature) in the specified negative label
    cdef float tn = unit_distribution[false_index] - fn

    if tn < 0.0:
        print('Something wrong')

    cdef float pos = tp + fn
    cdef float neg = fp + tn

    cdef float tpr = tp / pos
    cdef float fpr = fp / neg

    if verbose:
        print('For feature_index:{} sample_index:{}'.format(feature_index, sample_index))
        print('tp:{} fp:{} fn:{} tn:{} pos:{} neg:{} tpr:{} fpr:{}'.format(
            tp,
            fp,
            fn,
            tn,
            pos,
            neg,
            tpr,
            fpr
        ))
    cdef float bns_score = numpy.abs(norm.ppf(tpr) - norm.ppf(fpr))
    # cdef float bns_score = numpy.abs(norm.ppf(norm.cdf(tpr)) - norm.ppf(norm.cdf(fpr)))
    return bns_score


def main(X,
        numpy.ndarray[numpy.int64_t, ndim=1] unit_distribution,
        sample_range,
        feature_range,
        int true_index,
        bool verbose=False):
    """What you can do
    - calculate BNS score based on given data.
    - The function returns list of tuple, whose element is (sample_index, feature_index, score)
    - Your input matrix should be numpy.ndarray or scipy.sparse.csr_matrix. The matrix should represent document-frequency of each feature.
    """
    if isinstance(X, scipy.sparse.csr_matrix):
        X = X.toarray()

    cdef int sample_index, feature_index
    soa_score_csr_source = [
        (
            sample_index,
            feature_index,
            bns(X, unit_distribution, feature_index, sample_index, true_index, verbose)
         )
        for sample_index in sample_range
        for feature_index in feature_range
    ]

    return soa_score_csr_source