#! -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse.csr import csr_matrix
from numpy import ndarray
__author__ = 'kensuke-mi'


def fit_transform(csr_matrix, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
    assert isinstance(csr_matrix, (csr_matrix, ndarray))

    tf_idf_generator = TfidfTransformer(
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf
    )
    if isinstance(csr_matrix, csr_matrix):
        feat_matrix = csr_matrix.toarray()
    else:
        feat_matrix = csr_matrix

    tf_idf_weight_matrix = tf_idf_generator.fit_transform(
        X=feat_matrix
    )
    assert isinstance(tf_idf_weight_matrix, ndarray)