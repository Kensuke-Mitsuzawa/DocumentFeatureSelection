#! -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from scipy.sparse import csr_matrix
from logging import getLogger, StreamHandler
import logging
import joblib
import math

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)

__author__ = 'kensuke-mi'


# TODO normzalized pmiの導入
# http://sucrose.hatenablog.com/entry/2014/12/02/235959

class PMI(object):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None, n_jobs=1):
        assert isinstance(X, csr_matrix)

        matrix_size = X.shape
        sample_range = list(range(0, matrix_size[0]-1))
        feature_range = list(range(0, matrix_size[1]-1))

        logger.debug(msg='Start calculating PMI with n(process)={}'.format(n_jobs))
        logger.debug(msg='size(input_matrix)={} * {}'.format(X.shape[0], X.shape[1]))

        pmi_score_csr_source = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self.docId_word_PMI)(
                X=X,
                feature_index=feature_index,
                sample_index=sample_index
            )
            for sample_index in sample_range
            for feature_index in feature_range
        )

        row_list = [t[0] for t in pmi_score_csr_source]
        col_list = [t[1] for t in pmi_score_csr_source]
        data_list = [t[2] for t in pmi_score_csr_source]

        pmi_featured_csr_matrix = csr_matrix((data_list, (row_list, col_list)),
                                             shape=(X.shape[0],
                                                    X.shape[1]))

        logging.debug(msg='End calculating PMI')

        return pmi_featured_csr_matrix

    def docId_word_PMI(self, X, feature_index, sample_index):
        """Calculate PMI score for fit_format()

        :param X:
        :param vocabulary:
        :param label_id:
        :param word:
        :param label:
        :return:
        """
        assert isinstance(X, csr_matrix)
        assert isinstance(feature_index, int)
        assert isinstance(sample_index, int)

        pmi_score = self.pmi(
            X=X,
            feature_index=feature_index,
            sample_index=sample_index
        )
        return sample_index, feature_index, pmi_score

    def pmi(self, X, feature_index, sample_index):
        """get PMI score for given feature & sample index

        :param X:
        :param feature_index:
        :param sample_index:
        :return:
        """
        assert isinstance(X, csr_matrix)
        assert isinstance(feature_index, int)
        assert isinstance(sample_index, int)

        matrix_size = X.shape
        sample_indexes = [i for i in range(0, matrix_size[0] - 1) if i != sample_index]
        feature_indexes = [i for i in range(0, matrix_size[1] - 1) if i != feature_index]

        n_01 = X[sample_index, feature_indexes].sum()
        n_11 = X[sample_index, feature_index].sum()
        n_10 = X[sample_indexes, feature_index].sum()
        n_00 = X.sum() - n_01 - n_11 - n_10
        N = X.sum()

        if n_11 == 0.0 or n_10 == 0.0 or n_01 == 0.0 or n_00 == 0.0:
            return 0
        else:
            temp1 = n_11/N * math.log((N*n_11)/((n_10+n_11)*(n_01+n_11)), 2)
            temp2 = n_01/N * math.log((N*n_01)/((n_00+n_01)*(n_01+n_11)), 2)
            temp3 = n_10/N * math.log((N*n_10)/((n_10+n_11)*(n_00+n_10)), 2)
            temp4 = n_00/N * math.log((N*n_00)/((n_00+n_01)*(n_00+n_10)), 2)
            score = temp1 + temp2 + temp3 + temp4
            return score
