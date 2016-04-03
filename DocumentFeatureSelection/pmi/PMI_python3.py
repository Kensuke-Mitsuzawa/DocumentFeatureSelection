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

    def fit_transform(self, X, n_docs_distribution, n_jobs=1, verbose=False):
        """Main method of PMI class.
        """
        assert isinstance(X, csr_matrix)
        assert isinstance(n_docs_distribution, list)

        matrix_size = X.shape
        sample_range = list(range(0, matrix_size[0]))
        feature_range = list(range(0, matrix_size[1]))
        n_total_document = sum(n_docs_distribution)

        logger.debug(msg='Start calculating PMI with n(process)={}'.format(n_jobs))
        logger.debug(msg='size(input_matrix)={} * {}'.format(X.shape[0], X.shape[1]))

        pmi_score_csr_source = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self.docId_word_PMI)(
                X=X,
                n_docs_distribution=n_docs_distribution,
                feature_index=feature_index,
                sample_index=sample_index,
                n_total_doc=n_total_document,
                verbose=verbose
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

    def docId_word_PMI(self, X, n_docs_distribution, n_total_doc, feature_index, sample_index, verbose=False):
        """Calculate PMI score for fit_format()

        :param X:
        :param vocabulary:
        :param label_id:
        :param word:
        :param label:
        :return:
        """
        assert isinstance(X, csr_matrix)
        assert isinstance(n_docs_distribution, list)
        assert isinstance(feature_index, int)
        assert isinstance(sample_index, int)

        pmi_score = self.pmi(
            X=X,
            n_docs_distribution=n_docs_distribution,
            feature_index=feature_index,
            sample_index=sample_index,
            n_total_doc=n_total_doc,
            verbose=verbose
        )
        return sample_index, feature_index, pmi_score

    def pmi(self, X, n_docs_distribution, n_total_doc, feature_index, sample_index, verbose=False):
        """get PMI score for given feature & sample index

        :param X:
        :param feature_index:
        :param sample_index:
        :return:
        """
        assert isinstance(X, csr_matrix)
        assert isinstance(n_docs_distribution, list)
        assert isinstance(feature_index, int)
        assert isinstance(sample_index, int)

        matrix_size = X.shape
        sample_indexes = [i for i in range(0, matrix_size[0]) if i != sample_index]

        # n_11 is #docs having feature(i.e. word) in the specified index(label)
        n_11 = X[sample_index, feature_index]
        # n_01 is #docs NOT having feature in the specified index(label)
        n_01 = n_docs_distribution[sample_index] - n_11
        # n_10 is #docs having feature in NOT specified index(indexes except specified index)
        n_10 = X[sample_indexes, feature_index].sum()
        # n_00 is #docs NOT having feature in NOT specified index(indexes except specified index)
        n_00 = n_total_doc - (n_10 + n_docs_distribution[sample_index])

        if verbose:
            logging.debug('For feature_index:{} sample_index:{}'.format(feature_index, sample_index))
            logging.debug('n_11:{} n_01:{} n_10:{} n_00:{}'.format(
                n_11,
                n_01,
                n_10,
                n_00
            ))

        if n_11 == 0.0 or n_10 == 0.0 or n_01 == 0.0 or n_00 == 0.0:
            return 0
        else:
            temp1 = n_11/n_total_doc * math.log((n_total_doc*n_11)/((n_10+n_11)*(n_01+n_11)), 2)
            temp2 = n_01/n_total_doc * math.log((n_total_doc*n_01)/((n_00+n_01)*(n_01+n_11)), 2)
            temp3 = n_10/n_total_doc * math.log((n_total_doc*n_10)/((n_10+n_11)*(n_00+n_10)), 2)
            temp4 = n_00/n_total_doc * math.log((n_total_doc*n_00)/((n_00+n_01)*(n_00+n_10)), 2)
            score = temp1 + temp2 + temp3 + temp4

            if score < 0:
                raise Exception('score under 0 is detected. Something strange in Input matrix. Check your input matrix.')

            return score
