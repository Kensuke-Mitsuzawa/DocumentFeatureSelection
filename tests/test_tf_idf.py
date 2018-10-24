#! -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from scipy.sparse import csr_matrix
from DocumentFeatureSelection.common import data_converter
from DocumentFeatureSelection.common.data_converter import DataCsrMatrix
from DocumentFeatureSelection.tf_idf import tf_idf
from DocumentFeatureSelection.models import ScoredResultObject
import logging
import unittest
import numpy
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
__author__ = 'kensuke-mi'


class TestTfIdf(unittest.TestCase):
    def setUp(self):
        input_dict = {
            "label_a": [
                ["I", "aa", "aa", "aa", "aa", "aa"],
                ["bb", "aa", "aa", "aa", "aa", "aa"],
                ["I", "aa", "hero", "some", "ok", "aa"]
            ],
            "label_b": [
                ["bb", "bb", "bb"],
                ["bb", "bb", "bb"],
                ["hero", "ok", "bb"],
                ["hero", "cc", "bb"],
            ],
            "label_c": [
                ["cc", "cc", "cc"],
                ["cc", "cc", "bb"],
                ["xx", "xx", "cc"],
                ["aa", "xx", "cc"],
            ]
        }

        tf_matrix = numpy.array(
            [
                [2, 12, 1, 0, 1, 1, 1, 0],
                [0, 0, 8, 1, 2, 1, 0, 0],
                [0, 1, 1, 7, 0, 0, 0, 3]
             ]
        )

        data_csr_matrix = data_converter.DataConverter().convert_multi_docs2document_frequency_matrix(
            labeled_documents=input_dict,
            n_jobs=-1
        )
        assert isinstance(data_csr_matrix, DataCsrMatrix)
        self.label2id_dict = data_csr_matrix.label2id_dict
        self.csr_matrix_ = data_csr_matrix.csr_matrix_
        self.n_docs_distribution = data_csr_matrix.n_docs_distribution
        self.vocabulary = data_csr_matrix.vocabulary

        numpy.array_equal(data_csr_matrix.csr_matrix_.toarray(), tf_matrix)

    def test_normal_fit_transform(self):
        tf_idf_weighted_matrix = tf_idf.TFIDF().fit_transform(
            X=self.csr_matrix_,
        )
        assert isinstance(tf_idf_weighted_matrix, csr_matrix)

    def test_output_result_pmi(self):
        tf_idf_weighted_matrix = tf_idf.TFIDF().fit_transform(
            X=self.csr_matrix_,
        )
        assert isinstance(tf_idf_weighted_matrix, csr_matrix)

        tf_idf_scored_dict = ScoredResultObject(
            scored_matrix=tf_idf_weighted_matrix,
            label2id_dict=self.label2id_dict,
            feature2id_dict=self.vocabulary,
        ).convert_score_matrix2score_record(outformat='items')
        self.assertTrue(isinstance(tf_idf_scored_dict, list))
        assert isinstance(tf_idf_scored_dict, list)


if __name__ == '__main__':
    unittest.main()

