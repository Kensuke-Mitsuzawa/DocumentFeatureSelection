from DocumentFeatureSelection.common import data_converter
from DocumentFeatureSelection.common.data_converter import DataCsrMatrix
from scipy.sparse import csr_matrix
import unittest
import numpy


class TestDataConverter(unittest.TestCase):
    def setUp(self):
        self.input_dict = {
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

        self.input_dict_complex_feature = {
            "label_a": [
                [['a', 'b'], ['b', 'c'], ['a', 'b', 'c']],
                [['a', 'b'], ['b', 'c']],
                [['a', 'b']]
            ],
            "label_b": [
                [['b', 'c'], ['c', 'd']],
                [['b', 'c']],
                [['b', 'c', 'd']],
                [['b', 'c']],
            ],
            "label_c": [
                [['c', 'd'], ['a', 'b']],
                [['b', 'c'], ['a', 'b', 'c']],
                [['b', 'c']]
            ]
        }


    def test_check_same_csr_matrix(self):
        """複数回の変換を実施して、同一のcsr_matrixになることを確認する
        """
        n_joblib_tasks = 2

        data_csr_matrix1 = data_converter.DataConverter().labeledMultiDocs2DocFreqMatrix(
            labeled_documents=self.input_dict,
            n_jobs=n_joblib_tasks
        )
        assert isinstance(data_csr_matrix1, DataCsrMatrix)
        csr_matrix_1 = data_csr_matrix1.csr_matrix_
        label_group_dict_1 = data_csr_matrix1.label2id_dict
        vocabulary_1 = data_csr_matrix1.vocabulary
        n_doc_distri_1 = data_csr_matrix1.n_docs_distribution
        n_term_distir_1 = data_csr_matrix1.n_term_freq_distribution
        dense_matrix_1 = csr_matrix_1.toarray()

        data_csr_matrix2 = data_converter.DataConverter().labeledMultiDocs2DocFreqMatrix(
            labeled_documents=self.input_dict,
            n_jobs=n_joblib_tasks
        )
        assert isinstance(data_csr_matrix2, DataCsrMatrix)
        csr_matrix_2 = data_csr_matrix2.csr_matrix_
        label_group_dict_2 = data_csr_matrix2.label2id_dict
        vocabulary_2 = data_csr_matrix2.vocabulary
        n_doc_distri_2 = data_csr_matrix2.n_docs_distribution
        n_term_distir_2 = data_csr_matrix2.n_term_freq_distribution
        dense_matrix_2 = data_csr_matrix2.csr_matrix_.toarray()

        data_csr_matrix3 = data_converter.DataConverter().labeledMultiDocs2DocFreqMatrix(
            labeled_documents=self.input_dict,
            n_jobs=n_joblib_tasks
        )
        assert isinstance(data_csr_matrix3, DataCsrMatrix)
        csr_matrix_3 = data_csr_matrix3.csr_matrix_
        label_group_dict_3 = data_csr_matrix3.label2id_dict
        vocabulary_3 = data_csr_matrix3.vocabulary
        n_doc_distri_3 = data_csr_matrix3.n_docs_distribution
        n_term_distir_3 = data_csr_matrix3.n_term_freq_distribution
        dense_matrix_3 = data_csr_matrix3.csr_matrix_.toarray()

        assert numpy.array_equal(dense_matrix_1, dense_matrix_2)
        assert numpy.array_equal(dense_matrix_2, dense_matrix_3)
        assert numpy.array_equal(dense_matrix_1, dense_matrix_3)

        assert vocabulary_1 == vocabulary_2
        assert vocabulary_2 == vocabulary_3
        assert vocabulary_1 == vocabulary_3



    def test_basic_convert_data(self):
        """checks it works of not when n_jobs=1, n_process=1

        data convert過程のミスが疑われるので、整合性のチェックをする

        :return:
        """

        csr_matrix_information = data_converter.DataConverter().labeledMultiDocs2DocFreqMatrix(
            labeled_documents=self.input_dict,
            n_jobs=5
        )
        assert isinstance(csr_matrix_information, DataCsrMatrix)
        csr_matrix_ = csr_matrix_information.csr_matrix_
        label_group_dict = csr_matrix_information.label2id_dict
        vocabulary = csr_matrix_information.vocabulary

        assert isinstance(csr_matrix_, csr_matrix)
        assert isinstance(label_group_dict, dict)
        assert isinstance(vocabulary, dict)

        n_correct_sample = 3
        n_correct_featute = 8

        assert csr_matrix_.shape[0] == n_correct_sample
        assert csr_matrix_.shape[1] == n_correct_featute

        dense_matrix_constructed_matrix = csr_matrix_.toarray()

        # vocaburary id of correct matrix is {'cc': 3, 'aa': 1, 'some': 6, 'xx': 7, 'I': 0, 'ok': 5, 'hero': 4, 'bb': 2}
        # label id of correct matrix is {'label_c': 2, 'label_a': 0, 'label_b': 1}
        correct_array_numpy = numpy.array(
            [[2, 3, 1, 0, 1, 1, 1, 0],
             [0, 0, 4, 1, 2, 1, 0, 0],
             [0, 1, 1, 4, 0, 0, 0, 2]
         ]).astype(numpy.int64)
        assert numpy.array_equal(correct_array_numpy, dense_matrix_constructed_matrix)



    def test_multi_process_convert_data(self):
        """checks if it works or not when n_process is more than 1

        :return:
        """

        data_csr_object = data_converter.DataConverter().labeledMultiDocs2DocFreqMatrix(
            labeled_documents=self.input_dict,
            n_jobs=5
        )

        assert isinstance(data_csr_object.csr_matrix_, csr_matrix)
        assert isinstance(data_csr_object.label2id_dict, dict)
        assert isinstance(data_csr_object.vocabulary, dict)

    def test_complex_feature_convertion(self):
        """"""
        csr_matrix_information = data_converter.DataConverter().labeledMultiDocs2DocFreqMatrix(
            labeled_documents=self.input_dict_complex_feature,
            n_jobs=1
        )
        assert isinstance(csr_matrix_information, DataCsrMatrix)
        csr_matrix_ = csr_matrix_information.csr_matrix_
        label_group_dict = csr_matrix_information.label2id_dict
        vocabulary = csr_matrix_information.vocabulary

        assert isinstance(csr_matrix_, csr_matrix)
        assert isinstance(label_group_dict, dict)
        assert isinstance(vocabulary, dict)

        n_correct_sample = 3
        n_correct_feature = 5

        assert csr_matrix_.shape[0] == n_correct_sample
        assert csr_matrix_.shape[1] == n_correct_feature

        dense_matrix_constructed_matrix = csr_matrix_.toarray()

        # vocaburary id of correct matrix is {'cc': 3, 'aa': 1, 'some': 6, 'xx': 7, 'I': 0, 'ok': 5, 'hero': 4, 'bb': 2}
        # label id of correct matrix is {'label_c': 2, 'label_a': 0, 'label_b': 1}
        correct_array_numpy = numpy.array(
            [[1.0, 3.0, 0.0, 2.0, 0.0],
             [0.0, 0.0, 1.0, 3.0, 1.0],
             [1.0, 1.0, 0.0, 2.0, 1.0],
         ]).astype(numpy.int64)
        assert numpy.array_equal(correct_array_numpy, dense_matrix_constructed_matrix)


if __name__ == '__main__':
    unittest.main()
