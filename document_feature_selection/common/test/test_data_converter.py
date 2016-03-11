from document_feature_selection.common import data_converter_python3
from document_feature_selection.pmi import PMI_python3
from scipy.sparse import csr_matrix
import unittest
import logging


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

    def test_basic_convert_data(self):
        """checks it works of not when n_jobs=1, n_process=1

        :return:
        """

        csr_matrix_, label_group_dict, vocabulary = data_converter_python3.convert_data(
            labeled_structure=self.input_dict,
            ngram=1,
            n_jobs=1
        )

        assert isinstance(csr_matrix_, csr_matrix)
        assert isinstance(label_group_dict, dict)
        assert isinstance(vocabulary, dict)

    def test_multi_process_convert_data(self):
        """checks if it works or not when n_process is more than 1

        :return:
        """

        csr_matrix_, label_group_dict, vocabulary = data_converter_python3.convert_data(
            labeled_structure=self.input_dict,
            ngram=1,
            n_jobs=5
        )

        assert isinstance(csr_matrix_, csr_matrix)
        assert isinstance(label_group_dict, dict)
        assert isinstance(vocabulary, dict)

    def test_n_gram_multi_process_convert_data(self):
        """checks if it works or not when n_process is more than 1, and 3-gram

        :return:
        """

        csr_matrix_, label_group_dict, vocabulary = data_converter_python3.convert_data(
            labeled_structure=self.input_dict,
            ngram=3,
            n_jobs=5
        )

        assert isinstance(csr_matrix_, csr_matrix)
        assert isinstance(label_group_dict, dict)
        assert isinstance(vocabulary, dict)


    def test_get_pmi_feature_dictionary(self):
        """checks if it works or not, that getting scored dictionary object from scored_matrix

        :return:
        """
        csr_matrix_, label_group_dict, vocabulary = data_converter_python3.convert_data(
            labeled_structure=self.input_dict,
            ngram=1,
            n_jobs=5
        )

        assert isinstance(csr_matrix_, csr_matrix)
        assert isinstance(label_group_dict, dict)
        assert isinstance(vocabulary, dict)

        pmi_scored_matrix = PMI_python3.PMI().fit_transform(X=csr_matrix_, n_jobs=5)

        # main part of test
        # when sort is True, cut_zero is True, outformat is dict
        pmi_scored_dictionary_objects = data_converter_python3.get_weight_feature_dictionary(
            scored_matrix=pmi_scored_matrix,
            label_id_dict=label_group_dict,
            feature_id_dict=vocabulary,
            outformat='dict',
            sort_desc=True,
            n_jobs=5
        )
        assert isinstance(pmi_scored_dictionary_objects, dict)
        logging.debug(pmi_scored_dictionary_objects)

        # when sort is True, cut_zero is True, outformat is items
        pmi_scored_dictionary_objects = data_converter_python3.get_weight_feature_dictionary(
            scored_matrix=pmi_scored_matrix,
            label_id_dict=label_group_dict,
            feature_id_dict=vocabulary,
            outformat='items',
            sort_desc=True,
            n_jobs=5
        )
        assert isinstance(pmi_scored_dictionary_objects, list)
        for d in pmi_scored_dictionary_objects: assert isinstance(d, dict)
        logging.debug(pmi_scored_dictionary_objects)

        # when sort is True, cut_zero is False, outformat is dict
        pmi_scored_dictionary_objects = data_converter_python3.get_weight_feature_dictionary(
            scored_matrix=pmi_scored_matrix,
            label_id_dict=label_group_dict,
            feature_id_dict=vocabulary,
            outformat='dict',
            sort_desc=True,
            n_jobs=5
        )
        assert isinstance(pmi_scored_dictionary_objects, dict)
        logging.debug(pmi_scored_dictionary_objects)

        # when sort is True, cut_zero is False, outformat is items
        pmi_scored_dictionary_objects = data_converter_python3.get_weight_feature_dictionary(
            scored_matrix=pmi_scored_matrix,
            label_id_dict=label_group_dict,
            feature_id_dict=vocabulary,
            outformat='items',
            sort_desc=True,
            n_jobs=5
        )
        assert isinstance(pmi_scored_dictionary_objects, list)
        for d in pmi_scored_dictionary_objects: assert isinstance(d, dict)
        logging.debug(pmi_scored_dictionary_objects)


if __name__ == '__main__':
    unittest.main()
