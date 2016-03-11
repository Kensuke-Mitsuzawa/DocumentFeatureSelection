from document_feature_selection.common import data_converter_python3
from scipy.sparse import csr_matrix
import unittest


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



if __name__ == '__main__':
    unittest.main()
