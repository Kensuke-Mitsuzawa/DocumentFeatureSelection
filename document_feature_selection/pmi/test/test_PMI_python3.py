import unittest
from document_feature_selection.common import data_converter_python3
from document_feature_selection.pmi import PMI_python3
from scipy.sparse import csr_matrix


class TestPmiPython3(unittest.TestCase):
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
        self.csr_matrix, self.label_id, self.vocab_id = data_converter_python3.convert_data(input_dict,
                                                                                            ngram=1, n_jobs=5)

    def test_normal_fit_transform(self):
        pmi_object = PMI_python3.PMI()
        scored_matrix = pmi_object.fit_transform(
            X=self.csr_matrix,
            n_jobs=1
        )
        assert isinstance(scored_matrix, csr_matrix)

    def test_multi_process_fit_transform(self):
        pmi_object = PMI_python3.PMI()
        scored_matrix = pmi_object.fit_transform(
            X=self.csr_matrix,
            n_jobs=5
        )
        assert isinstance(scored_matrix, csr_matrix)


if __name__ == '__main__':
    unittest.main()
