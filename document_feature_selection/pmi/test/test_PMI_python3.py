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

        #self.csr_matrix, self.label_id, self.vocab_id, self.n_docs_distribution = data_converter_python3.convert_data(
        #        input_dict,
        #        ngram=1, n_jobs=5)
        self.data_csr_matrix = data_converter_python3.DataConverter().convert_data(
            labeled_documents=input_dict,
            ngram=1,
            n_jobs=5
        )

        print(self.data_csr_matrix)


    def test_normal_fit_transform(self):
        pmi_object = PMI_python3.PMI()
        scored_matrix = pmi_object.fit_transform(
            X=self.csr_matrix,
            n_jobs=1,
            n_docs_distribution=self.n_docs_distribution
        )
        assert isinstance(scored_matrix, csr_matrix)

    def test_multi_process_fit_transform(self):
        pmi_object = PMI_python3.PMI()
        scored_matrix = pmi_object.fit_transform(
            X=self.csr_matrix,
            n_jobs=5,
            n_docs_distribution=self.n_docs_distribution
        )
        assert isinstance(scored_matrix, csr_matrix)

    def test_output_result_pmi(self):
        pmi_object = PMI_python3.PMI()
        scored_matrix = pmi_object.fit_transform(
            X=self.csr_matrix,
            n_jobs=1,
            n_docs_distribution=self.n_docs_distribution
        )
        assert isinstance(scored_matrix, csr_matrix)

        # TODO PMI classの中で一発変換したいところ。inputからoutputまで通しで
        # TODO それって、別のクラスを立てた方がいいのでは？
        pmi_scored_dict = data_converter_python3.get_weight_feature_dictionary(
            scored_matrix=scored_matrix,
            label_id_dict=self.label_id,
            feature_id_dict=self.vocab_id,
            outformat='items',
            n_jobs=5
        )
        import pprint
        pprint.pprint(pmi_scored_dict)



if __name__ == '__main__':
    unittest.main()
