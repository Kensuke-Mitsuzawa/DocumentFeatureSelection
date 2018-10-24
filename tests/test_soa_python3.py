from DocumentFeatureSelection.soa import soa_python3
from DocumentFeatureSelection.common import data_converter
from DocumentFeatureSelection.models import ScoredResultObject
import unittest


class TestSoaPython3(unittest.TestCase):
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

    def test_soa_with_term_freq(self):
        data_csr_matrix = data_converter.DataConverter().labeledMultiDocs2TermFreqMatrix(
            labeled_documents=self.input_dict,
            n_jobs=5
        )
        assert isinstance(data_csr_matrix, data_converter.DataCsrMatrix)
        label2id_dict = data_csr_matrix.label2id_dict
        csr_matrix_ = data_csr_matrix.csr_matrix_
        n_docs_distribution = data_csr_matrix.n_docs_distribution
        vocabulary = data_csr_matrix.vocabulary

        scored_matrix_term_freq = soa_python3.SOA().fit_transform(
            X=csr_matrix_,
            unit_distribution=n_docs_distribution,
            verbose=True
        )

        soa_scores_term_freq = ScoredResultObject(
            scored_matrix=scored_matrix_term_freq,
            label2id_dict=label2id_dict,
            feature2id_dict=vocabulary
        ).convert_score_matrix2score_record()
        self.assertTrue(isinstance(soa_scores_term_freq, list))

        #import pprint
        #print('term freq based soa')
        #pprint.pprint(soa_scores_term_freq)

    def test_soa_doc_freq(self):
        data_csr_matrix = data_converter.DataConverter().convert_multi_docs2document_frequency_matrix(
            labeled_documents=self.input_dict,
            n_jobs=5
        )
        assert isinstance(data_csr_matrix, data_converter.DataCsrMatrix)
        label2id_dict = data_csr_matrix.label2id_dict
        csr_matrix_ = data_csr_matrix.csr_matrix_
        n_docs_distribution = data_csr_matrix.n_docs_distribution
        vocabulary = data_csr_matrix.vocabulary

        scored_matrix_doc_freq = soa_python3.SOA().fit_transform(
            X=csr_matrix_,
            unit_distribution=n_docs_distribution,
            verbose=True
        )

        soa_scores_doc_freq = ScoredResultObject(
            scored_matrix=scored_matrix_doc_freq,
            label2id_dict=label2id_dict,
            feature2id_dict=vocabulary
        ).convert_score_matrix2score_record()
        self.assertTrue(isinstance(soa_scores_doc_freq, list))

        #import pprint
        #print('doc freq based soa')
        #pprint.pprint(soa_scores_doc_freq)


if __name__ == '__main__':
    unittest.main()
