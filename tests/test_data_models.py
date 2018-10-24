from DocumentFeatureSelection.common import data_converter
from DocumentFeatureSelection.pmi import PMI_python3
from DocumentFeatureSelection.models import ScoredResultObject
from scipy.sparse import csr_matrix
import unittest
import numpy
import logging


class TestDataModels(unittest.TestCase):
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


    def test_get_pmi_feature_dictionary(self):
        """checks if it works or not, that getting scored dictionary object from scored_matrix

        :return:
        """
        data_csr_object = data_converter.DataConverter().labeledMultiDocs2DocFreqMatrix(
            labeled_documents=self.input_dict,
            ngram=1,
            n_jobs=5
        )

        assert isinstance(data_csr_object.csr_matrix_, csr_matrix)
        assert isinstance(data_csr_object.label2id_dict, dict)
        assert isinstance(data_csr_object.vocabulary, dict)

        pmi_scored_matrix = PMI_python3.PMI().fit_transform(X=data_csr_object.csr_matrix_, n_jobs=5,
                                                            n_docs_distribution=data_csr_object.n_docs_distribution)

        # main part of test
        # when sort is True, cut_zero is True, outformat is dict
        pmi_scored_dictionary_objects = ScoredResultObject(
            scored_matrix=pmi_scored_matrix,
            label2id_dict=data_csr_object.label2id_dict,
            feature2id_dict=data_csr_object.vocabulary
        ).ScoreMatrix2ScoreDictionary(
            outformat='dict',
            sort_desc=True,
            n_jobs=5
        )
        assert isinstance(pmi_scored_dictionary_objects, dict)
        logging.debug(pmi_scored_dictionary_objects)

        # when sort is True, cut_zero is True, outformat is items
        pmi_scored_dictionary_objects = ScoredResultObject(
            scored_matrix=pmi_scored_matrix,
            label2id_dict=data_csr_object.label2id_dict,
            feature2id_dict=data_csr_object.vocabulary).ScoreMatrix2ScoreDictionary(
            outformat='items',
            sort_desc=True,
            n_jobs=5
        )
        assert isinstance(pmi_scored_dictionary_objects, list)
        for d in pmi_scored_dictionary_objects:
            assert isinstance(d, dict)

        # when sort is True, cut_zero is False, outformat is dict
        pmi_scored_dictionary_objects = ScoredResultObject(
            scored_matrix=pmi_scored_matrix,
            label2id_dict=data_csr_object.label2id_dict,
            feature2id_dict=data_csr_object.vocabulary
        ).ScoreMatrix2ScoreDictionary(
            outformat='dict',
            sort_desc=True,
            n_jobs=5
        )
        assert isinstance(pmi_scored_dictionary_objects, dict)
        logging.debug(pmi_scored_dictionary_objects)

        # when sort is True, cut_zero is False, outformat is items
        pmi_scored_dictionary_objects = ScoredResultObject(
            scored_matrix=pmi_scored_matrix,
            label2id_dict=data_csr_object.label2id_dict,
            feature2id_dict=data_csr_object.vocabulary
        ).ScoreMatrix2ScoreDictionary(
            outformat='items',
            sort_desc=True,
            n_jobs=5
        )
        assert isinstance(pmi_scored_dictionary_objects, list)
        for d in pmi_scored_dictionary_objects:
            assert isinstance(d, dict)


if __name__ == '__main__':
    unittest.main()