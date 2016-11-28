import unittest
from DocumentFeatureSelection import interface
from DocumentFeatureSelection.models import ScoredResultObject
import shelve
import os
import numpy

class TestInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.input_dict = {
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
        cls.method = ['pmi', 'tf_idf', 'soa']
        cls.bool_cython = [False, True]
        cls.joblib_range = range(0, 2)
        cls.path_shelve_file = './shelve'

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.path_shelve_file+'.db')

    def test_interface_shelve(self):
        shelve_obj = shelve.open(self.path_shelve_file)
        for key, value in self.input_dict.items(): shelve_obj[key] = value

        for method_name in self.method:
            for cython_flag in self.bool_cython:
                scored_result_shelve = interface.run_feature_selection(
                        input_dict=shelve_obj,
                        method=method_name, use_cython=cython_flag)  # type: ScoredResultObject
                self.assertIsInstance(scored_result_shelve, ScoredResultObject)
                self.assertIsInstance(scored_result_shelve.ScoreMatrix2ScoreDictionary(), list)

                # You check if result is same between data-source = shelve_obj and data-source = dict-object
                scored_result_dict = interface.run_feature_selection(
                        input_dict=self.input_dict,
                        method=method_name, use_cython=cython_flag)  # type: ScoredResultObject
                self.assertIsInstance(scored_result_dict, ScoredResultObject)
                self.assertIsInstance(scored_result_dict.ScoreMatrix2ScoreDictionary(), list)

                numpy.testing.assert_array_equal(scored_result_shelve.scored_matrix.toarray(), scored_result_dict.scored_matrix.toarray())

if __name__ == '__main__':
    unittest.main()
