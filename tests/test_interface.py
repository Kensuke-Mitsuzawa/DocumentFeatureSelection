import unittest
from DocumentFeatureSelection import interface
from DocumentFeatureSelection.models import ScoredResultObject
from DocumentFeatureSelection.models import PersistentDict
from sqlitedict import SqliteDict
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
        cls.bool_cython = [True, False]
        cls.is_use_cache = [True, False]
        cls.is_use_memmap = [True, False]
        cls.joblib_range = range(0, 2)
        cls.path_shelve_file = './shelve'
        cls.path_sqlite3_persistent = './temp_db.sqlite3'

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.path_sqlite3_persistent)

    def test_interface_shelve(self):
        """パラメタ条件を組み合わせてテストを実行する　
        - cythonモード使う or not
        - cacheモード使う or not
        - memmapモード使う or not
        """
        shelve_obj = PersistentDict(self.path_shelve_file, 'c', 'json')
        for key, value in self.input_dict.items(): shelve_obj[key] = value

        sqlite3_dict_obj = SqliteDict(filename=self.path_sqlite3_persistent, autocommit=True)
        for key, value in self.input_dict.items(): sqlite3_dict_obj[key] = value

        for method_name in self.method:
            for cython_flag in self.bool_cython:
                for cache_flag in self.is_use_cache:
                    for memmap_flag in self.is_use_memmap:
                        scored_result_persisted = interface.run_feature_selection(
                            input_dict=shelve_obj,
                            method=method_name,
                            use_cython=cython_flag,
                            is_use_cache=cache_flag,
                            is_use_memmap=memmap_flag
                        )  # type: ScoredResultObject
                        self.assertIsInstance(scored_result_persisted, ScoredResultObject)
                        self.assertIsInstance(scored_result_persisted.ScoreMatrix2ScoreDictionary(), list)

                        scored_result_sqlite3_persisted = interface.run_feature_selection(
                            input_dict=sqlite3_dict_obj,
                            method=method_name, use_cython=cython_flag, is_use_cache=cache_flag)  # type: ScoredResultObject
                        self.assertIsInstance(scored_result_sqlite3_persisted, ScoredResultObject)
                        self.assertIsInstance(scored_result_sqlite3_persisted.ScoreMatrix2ScoreDictionary(), list)

                        # You check if result is same between data-source = shelve_obj and data-source = dict-object
                        scored_result_dict = interface.run_feature_selection(
                            input_dict=self.input_dict,
                            method=method_name, use_cython=cython_flag, is_use_cache=cache_flag)  # type: ScoredResultObject
                        self.assertIsInstance(scored_result_dict, ScoredResultObject)
                        self.assertIsInstance(scored_result_dict.ScoreMatrix2ScoreDictionary(), list)

                        numpy.testing.assert_array_equal(scored_result_persisted.scored_matrix.toarray(),
                                                         scored_result_dict.scored_matrix.toarray())
                        numpy.testing.assert_array_equal(scored_result_sqlite3_persisted.scored_matrix.toarray(),
                                                         scored_result_dict.scored_matrix.toarray())


if __name__ == '__main__':
    unittest.main()
