__author__ = 'kensuke-mi'

import sys
import unittest
python_version = sys.version_info


def suite():
    suite = unittest.TestSuite()
    if python_version >= (3, 0, 0):
        from .test_data_converter import TestDataConverter
        from .test_PMI_python3 import TestPmiPython3
        from .test_tf_idf import TestTfIdf
        from .test_soa_python3 import TestSoaPython3
        from .test_bns_python3 import TestBnsPython3
        suite.addTest(unittest.makeSuite(TestDataConverter))
        suite.addTest(unittest.makeSuite(TestPmiPython3))
        suite.addTest(unittest.makeSuite(TestTfIdf))
        suite.addTest(unittest.makeSuite(TestSoaPython3))
        suite.addTest(unittest.makeSuite(TestBnsPython3))
    else:
        pass


    return suite