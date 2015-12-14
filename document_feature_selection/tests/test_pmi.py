# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import unittest
from document_feature_selection.pmi.test.test_pmi import *

__author__ = 'kensuke-mi'

__all__ = ['SampleTest']


class TestPMI(unittest.TestCase):
    def test_pmi_main(self):
        test_make_csr_main()

    def test_fit_transform(self):
        test_fit_transform_pmi()

    def test_class_get_score_objects(self):
        test_class_get_score_objects()


if __name__ == '__main__':
    unittest.main()
