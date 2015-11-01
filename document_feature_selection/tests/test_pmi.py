# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import unittest
from document_feature_selection.pmi.test.test_pmi import *

__author__ = 'kensuke-mi'

__all__ = ['SampleTest']


class TestPMI(unittest.TestCase):
    def test_pmi_main(self):
        test_make_csr_main()

    def test_pmi_calc(self):
        test_pmi_calc()


if __name__ == '__main__':
    unittest.main()
