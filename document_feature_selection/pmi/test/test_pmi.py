#! -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from document_feature_selection.pmi.pmi_csr_matrix import make_pmi_matrix
from document_feature_selection.pmi.pmi import pmi_single_process_main
from scipy.sparse import csr_matrix
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
__author__ = 'kensuke-mi'


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


def test_make_csr_main():
    pmi_document_freq_csr_matrix, label_group_dict, vocabulary = make_pmi_matrix(input_dict, logger)
    assert isinstance(pmi_document_freq_csr_matrix, csr_matrix)
    assert isinstance(label_group_dict, dict)
    assert isinstance(vocabulary, dict)


def test_pmi_calc():
    pmi_document_freq_csr_matrix, label_group_dict, vocabulary = make_pmi_matrix(input_dict, logger)
    pmi_single_process_main(pmi_document_freq_csr_matrix, vocabulary, label_group_dict)
