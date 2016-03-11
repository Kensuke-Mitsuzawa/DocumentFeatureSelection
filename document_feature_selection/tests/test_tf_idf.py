#! -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from document_feature_selection.pmi.UNUSED_pmi_csr_matrix import make_pmi_matrix
from document_feature_selection.pmi.UNUSED_pmi import pmi_single_process_main
from document_feature_selection.pmi.UNUSED_pmi import fit_format
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


    pmi_document_freq_csr_matrix, label_group_dict, vocabulary = make_pmi_matrix(input_dict, logger, ngram=2)
    assert isinstance(pmi_document_freq_csr_matrix, csr_matrix)
    assert isinstance(label_group_dict, dict)
    assert isinstance(vocabulary, dict)


def test_pmi_calc():
    pmi_document_freq_csr_matrix, label_group_dict, vocabulary = make_pmi_matrix(input_dict, logger, ngram=2)
    pmi_score_objects = pmi_single_process_main(pmi_document_freq_csr_matrix, vocabulary, label_group_dict, logger, cut_zero=True)
    assert isinstance(pmi_score_objects, list)
    assert isinstance(pmi_score_objects[0], dict)
    assert pmi_score_objects[0].has_key('score')
    assert pmi_score_objects[0].has_key('word')
    assert pmi_score_objects[0].has_key('label')
    print(pmi_score_objects)


def test_fit_transform_pmi():
    pmi_document_freq_csr_matrix, label_group_dict, vocabulary = make_pmi_matrix(input_dict, logger, ngram=2)
    pmi_featured_csr_matrix = fit_format(pmi_document_freq_csr_matrix, vocabulary, label_group_dict)
    assert isinstance(pmi_featured_csr_matrix, csr_matrix)
    print(pmi_featured_csr_matrix.toarray())