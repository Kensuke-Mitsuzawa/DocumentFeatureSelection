#! -*- coding: utf-8 -*-
__author__ = 'kensuke-mi'

from document_feature_selection.pmi import PMI
from scipy.sparse.csr import csr_matrix
import logging
logger = logging.Logger(level=logging.DEBUG, name='test')
import pprint

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

pmi_object = PMI(ngram=1, logger=logger)
pmi_feature_matrix = pmi_object.fit_transform(labeled_structure=input_dict)
assert isinstance(pmi_feature_matrix, csr_matrix)
pprint.pprint(pmi_feature_matrix.toarray())

# You can cut off words with PMI score=0
pmi_score_result = pmi_object.get_pmi_feature_dictionary(cut_zero=True)
pprint.pprint(pmi_score_result)

# You can get PMI score for n-gram(n>1)
bi_gram_pmi_object = PMI(ngram=2, logger=logger)
bi_gram_pmi_object.fit_transform(labeled_structure=input_dict)
pmi_score_result = bi_gram_pmi_object.get_pmi_feature_dictionary(cut_zero=True)
pprint.pprint(bi_gram_pmi_object.fit_transform(labeled_structure=input_dict).toarray())
pprint.pprint(pmi_score_result)