#! -*- coding: utf-8 -*-
__author__ = 'kensuke-mi'

from document_feature_selection.pmi import get_pmi_score
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

pmi_score_result = get_pmi_score(labeled_structure=input_dict, logger=logger, ngram=1)
pprint.pprint(pmi_score_result)

# You can cut off words with PMI score=0
pmi_score_result = get_pmi_score(labeled_structure=input_dict, logger=logger, ngram=1, cut_zero=True)
pprint.pprint(pmi_score_result)

# You can get PMI score for n-gram(n>1)
pmi_score_result = get_pmi_score(labeled_structure=input_dict, logger=logger, ngram=2, cut_zero=True)
pprint.pprint(pmi_score_result)