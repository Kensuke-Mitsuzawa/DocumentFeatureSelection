#! -*- coding: utf-8 -*-
__author__ = 'kensuke-mi'

from document_feature_selection import PMI
from document_feature_selection import data_converter
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


sparse_documen_matrix, label_id_dict, feature_dict = data_converter.convert_data(
    labeled_structure=input_dict,
    ngram=1,
    n_jobs=5
)

scored_sparse_matrix = PMI().fit_transform(X=sparse_documen_matrix)
assert isinstance(scored_sparse_matrix, csr_matrix)
pprint.pprint(scored_sparse_matrix.toarray())

# You can cut off words with PMI score=0
pmi_score_result = data_converter.get_weight_feature_dictionary(
    scored_matrix=scored_sparse_matrix,
    label_id_dict=label_id_dict,
    feature_id_dict=feature_dict
)
pprint.pprint(pmi_score_result)

# You can get PMI score for n-gram(n>1)