#! -*- coding: utf-8 -*-
__author__ = 'kensuke-mi'

from DocumentFeatureSelection import interface
import logging
import pprint
logger = logging.getLogger('sample usage')
logger.level = logging.ERROR


# ======================================================================================================
# basic usage

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# tf idf

tf_idf_scored_object = interface.run_feature_selection(
    input_dict=input_dict,
    method='tf_idf',
    n_jobs=5
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# pmi
pmi_scored_object = interface.run_feature_selection(
    input_dict=input_dict,
    method='pmi',
    n_jobs=1,
    use_cython=False
)
pprint.pprint(pmi_scored_object.ScoreMatrix2ScoreDictionary())

# you can use cython version pmi also
# !Warning! The output value with "use_cython=True" is veeeery little different such as the 10th decimal place.
pmi_scored_object_cython = interface.run_feature_selection(
    input_dict=input_dict,
    method='pmi',
    n_jobs=1,
    use_cython=True
)
pprint.pprint(pmi_scored_object_cython.ScoreMatrix2ScoreDictionary())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# soa
soa_scored_object = interface.run_feature_selection(
    input_dict=input_dict,
    method='soa',
    n_jobs=5
)
pprint.pprint(soa_scored_object.ScoreMatrix2ScoreDictionary())

soa_scored_object_cython = interface.run_feature_selection(
    input_dict=input_dict,
    method='soa',
    n_jobs=1,
    use_cython=True
)
pprint.pprint(soa_scored_object_cython.ScoreMatrix2ScoreDictionary())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# bns
input_dict = {
    "label1": [
        ["I", "aa", "aa", "aa", "aa", "aa"],
        ["bb", "aa", "aa", "aa", "aa", "aa"],
        ["I", "aa", "hero", "some", "ok", "aa"]
    ],
    "label2": [
        ["bb", "bb", "bb"],
        ["bb", "bb", "bb"],
        ["hero", "ok", "bb"],
        ["hero", "cc", "bb"],
    ]
}
bns_scored_object = interface.run_feature_selection(
    input_dict=input_dict,
    method='bns',
    n_jobs=1
)
pprint.pprint(bns_scored_object.ScoreMatrix2ScoreDictionary())

bns_scored_object = interface.run_feature_selection(
    input_dict=input_dict,
    method='bns',
    use_cython=True
)
