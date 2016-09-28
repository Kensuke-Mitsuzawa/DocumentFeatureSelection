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
    ngram=1,
    n_jobs=5
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# pmi
pmi_scored_object = interface.run_feature_selection(
    input_dict=input_dict,
    method='pmi',
    ngram=1,
    n_jobs=1,
    use_cython=False
)
pprint.pprint(pmi_scored_object.ScoreMatrix2ScoreDictionary())

# you can use cython version pmi also
# !Warning! The output value with "use_cython=True" is veeeery little different such as the 10th decimal place.
pmi_scored_object_cython = interface.run_feature_selection(
    input_dict=input_dict,
    method='pmi',
    ngram=1,
    n_jobs=1,
    use_cython=True
)
pprint.pprint(pmi_scored_object_cython.ScoreMatrix2ScoreDictionary())

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# soa
soa_scored_object = interface.run_feature_selection(
    input_dict=input_dict,
    method='soa',
    ngram=1,
    n_jobs=5
)
pprint.pprint(soa_scored_object.ScoreMatrix2ScoreDictionary())

soa_scored_object_cython = interface.run_feature_selection(
    input_dict=input_dict,
    method='soa',
    ngram=1,
    n_jobs=1,
    use_cython=True
)
pprint.pprint(soa_scored_object_cython.ScoreMatrix2ScoreDictionary())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# bns
input_dict = {
    "positive": [
        ["I", "aa", "aa", "aa", "aa", "aa"],
        ["bb", "aa", "aa", "aa", "aa", "aa"],
        ["I", "aa", "hero", "some", "ok", "aa"]
    ],
    "negative": [
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


# ======================================================================================================
# expert usage
# you can put complex-structure-feature as feature.
# One feature is tuple of tuple. Concretely (("he", "N"), ("is", "V")) is one feature.
# You can NOT use ngram argument for expert input
input_dict_tuple_feature = {
    "label_a": [
        [ (("he", "N"), ("is", "V")), (("very", "ADV"), ("good", "ADJ")), (("guy", "N"),) ],
        [ (("you", "N"), ("are", "V")), (("very", "ADV"), ("awesome", "ADJ")), (("guy", "N"),) ],
        [ (("i", "N"), ("am", "V")), (("very", "ADV"), ("good", "ADJ")), (("guy", "N"),) ]
    ],
    "label_b": [
        [ (("she", "N"), ("is", "V")), (("very", "ADV"), ("good", "ADJ")), (("girl", "N"),) ],
        [ (("you", "N"), ("are", "V")), (("very", "ADV"), ("awesome", "ADJ")), (("girl", "N"),) ],
        [ (("she", "N"), ("is", "V")), (("very", "ADV"), ("good", "ADJ")), (("guy", "N"),) ]
    ]
}

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# tf idf
tf_idf_scored_object = interface.run_feature_selection(
    input_dict=input_dict_tuple_feature,
    method='tf_idf',
    n_jobs=5
)
pprint.pprint(tf_idf_scored_object.ScoreMatrix2ScoreDictionary())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# pmi
pmi_scored_object = interface.run_feature_selection(
    input_dict=input_dict_tuple_feature,
    method='pmi',
    n_jobs=5
)
pprint.pprint(pmi_scored_object.ScoreMatrix2ScoreDictionary())


pmi_scored_object_cython = interface.run_feature_selection(
    input_dict=input_dict_tuple_feature,
    method='pmi',
    n_jobs=1,
    use_cython=True
)
pprint.pprint(pmi_scored_object_cython.ScoreMatrix2ScoreDictionary())


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# soa
soa_scored_object = interface.run_feature_selection(
    input_dict=input_dict_tuple_feature,
    method='soa',
    n_jobs=5
)
pprint.pprint(soa_scored_object.ScoreMatrix2ScoreDictionary())


soa_scored_object_cython = interface.run_feature_selection(
    input_dict=input_dict_tuple_feature,
    method='soa',
    n_jobs=1,
    use_cython=True
)
pprint.pprint(soa_scored_object_cython.ScoreMatrix2ScoreDictionary())




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# bns
input_dict_tuple_feature = {
    "positive": [
        [ (("he", "N"), ("is", "V")), (("very", "ADV"), ("good", "ADJ")), (("guy", "N"),) ],
        [ (("you", "N"), ("are", "V")), (("very", "ADV"), ("awesome", "ADJ")), (("guy", "N"),) ],
        [ (("i", "N"), ("am", "V")), (("very", "ADV"), ("good", "ADJ")), (("guy", "N"),) ]
    ],
    "negative": [
        [ (("she", "N"), ("is", "V")), (("very", "ADV"), ("good", "ADJ")), (("girl", "N"),) ],
        [ (("you", "N"), ("are", "V")), (("very", "ADV"), ("awesome", "ADJ")), (("girl", "N"),) ],
        [ (("she", "N"), ("is", "V")), (("very", "ADV"), ("good", "ADJ")), (("guy", "N"),) ]
    ]
}


bns_scored_object = interface.run_feature_selection(
    input_dict=input_dict_tuple_feature,
    method='bns',
    n_jobs=5
)
pprint.pprint(bns_scored_object.ScoreMatrix2ScoreDictionary())