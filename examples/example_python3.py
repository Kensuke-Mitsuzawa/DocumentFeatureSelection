#! -*- coding: utf-8 -*-
__author__ = 'kensuke-mi'

from DocumentFeatureSelection import interface
import logging
import pprint
logger = logging.getLogger('sample usage')
logger.level = logging.DEBUG


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

pmi_scored_object = interface.run_feature_selection(
    input_dict=input_dict,
    method='pmi',
    ngram=1,
    n_jobs=5
)
pprint.pprint(pmi_scored_object.ScoreMatrix2ScoreDictionary())


soa_scored_object = interface.run_feature_selection(
    input_dict=input_dict,
    method='soa',
    ngram=1,
    n_jobs=5
)
pprint.pprint(soa_scored_object.ScoreMatrix2ScoreDictionary())


tf_idf_scored_object = interface.run_feature_selection(
    input_dict=input_dict,
    method='tf_idf',
    ngram=1,
    n_jobs=5
)
pprint.pprint(tf_idf_scored_object.ScoreMatrix2ScoreDictionary())



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


pmi_scored_object = interface.run_feature_selection(
    input_dict=input_dict_tuple_feature,
    method='pmi',
    n_jobs=5
)
pprint.pprint(pmi_scored_object.ScoreMatrix2ScoreDictionary())


soa_scored_object = interface.run_feature_selection(
    input_dict=input_dict_tuple_feature,
    method='soa',
    n_jobs=5
)
pprint.pprint(soa_scored_object.ScoreMatrix2ScoreDictionary())


tf_idf_scored_object = interface.run_feature_selection(
    input_dict=input_dict_tuple_feature,
    method='tf_idf',
    n_jobs=5
)
pprint.pprint(tf_idf_scored_object.ScoreMatrix2ScoreDictionary())