#! -*- coding: utf-8 -*-
__author__ = 'kensuke-mi'

from DocumentFeatureSelection import interface
from DocumentFeatureSelection.init_logger import logger
import logging
import pprint

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