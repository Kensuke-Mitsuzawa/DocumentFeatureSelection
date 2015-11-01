#! -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from scipy.sparse import csr_matrix
import logging
import math

__author__ = 'kensuke-mi'


def PMI(pmi_csr_matrix, word_index, label_index, label_indexes, feature_indexes):
    assert isinstance(pmi_csr_matrix, csr_matrix)
    assert isinstance(word_index, int)
    assert isinstance(label_index, int)

    label_indexes.pop(label_index)
    feature_indexes.pop(word_index)

    n_01 = pmi_csr_matrix[label_index, feature_indexes].sum()
    n_11 = pmi_csr_matrix[label_index, word_index].sum()
    n_10 = pmi_csr_matrix[label_indexes, word_index].sum()
    n_00 = pmi_csr_matrix.sum() - n_01 - n_11 - n_10
    N = pmi_csr_matrix.sum()

    if n_11 == 0.0 or n_10 == 0.0 or n_01 == 0.0 or n_00 == 0.0:
        return 0
    else:
        temp1 = n_11/N * math.log((N*n_11)/((n_10+n_11)*(n_01+n_11)), 2)
        temp2 = n_01/N * math.log((N*n_01)/((n_00+n_01)*(n_01+n_11)), 2)
        temp3 = n_10/N * math.log((N*n_10)/((n_10+n_11)*(n_00+n_10)), 2)
        temp4 = n_00/N * math.log((N*n_00)/((n_00+n_01)*(n_00+n_10)), 2)
        score = temp1 + temp2 + temp3 + temp4
        return score


def label_word_PMI(pmi_csr_matrix, vocabulary, label_id, word, label):
    assert isinstance(pmi_csr_matrix, csr_matrix)
    assert isinstance(vocabulary, dict)
    assert isinstance(label_id, dict)
    assert isinstance(word, (str, unicode))
    assert isinstance(label, (str, unicode))

    word_index = vocabulary[word]
    label_index = label_id[label]
    label_indexes = [i for i in range(0, pmi_csr_matrix.shape[0])]
    feature_indexes = [i for i in range(0, pmi_csr_matrix.shape[1])]

    pmi_score = PMI(pmi_csr_matrix, word_index, label_index, label_indexes, feature_indexes)
    return {'word': word, 'label': label, 'score': pmi_score}


def __conv_into_dict_format(pmi_word_score_items):
    out_format_structure = {}
    for item in pmi_word_score_items:
        if out_format_structure not in item['label']:
            out_format_structure[item['label']] = [{'word': item['word'], 'score': item['score']}]
        else:
            out_format_structure[item['label']].append({'word': item['word'], 'score': item['score']})
    return out_format_structure


def pmi_single_process_main(pmi_csr_matrix, vocabulary, label_id, logger, outformat='items', cut_zero=False):
    """This function returns PMI score between label and words.
    Input csr matrix must be 'document-frequency' matrix, where records #document that word appears in document set.
    [NOTE] This is not FREQUENCY.
    Ex.
    If 'iPhone' appears in 5 documents of 'IT' category document set, value must be 5.
    Even if 'iPhone' appears 10 time in 'IT' category document set, it does not matter.
    -----------------------------------------------
    :param pmi_csr_matrix document-frequency of input data:
    :param vocabulary vocabulary set dict of input data:
    :param label_id document id dict of input data:
    :param logger logging.Logger:
    :param outformat you can choose 'items' or 'dict':
    :return:
    """
    assert isinstance(logger, logging.Logger)
    assert isinstance(pmi_csr_matrix, csr_matrix)
    assert isinstance(vocabulary, dict)
    assert isinstance(label_id, dict)
    assert isinstance(cut_zero, bool)

    logging.debug(msg='Start calculating PMI')
    pmi_word_score_items = []
    for v in vocabulary.keys():
        for l in label_id.keys():
            pmi_word_score_items.append(label_word_PMI(pmi_csr_matrix, vocabulary, label_id, v, l))
    logging.debug(msg='End calculating PMI')

    pmi_word_score_items.sort(key=lambda x: x['score'], reverse=True)
    if cut_zero==True:
        pmi_word_score_items = [item for item in pmi_word_score_items if item['score'] > 0]

    if outformat=='dict':
        out_format_structure = __conv_into_dict_format(pmi_word_score_items)
    else:
        out_format_structure = pmi_word_score_items


    return out_format_structure
