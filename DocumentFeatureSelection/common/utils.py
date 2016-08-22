#! -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from scipy.sparse.csr import csr_matrix
from numpy import ndarray, int32, int64
from DocumentFeatureSelection import init_logger
import logging
import collections
import joblib
import sys
python_version = sys.version_info
logger = init_logger.init_logger(logging.getLogger(init_logger.LOGGER_NAME))

__author__ = 'kensuke-mi'

ROW_COL_VAL = collections.namedtuple('ROW_COL_VAL', 'row col val')


def flatten(lis):
     for item in lis:
         if isinstance(item, list) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:
             yield item


def __conv_into_dict_format(pmi_word_score_items):
    out_format_structure = {}
    for item in pmi_word_score_items:
        if out_format_structure not in item['label']:
            out_format_structure[item['label']] = [{'word': item['word'], 'score': item['score']}]
        else:
            out_format_structure[item['label']].append({'word': item['word'], 'score': item['score']})
    return out_format_structure


def extract_from_csr_matrix(weight_csr_matrix, vocabulary, label_id, row_id, col_id):
    assert isinstance(weight_csr_matrix, csr_matrix)
    assert isinstance(vocabulary, dict)
    assert isinstance(label_id, dict)


def __get_value_index(row_index, column_index, weight_csr_matrix, verbose=False):
    assert isinstance(row_index, (int, int32))
    assert isinstance(column_index, (int, int32))
    assert isinstance(weight_csr_matrix, csr_matrix)

    value = weight_csr_matrix[row_index, column_index]

    return value


def make_non_zero_information(weight_csr_matrix):
    """Construct Tuple of matrix value. Return value is array of ROW_COL_VAL namedtuple.

    :param weight_csr_matrix:
    :return:
    """
    assert isinstance(weight_csr_matrix, csr_matrix)

    row_col_index_array = weight_csr_matrix.nonzero()
    row_indexes = row_col_index_array[0]
    column_indexes = row_col_index_array[1]
    assert len(row_indexes) == len(column_indexes)

    value_index_items = [
        ROW_COL_VAL(
            row_indexes[i],
            column_indexes[i],
            __get_value_index(row_indexes[i], column_indexes[i], weight_csr_matrix)
        )
        for i
        in range(0, len(row_indexes))]

    return value_index_items


def get_label(row_col_val_tuple, label_id):
    assert isinstance(row_col_val_tuple, ROW_COL_VAL)
    assert isinstance(label_id, dict)

    return label_id[row_col_val_tuple.row]


def get_word(row_col_val_tuple, vocabulary):
    assert isinstance(row_col_val_tuple, ROW_COL_VAL)
    assert isinstance(vocabulary, dict)

    return vocabulary[row_col_val_tuple.col]


def SUB_FUNC_feature_extraction(row_col_val_tuple, id2label, id2vocab):
    """This function returns PMI score between label and words.

    Input csr matrix must be 'document-frequency' matrix, where records #document that word appears in document set.
    [NOTE] This is not TERM-FREQUENCY.

    For example,
    If 'iPhone' appears in 5 documents of 'IT' category document set, value must be 5.
    Even if 10 'iPhone' words in 'IT' category document set, value is still 5.

    """
    assert isinstance(row_col_val_tuple, tuple)
    assert isinstance(row_col_val_tuple, ROW_COL_VAL)

    return {
        'score': row_col_val_tuple.val,
        'label': get_label(row_col_val_tuple, id2label),
        'word': get_word(row_col_val_tuple, id2vocab)
    }


def get_feature_dictionary(weighted_matrix, vocabulary, label_group_dict, n_jobs=1):
    """Get dictionary structure of PMI featured scores.

    You can choose 'dict' or 'items' for ```outformat``` parameter.

    If outformat='dict', you get

    >>> {label_name:
            {
                feature: score
            }
        }

    Else if outformat='items', you get

    >>> [
        {
            feature: score
        }
        ]


    :param string outformat: format type of output dictionary. You can choose 'items' or 'dict'
    :param bool cut_zero: return all result or not. If cut_zero = True, the method cuts zero features.
    """
    assert isinstance(weighted_matrix, csr_matrix)
    assert isinstance(vocabulary, dict)
    assert isinstance(label_group_dict, dict)
    assert isinstance(n_jobs, int)

    logger.debug(msg='Start making scored dictionary object from scored matrix')
    logger.debug(msg='Input matrix size= {} * {}'.format(weighted_matrix.shape[0], weighted_matrix.shape[1]))

    value_index_items = make_non_zero_information(weighted_matrix)
    id2label = {id:label for label, id in label_group_dict.items()}
    if python_version > (3, 0, 0):
        id2vocab = {id:voc for voc, id in vocabulary.items()}
    else:
        id2vocab = {id:voc for voc, id in vocabulary.viewitems()}

    score_objects = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(SUB_FUNC_feature_extraction)(
            row_col_val_tuple,
            id2label,
            id2vocab
        )
        for row_col_val_tuple in value_index_items
    )

    logger.debug(msg='Finished making scored dictionary')


    return score_objects


