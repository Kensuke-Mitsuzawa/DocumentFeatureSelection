from logging import getLogger, StreamHandler
import joblib
import sys
import logging
import numpy
from typing import List, Tuple, Dict
from scipy.sparse import csr_matrix

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)

python_version = sys.version_info
__author__ = 'kensuke-mi'


class PosTuple(object):
    __slots__ = ['doc_id', 'word_id', 'document_frequency']
    def __init__(self, doc_id, word_id, document_frequency):
        self.doc_id = doc_id
        self.word_id = word_id
        self.document_frequency = document_frequency


PARAM_JOBLIB_BACKEND = ['multiprocessing', 'threading']

def get_data_col_row_values(doc_id:int, word:int, doc_freq:int, vocaburary:numpy.ndarray)->numpy.array:
    """* what you can do
     - You get array of [document_id, feature_id, value(frequency)]
    """
    assert isinstance(vocaburary, numpy.ndarray)
    col_element = vocaburary[numpy.where(vocaburary['key']==word)]
    assert len(col_element) == 1
    col_value = col_element[0]['value']
    # df value is word frequency in documents
    df_value = doc_freq

    return numpy.array([doc_id, col_value, df_value])

def SUB_FUNC_make_value_pairs(doc_id:int, doc_freq_obj:numpy.ndarray, vocabulary:numpy.ndarray)->numpy.ndarray:

    value_pairs = numpy.array([
        get_data_col_row_values(doc_id=doc_id, word=key_value_tuple['key'], doc_freq=key_value_tuple['value'], vocaburary=vocabulary)
        for key_value_tuple
        in doc_freq_obj])

    return value_pairs


def make_csr_list(value_position_list:List[numpy.ndarray])->Tuple[List[int], List[int], List[int]]:
    data = []
    row = []
    col = []
    for position_tuple in value_position_list:
        row.append(position_tuple[0])
        col.append(position_tuple[1])
        data.append(position_tuple[2])

    return row, col, data


def preprocess_csr_matrix(feature_frequency, vocabulary, n_jobs:int, joblib_backend:str='Parallel'):
    """This function makes information to make csr matrix. Data-list/Row-list/Col-list

    :param feature_frequency list: list having dictionary of {feature: frequency}
    :param label2id_dict dict: dictionary of {feature: feature_id}
    :return: tuple having lists to construct csr matrix
    :rtype tuple:

    Example,

    feature_frequency is
    >>> [{'some': 1, 'bb': 1, 'hero': 1, 'aa': 3, 'I': 2, 'ok': 1}, {'cc': 1, 'bb': 4, 'ok': 1, 'hero': 2}, {'cc': 4, 'bb': 1, 'xx': 2, 'aa': 1}]

    vocaburary is
    >>> {'some': 6, 'bb': 2, 'xx': 7, 'hero': 4, 'aa': 1, 'cc': 3, 'I': 0, 'ok': 5}

    return value is
    >>> ([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], [0, 1, 2, 4, 5, 6, 2, 3, 4, 5, 1, 2, 3, 7], [2, 3, 1, 1, 1, 1, 4, 1, 2, 1, 1, 1, 4, 2])

    """
    if not joblib_backend in PARAM_JOBLIB_BACKEND:
        assert Exception('joblib_backend parameter must be either of {}. However your input is {}.'.format(PARAM_JOBLIB_BACKEND, joblib_backend))

    assert isinstance(feature_frequency, list)
    assert isinstance(vocabulary, numpy.ndarray)
    assert isinstance(n_jobs, int)

    logger.debug(msg='making tuple pairs for csr matrix with n(process)={}'.format(n_jobs))

    set_value_position_list = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(
        joblib.delayed(SUB_FUNC_make_value_pairs)(
            doc_id,
            doc_freq_obj,
            vocabulary
        )
        for doc_id, doc_freq_obj in enumerate(feature_frequency)
    )  # type: List[numpy.ndarray]

    # make 2-d list into 1-d list
    value_position_list = sorted(
            [l for set in set_value_position_list for l in set],
        key=lambda pos_tuple: (pos_tuple[0], pos_tuple[1], pos_tuple[2]))

    row, col, data = make_csr_list(value_position_list)

    return row, col, data


def make_csr_objects(row, col, data, n_feature, n_docs):
    """This is main function of making csr_matrix from given data

    :param row:
    :param col:
    :param data:
    :param n_feature:
    :param n_docs:
    :return:
    """
    assert isinstance(row, list)
    assert isinstance(col, list)
    assert isinstance(data, list)
    assert isinstance(n_feature, int)
    assert isinstance(n_docs, int)

    return csr_matrix((data, (row, col)), shape=(n_docs, n_feature))