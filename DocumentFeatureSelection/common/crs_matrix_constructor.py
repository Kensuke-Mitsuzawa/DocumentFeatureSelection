from logging import getLogger, StreamHandler
from collections import namedtuple
import joblib
import sys
import logging
from scipy.sparse import csr_matrix

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)

python_version = sys.version_info
__author__ = 'kensuke-mi'

PosTuple = namedtuple('PosTuple', ('doc_id', 'word_id', 'document_frequency'))


def get_data_col_row_values(doc_id:int, word:int, doc_freq:int, vocaburary):
    assert isinstance(vocaburary, dict)
    try:
        col_value = vocaburary[word]
    except KeyError:
        print()
    # df value is word frequency in documents
    df_value = doc_freq

    return PosTuple(doc_id, col_value, df_value)

def SUB_FUNC_make_value_pairs(doc_id:int, doc_freq_obj, vocabulary):
    value_pairs = [
        get_data_col_row_values(doc_id=doc_id, word=word, doc_freq=freq, vocaburary=vocabulary)
        for word, freq
        in doc_freq_obj.items()
        ]
    assert isinstance(value_pairs, list)
    return value_pairs

def make_csr_list(value_position_list):
    data = []
    row = []
    col = []
    for position_tuple in value_position_list:
        row.append(position_tuple.doc_id)
        col.append(position_tuple.word_id)
        data.append(position_tuple.document_frequency)

    return row, col, data


def preprocess_csr_matrix(feature_frequency, vocabulary, n_jobs):
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
    assert isinstance(feature_frequency, list)
    assert isinstance(vocabulary, dict)
    assert isinstance(n_jobs, int)

    logger.debug(msg='making tuple pairs for csr matrix with n(process)={}'.format(n_jobs))

    set_value_position_list = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(SUB_FUNC_make_value_pairs)(
            doc_id,
            doc_freq_obj,
            vocabulary
        )
        for doc_id, doc_freq_obj in enumerate(feature_frequency)
    )
    value_position_list = sorted(
            [l for set in set_value_position_list for l in set],
        key=lambda pos_tuple: (pos_tuple[0], pos_tuple[1], pos_tuple[2])
    )

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