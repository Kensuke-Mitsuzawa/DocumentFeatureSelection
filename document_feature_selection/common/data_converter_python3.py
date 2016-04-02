#! -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from document_feature_selection.common import utils
from collections import namedtuple
from logging import getLogger, StreamHandler
from scipy.sparse import csr_matrix
from nltk.util import ngrams
import logging
import joblib
import sys

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)

python_version = sys.version_info

PosTuple = namedtuple('PosTuple', ('doc_id', 'word_id', 'document_frequency'))

__author__ = 'kensuke-mi'

"""

Example:
    >>> input_format = {
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
"""




def check_data_structure(labeled_structure):
    """This function checks input data structure

    :param labeled_structure:
    :return:
    """
    assert isinstance(labeled_structure, dict)
    for key in labeled_structure.keys():
        docs_in_label = labeled_structure[key]
        assert isinstance(docs_in_label, list)
        for doc in docs_in_label:
            for t in doc:
                if not isinstance(t, (str)):
                    raise TypeError('String type must be str type')

    return True


def generate_document_dict(documents):
    """This function gets Document-frequency in given list of documents

    :param documents [[str]]:
    :return dict that represents document frequency:
    """
    assert isinstance(documents, list)
    assert isinstance(documents[0], list)
    V = list(set([t for d in documents for t in d]))
    document_frequency_dict = {}
    for v in V:
        binary_count = [1 for d in documents if v in d]
        document_frequency_dict[v] = sum(binary_count)

    assert isinstance(document_frequency_dict, dict)
    return document_frequency_dict


def make_document_frequency_data(labeled_structure):
    """This function converts data structure from lable: 2-dim list
    INPUT
    >>> {'a': [ ['aa', 'bb'], ['bb', 'cc'] ]}

    OUTPUT
    >>> {'a': [ ['aa', 'bb', 'cc'] ]}

    :param labeled_structure:
    :return:
    """
    assert isinstance(labeled_structure, dict)

    vocabulary_list = list(set(utils.flatten(labeled_structure.values())))
    vocabulary_list = sorted(vocabulary_list)

    v = {t: index for index, t in enumerate(vocabulary_list)}

    # make label: id dictionary structure
    label_group_dict = {}
    # make list of document-frequency
    token_freq_document = []
    document_index = 0

    for key, docs in sorted(labeled_structure.items(), key=lambda key_value_tuple: key_value_tuple[0]):
        token_freq_document.append(generate_document_dict(docs))
        label_group_dict.update({key: document_index})
        document_index += 1

    assert isinstance(v, dict)
    assert isinstance(token_freq_document, list)
    assert isinstance(label_group_dict, dict)
    return token_freq_document, label_group_dict, v


def make_csr_objects(row, col, data, n_feature, n_docs):
    assert isinstance(row, list)
    assert isinstance(col, list)
    assert isinstance(data, list)
    assert isinstance(n_feature, int)
    assert isinstance(n_docs, int)

    return csr_matrix((data, (row, col)), shape=(n_docs, n_feature))


def get_data_col_row_values(doc_id, word, doc_freq, vocaburary):
    assert isinstance(vocaburary, dict)
    col_value = vocaburary[word]
    # df value is word frequency in documents
    df_value = doc_freq

    return PosTuple(doc_id, col_value, df_value)


def SUB_FUNC_make_value_pairs(doc_id, doc_freq_obj, vocabulary):
    value_pairs = [
        get_data_col_row_values(doc_id=doc_id, word=word, doc_freq=freq, vocaburary=vocabulary)
        for word, freq
        in doc_freq_obj.items()
        ]
    assert isinstance(value_pairs, list)
    return value_pairs


def preprocess_csr_matrix(token_freq_document, vocabulary, n_jobs):
    """This function makes information to make csr matrix. Data-list/Row-list/Col-list

    :param token_freq_document:
    :param vocabulary:
    :return:
    """
    assert isinstance(token_freq_document, list)
    assert isinstance(vocabulary, dict)
    assert isinstance(n_jobs, int)

    logger.debug(msg='making tuple pairs for csr matrix with n(process)={}'.format(n_jobs))

    set_value_position_list = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(SUB_FUNC_make_value_pairs)(
            doc_id,
            doc_freq_obj,
            vocabulary
        )
        for doc_id, doc_freq_obj in enumerate(token_freq_document)
    )
    value_position_list = sorted(
            [l for set in set_value_position_list for l in set],
        key=lambda pos_tuple: (pos_tuple[0], pos_tuple[1], pos_tuple[2])
    )

    row, col, data = make_csr_list(value_position_list)

    return row, col, data


def make_csr_list(value_position_list):
    data = []
    row = []
    col = []
    for position_tuple in value_position_list:
        row.append(position_tuple.doc_id)
        col.append(position_tuple.word_id)
        data.append(position_tuple.document_frequency)

    return row, col, data


def SUB_FUNC_ngram_data_conversion(key, docs, n):
    """This function converts list of tokens into list of n_grams tokens

    :param key: key name of document
    :param docs: lits of tokens
    :param n: n of n_gram
    """

    assert isinstance(key, str)
    assert isinstance(docs, list)
    assert isinstance(n, int)

    character_joiner = lambda ngram_tuple: '_'.join(ngram_tuple)
    generate_nGram = lambda ngram_d: [character_joiner(g) for g in ngram_d]

    new_docs = [
        generate_nGram(ngrams(d, n))
        for d in docs
    ]

    assert isinstance(new_docs, list)
    assert isinstance(new_docs[0], list)

    return (key, new_docs)


def count_document_distribution(labeled_documents, label2id_dict):
    """This method count n(docs) per label.

    :param labeled_documents:
    :param label2id_dict:
    :return:
    """
    assert isinstance(labeled_documents, dict)
    assert isinstance(label2id_dict, dict)

    # count n(docs) per label
    n_doc_distribution = {
        label: len(document_lists)
        for label, document_lists
        in labeled_documents.items()
    }

    # make list of distribution
    n_doc_distribution_list = [0] * len(labeled_documents.keys())

    for label_string, n_doc in n_doc_distribution.items():
        n_doc_distribution_list[label2id_dict[label_string]] = n_doc

    return n_doc_distribution_list


def convert_data(labeled_structure, ngram=1, n_jobs=1):
    """This function makes document-frequency matrix for PMI calculation.
    Document-frequency matrix is scipy.csr_matrix.

    labeled_structure must be following key-value pair

        >>> {
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

    There is 3 Output data.

    vocaburary is, dict object with token: feature_id
    >>> {'I_aa_hero': 4, 'xx_xx_cc': 1, 'I_aa_aa': 2, 'bb_aa_aa': 3, 'cc_cc_bb': 8}

    label_group_dict is, dict object with label_name: label_id
    >>> {'label_b': 0, 'label_c': 1, 'label_a': 2}

    csr_matrix is, sparse matrix from scipy.sparse


    :param dict labeled_structure: above data structure
    :param int ngram: you can get score with ngram-words
    :return: `(csr_matrix: scipy.csr_matrix, label_group_dict: dict, vocabulary: dict)`
    :rtype: tuple
    """
    check_data_structure(labeled_structure)

    if ngram > 1:
        logger.debug(msg='Now making {}-gram data strucutre with n(process) = {}'.format(ngram, n_jobs))
        key_docs_tuples = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(SUB_FUNC_ngram_data_conversion)(
                key=key,
                docs=docs,
                n=ngram
            )
            for key, docs in labeled_structure.items()
        )
        labeled_structure = dict(key_docs_tuples)
        logger.debug(msg='Finished making N-gram')


    logger.debug(msg='Now pre-processing before CSR matrix')
    # convert data structure
    token_freq_document, label2id_dict, vocabulary = make_document_frequency_data(labeled_structure)
    # make set of tuples to construct csr_matrix
    row, col, data = preprocess_csr_matrix(
            token_freq_document=token_freq_document,
            vocabulary=vocabulary,
            n_jobs=n_jobs
    )
    logger.debug(msg='Finished pre-processing before CSR matrix')
    csr_matrix_ = make_csr_objects(row, col, data, max(vocabulary.values())+1, len(token_freq_document))

    # count n(docs) per label
    n_docs_distribution = count_document_distribution(
        labeled_documents=labeled_structure,
        label2id_dict=label2id_dict
    )

    assert isinstance(csr_matrix_, csr_matrix)
    assert isinstance(label2id_dict, dict)
    assert isinstance(vocabulary, dict)
    assert isinstance(n_docs_distribution, list)
    return csr_matrix_, label2id_dict, vocabulary, n_docs_distribution

# -------------------------------------------------------------------------------------------------------------------
# function for output

def __conv_into_dict_format(word_score_items):
    out_format_structure = {}
    for item in word_score_items:
        if item['label'] not in out_format_structure :
            out_format_structure[item['label']] = [{'word': item['word'], 'score': item['score']}]
        else:
            out_format_structure[item['label']].append({'word': item['word'], 'score': item['score']})
    return out_format_structure


def get_weight_feature_dictionary(scored_matrix, label_id_dict, feature_id_dict, outformat='items',
                                  sort_desc=True, n_jobs=1):
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

    """

    scored_objects = utils.get_feature_dictionary(
        weighted_matrix=scored_matrix,
        vocabulary=feature_id_dict,
        label_group_dict=label_id_dict,
        logger=logger,
        n_jobs=n_jobs
    )

    if sort_desc: scored_objects = \
        sorted(scored_objects, key=lambda x: x['score'], reverse=True)

    if outformat=='dict':
        out_format_structure = __conv_into_dict_format(scored_objects)
    elif outformat=='items':
        out_format_structure = scored_objects
    else:
        raise ValueError('outformat must be either of {dict, items}')

    return out_format_structure