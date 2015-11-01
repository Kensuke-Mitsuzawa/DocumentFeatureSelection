#! -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
import logging
from scipy.sparse import csr_matrix
from collections import namedtuple
from nltk.util import ngrams
PosTuple = namedtuple('PosTuple', ('doc_id', 'word_id', 'document_frequency'))

__author__ = 'kensuke-mi'
"""This code is data pre-processing functions before PMI calculation.
Main funtion is 'make_pmi_matrix()'
"""


def __check_data_structure(labeled_structure):
    assert isinstance(labeled_structure, dict)
    for key in labeled_structure.keys():
        docs_in_label = labeled_structure[key]
        assert isinstance(docs_in_label, list)
        for doc in docs_in_label:
            for t in doc: assert isinstance(t, (str, unicode))

    return True


def __generate_document_dict(documents):
    """This function gets #Document-frequency in given list of documents

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

    return document_frequency_dict


def __data_convert(labeled_structure):
    assert isinstance(labeled_structure, dict)

    vocabulary = list(set([
        t
        for all_docs
        in labeled_structure.values()
        for docs
        in all_docs
        for t
        in docs
    ]))
    n_feature = len(vocabulary)
    v = {t: index for index, t in enumerate(vocabulary)}

    # make list of document-frequency
    label_group_dict = {}

    token_freq_document = []
    document_index = 0
    for key, docs in labeled_structure.items():
        token_freq_document.append(__generate_document_dict(docs))
        label_group_dict.update({key: document_index})
        document_index += 1


    return token_freq_document, label_group_dict, v


def get_data_col_row_values(doc_id, word, doc_freq, vocaburary):
    col_value = vocaburary[word]
    df_value = doc_freq

    return PosTuple(doc_id, col_value, df_value)


def preprocess_csr_matrix(token_freq_document, vocabulary):
    # MEMO 分散化可能
    """This function makes information to make csr matrix. Data-list/Row-list/Col-list

    :param token_freq_document:
    :param vocabulary:
    :return:
    """
    assert isinstance(token_freq_document, list)

    value_position_list = []
    for doc_id, doc_freq_obj in enumerate(token_freq_document):
        value_pairs = [
            get_data_col_row_values(doc_id=doc_id, word=word, doc_freq=freq, vocaburary=vocabulary)
            for word, freq
            in doc_freq_obj.items()
            ]
        value_position_list += value_pairs

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


def make_csr_objects(row, col, data, n_feature, n_docs):
    assert isinstance(row, list)
    assert isinstance(col, list)
    assert isinstance(data, list)
    assert isinstance(n_feature, int)
    assert isinstance(n_docs, int)

    return csr_matrix((data, (row, col)), shape=(n_docs, n_feature))


def ngram_data_conversion(labeled_structure, n):

    character_joiner = lambda ngram_tuple: '_'.join(ngram_tuple)

    new_documents = {}
    for key in labeled_structure.keys():
        docs = labeled_structure[key]
        new_docs = []
        for d in docs:
            ngram_d = ngrams(d, n)
            generated_ngrams = [character_joiner(g) for g in ngram_d]
            new_docs.append(generated_ngrams)
        new_documents[key] = new_docs
    return new_documents


def make_pmi_matrix(labeled_structure, logger, ngram=1):
    """This function makes document-frequency matrix for PMI calculation.
    Document-frequency matrix is scipy.csr_matrix.

    labeled_structure must be following key-value pair

    {
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

    -----------------------------------------------
    :param labeled_structure above data structure:
    :param logger logging.Logger:
    :param ngram you can get score with ngram-words:
    :return (csr_matrix: scipy.csr_matrix, label_group_dict: dict, vocabulary: dict):
    """
    assert isinstance(logger, logging.Logger)
    __check_data_structure(labeled_structure)

    if ngram > 1:
        labeled_structure = ngram_data_conversion(labeled_structure, ngram)

    logging.debug(msg='Now pre-processing before CSR matrix')
    token_freq_document, label_group_dict, vocabulary = __data_convert(labeled_structure)
    row, col, data = preprocess_csr_matrix(token_freq_document=token_freq_document, vocabulary=vocabulary)
    logging.debug(msg='Finished pre-processing before CSR matrix')
    csr_matrix = make_csr_objects(row, col, data, max(vocabulary.values())+1, len(token_freq_document))

    return csr_matrix, label_group_dict, vocabulary