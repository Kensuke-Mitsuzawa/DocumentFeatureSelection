#! -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from DocumentFeatureSelection.common import utils
from scipy.sparse import csr_matrix
from DocumentFeatureSelection.common import crs_matrix_constructor
from DocumentFeatureSelection.common import labeledMultiDocs2labeledDocsSet
from DocumentFeatureSelection.common import ngram_constructor
from DocumentFeatureSelection.models import DataCsrMatrix, FeatureType
from DocumentFeatureSelection import init_logger
import logging
import sys
import numpy
import pickle
from typing import Dict, List, Tuple, Union, Any
python_version = sys.version_info
logger = init_logger.init_logger(logging.getLogger(init_logger.LOGGER_NAME))

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


class DataConverter(object):
    """
    """
    def __check_data_structure(self, labeled_documents):
        # type: (Dict[str, Union[str, List[Any], Tuple[Any]]])->bool
        """* what you can do
        - This function checks input data structure
        """
        assert isinstance(labeled_documents, dict)
        for key in labeled_documents.keys():
            docs_in_label = labeled_documents[key]
            assert isinstance(docs_in_label, list)
            for doc in docs_in_label:
                for t in doc:
                    if isinstance(t, (str)):
                        return True
                    elif isinstance(t, tuple):
                        return True
                    else:
                        raise TypeError('Feature format must be either str or tuple')

        return True


    def count_term_frequency_distribution(self, labeled_documents:Dict[str,List[Any]], label2id:Dict[str,int]):
        """Count term-distribution per label.
        """
        assert isinstance(labeled_documents, dict)
        assert isinstance(label2id, dict)

        # count total term-frequency per label
        term_frequency_distribution = {
            label: len(list(utils.flatten(document_lists)))
            for label, document_lists
            in labeled_documents.items()
        }

        # make list of distribution
        term_frequency_distribution_list = [0] * len(labeled_documents.keys())

        for label_string, n_doc in term_frequency_distribution.items():
            #term_index = label2id[numpy.where(label2id['key'] == label_string.encode('utf-8'))][0]['value']
            term_index = label2id[label_string]
            term_frequency_distribution_list[term_index] = n_doc

        return numpy.array(term_frequency_distribution_list, dtype='i8')


    def count_document_distribution(self, labeled_documents:Dict[str,List[Any]], label2id:Dict[str,int])->numpy.ndarray:
        """This method count n(docs) per label.

        :param labeled_documents:
        :param label2id_dict:
        :return:
        """
        assert isinstance(labeled_documents, dict)
        assert isinstance(label2id, dict)

        # count n(docs) per label
        n_doc_distribution = {
            label: len(document_lists)
            for label, document_lists
            in labeled_documents.items()
        }

        # make list of distribution
        n_doc_distribution_list = [0] * len(labeled_documents.keys())

        for label_string, n_doc in n_doc_distribution.items():
            #docs_index = label2id[numpy.where(label2id['key'] == label_string.encode('utf-8'))][0]['value']
            docs_index = label2id[label_string]
            n_doc_distribution_list[docs_index] = n_doc

        return numpy.array(n_doc_distribution_list, dtype='i8')

    def labeledMultiDocs2TermFreqMatrix(self, labeled_documents, ngram=1, n_jobs=1, joblib_backend='auto'):
        """This function makes TERM-frequency matrix for TF-IDF calculation.
        TERM-frequency matrix is scipy.csr_matrix.
        """
        self.__check_data_structure(labeled_documents)

        if ngram > 1:
            labeled_documents = ngram_constructor.ngram_constructor(
                labeled_documents=labeled_documents,
                ngram=ngram,
                n_jobs=n_jobs
            )

        logger.debug(msg='Now pre-processing before CSR matrix')
        # convert data structure
        set_document_information = labeledMultiDocs2labeledDocsSet.multiDocs2TermFreqInfo(labeled_documents)

        # count n(docs) per label
        n_docs_distribution = self.count_document_distribution(
            labeled_documents=labeled_documents,
            label2id=set_document_information.label2id
        )
        # count term-frequency per label
        term_frequency_distribution = self.count_term_frequency_distribution(
            labeled_documents=labeled_documents,
            label2id=set_document_information.label2id
        )

        return DataCsrMatrix(
                set_document_information.matrix_object,
                set_document_information.label2id,
                set_document_information.feature2id,
                n_docs_distribution, term_frequency_distribution)


    def labeledMultiDocs2DocFreqMatrix(self,
                                       labeled_documents:Dict[str,List[Any]],
                                       ngram:int=1,
                                       n_jobs:int=1,
                                       joblib_backend:str='auto')->DataCsrMatrix:
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
        self.__check_data_structure(labeled_documents)

        if ngram > 1:
            labeled_documents = ngram_constructor.ngram_constructor(
                labeled_documents=labeled_documents,
                ngram=ngram,
                n_jobs=n_jobs)

        logger.debug(msg='Now pre-processing before CSR matrix')
        # convert data structure
        set_document_information = labeledMultiDocs2labeledDocsSet.multiDocs2DocFreqInfo(labeled_documents,
                                                                                         n_jobs=n_jobs)
        assert isinstance(set_document_information, labeledMultiDocs2labeledDocsSet.SetDocumentInformation)

        # count n(docs) per label
        n_docs_distribution = self.count_document_distribution(
            labeled_documents=labeled_documents,
            label2id=set_document_information.label2id
        )
        # count term-frequency per label
        term_frequency_distribution = self.count_term_frequency_distribution(
            labeled_documents=labeled_documents,
            label2id=set_document_information.label2id
        )
        return DataCsrMatrix(
                set_document_information.matrix_object,
                set_document_information.label2id,
                set_document_information.feature2id,
                n_docs_distribution, term_frequency_distribution)



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


def ScoreMatrix2ScoreDictionary(scored_matrix:csr_matrix,
                                label2id_dict:Dict[str,int],
                                feature2id_dict:Dict[FeatureType,int],
                                outformat:str='items',
                                sort_desc:bool=True,
                                n_jobs:int=1):
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
        vocabulary=feature2id_dict,
        label_group_dict=label2id_dict,
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