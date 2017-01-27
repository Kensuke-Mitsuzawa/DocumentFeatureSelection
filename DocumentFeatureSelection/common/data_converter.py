#! -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from DocumentFeatureSelection.common import utils, labeledMultiDocs2labeledDocsSet, ngram_constructor
from DocumentFeatureSelection.models import DataCsrMatrix, AvailableInputTypes
from DocumentFeatureSelection import init_logger
from sqlitedict import SqliteDict
import logging
import sys
import numpy
import tempfile
from typing import Dict
python_version = sys.version_info
logger = init_logger.init_logger(logging.getLogger(init_logger.LOGGER_NAME))

__author__ = 'kensuke-mi'


class DataConverter(object):
    """This class is for converting data type from dict-object into DataCsrMatrix-object which saves information of matrix.
    """
    def __check_data_structure(self, labeled_documents):
        # type: AvailableInputTypes->bool
        """* what you can do
        - This function checks input data structure
        """
        assert isinstance(labeled_documents, (SqliteDict, dict))
        for key, value in labeled_documents.items():
            docs_in_label = labeled_documents[key]
            if not isinstance(docs_in_label, list):
                logger.error(msg=docs_in_label)
                raise TypeError('It expects list object. But your object has {}'.format(type(docs_in_label)))
            for doc in docs_in_label:
                for t in doc:
                    if isinstance(t, (str)):
                        return True
                    elif isinstance(t, tuple):
                        return True
                    else:
                        raise TypeError('Feature format must be either str or tuple')

        return True

    def count_term_frequency_distribution(self, labeled_documents:AvailableInputTypes, label2id:Dict[str,int]):
        """Count term-distribution per label.
        """
        assert isinstance(labeled_documents, (SqliteDict, dict))
        assert isinstance(label2id, dict)

        # count total term-frequency per label
        term_frequency_distribution = {
            label: len(list(utils.flatten(document_lists)))
            for label, document_lists
            in labeled_documents.items()
        }

        # make list of distribution
        term_frequency_distribution_list = [0] * len(labeled_documents)

        for label_string, n_doc in term_frequency_distribution.items():
            #term_index = label2id[numpy.where(label2id['key'] == label_string.encode('utf-8'))][0]['value']
            term_index = label2id[label_string]
            term_frequency_distribution_list[term_index] = n_doc

        return numpy.array(term_frequency_distribution_list, dtype='i8')

    def count_document_distribution(self, labeled_documents:AvailableInputTypes, label2id:Dict[str,int])->numpy.ndarray:
        """This method count n(docs) per label.
        """
        assert isinstance(labeled_documents, (SqliteDict, dict))
        assert isinstance(label2id, dict)

        # count n(docs) per label
        n_doc_distribution = {
            label: len(document_lists)
            for label, document_lists
            in labeled_documents.items()
        }

        # make list of distribution
        n_doc_distribution_list = [0] * len(labeled_documents)

        for label_string, n_doc in n_doc_distribution.items():
            #docs_index = label2id[numpy.where(label2id['key'] == label_string.encode('utf-8'))][0]['value']
            docs_index = label2id[label_string]
            n_doc_distribution_list[docs_index] = n_doc

        return numpy.array(n_doc_distribution_list, dtype='i8')

    def labeledMultiDocs2TermFreqMatrix(self,
                                        labeled_documents:AvailableInputTypes,
                                        is_use_cache:bool=False,
                                        is_use_memmap:bool=False,
                                        path_working_dir:str=tempfile.mkdtemp(),
                                        joblib_backend:str='auto',
                                        cache_backend:str='PersistentDict',
                                        ngram:int=1,
                                        n_jobs:int=1):
        """* What you can do
        - This function makes TERM-frequency matrix for TF-IDF calculation.
        - TERM-frequency matrix is scipy.csr_matrix.

        * Params
        - labeled_documents: Dict object which has category-name as key, and list of features as value
        - is_use_cache: boolean flag to use disk-drive for keeping objects which tends to be huge.
        - path_working_dir: path to directory for saving cache files
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
            csr_matrix_=set_document_information.matrix_object,
            label2id_dict=set_document_information.label2id,
            vocabulary=set_document_information.feature2id,
            n_docs_distribution=n_docs_distribution,
            n_term_freq_distribution=term_frequency_distribution,
            is_use_cache=is_use_cache,
            is_use_memmap=is_use_memmap,
            path_working_dir=path_working_dir,
            cache_backend=cache_backend
        )

    def labeledMultiDocs2DocFreqMatrix(self,
                                       labeled_documents:AvailableInputTypes,
                                       is_use_cache:bool=False,
                                       is_use_memmap:bool=False,
                                       path_working_dir:str=None,
                                       ngram:int=1,
                                       n_jobs:int=1,
                                       joblib_backend:str='auto')->DataCsrMatrix:
        """This function makes document-frequency matrix. Document-frequency matrix is scipy.csr_matrix.

        * Input object
        - "labeled_structure" is either of Dict object or shelve.DbfilenameShelf. The example format is below
            >>> {"label_a": [["I", "aa", "aa", "aa", "aa", "aa"],["bb", "aa", "aa", "aa", "aa", "aa"],["I", "aa", "hero", "some", "ok", "aa"]],
            >>> "label_b": [["bb", "bb", "bb"],["bb", "bb", "bb"],["hero", "ok", "bb"],["hero", "cc", "bb"],],
            >>> "label_c": [["cc", "cc", "cc"],["cc", "cc", "bb"],["xx", "xx", "cc"],["aa", "xx", "cc"],]}

        * Output
        - DataCsrMatrix object.
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
            csr_matrix_=set_document_information.matrix_object,
            label2id_dict=set_document_information.label2id,
            vocabulary=set_document_information.feature2id,
            n_docs_distribution=n_docs_distribution,
            n_term_freq_distribution=term_frequency_distribution,
            is_use_cache=is_use_cache,
            is_use_memmap=is_use_memmap,
            path_working_dir=path_working_dir
        )


'''
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


def scored_matrix2score_dictionary(scored_matrix:csr_matrix,
                                label2id_dict:Dict[str,int],
                                feature2id_dict:Dict[FeatureType,int],
                                outformat:str='items',
                                sort_desc:bool=True,
                                n_jobs:int=1):
    """Get dictionary structure of PMI featured scores.

    You can choose 'dict' or 'items' for ```outformat``` parameter.

    If outformat='dict', you get

    >>> {label_name:{feature: score}}

    Else if outformat='items', you get

    >>> [{feature: score}]
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

# for old version code
ScoreMatrix2ScoreDictionary = scored_matrix2score_dictionary
'''