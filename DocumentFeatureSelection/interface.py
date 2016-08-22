from DocumentFeatureSelection.models import DataCsrMatrix, ScoredResultObject
from DocumentFeatureSelection.common import data_converter
from DocumentFeatureSelection.soa.soa_python3 import SOA
from DocumentFeatureSelection.pmi.PMI_python3 import PMI
from DocumentFeatureSelection.tf_idf.tf_idf import TFIDF
from DocumentFeatureSelection.bns.bns_python3 import BNS
from DocumentFeatureSelection import init_logger
from typing import List, Dict, Any, Union, Tuple
from scipy.sparse.csr import csr_matrix
import logging
logger = init_logger.init_logger(logging.getLogger(init_logger.LOGGER_NAME))
METHOD_NAMES = ['soa', 'pmi', 'tf_idf', 'bns']


def run_feature_selection(input_dict:Dict[str,List[List[Union[str,Tuple[Any]]]]],
                          method:str,
                          ngram:int=1,
                          n_jobs:int=1,
                          matrix_form=None)->ScoredResultObject:
    if not method in METHOD_NAMES:
        raise Exception('method name must be either of {}. Yours: {}'.format(METHOD_NAMES, method))

    if method == 'tf_idf':
        # getting term-frequency matrix.
        # ATTENTION: the input for TF-IDF MUST be term-frequency matrix. NOT document-frequency matrix
        matrix_data_object = data_converter.DataConverter().labeledMultiDocs2TermFreqMatrix(
            labeled_documents=input_dict,
            ngram=ngram,
            n_jobs=n_jobs
        )
        assert isinstance(matrix_data_object, DataCsrMatrix)

        scored_sparse_matrix = TFIDF().fit_transform(X=matrix_data_object.csr_matrix_)
        assert isinstance(scored_sparse_matrix, csr_matrix)

    elif method in ['soa', 'pmi'] and matrix_form is None:
        matrix_data_object = data_converter.DataConverter().labeledMultiDocs2DocFreqMatrix(
            labeled_documents=input_dict,
            ngram=ngram,
            n_jobs=n_jobs
        )
        assert isinstance(matrix_data_object, DataCsrMatrix)
        if method == 'pmi':
            scored_sparse_matrix = PMI().fit_transform(X=matrix_data_object.csr_matrix_,
                                                           n_docs_distribution=matrix_data_object.n_docs_distribution)
            assert isinstance(scored_sparse_matrix, csr_matrix)
        elif method == 'soa':
            scored_sparse_matrix = SOA().fit_transform(X=matrix_data_object.csr_matrix_,
                                                           unit_distribution=matrix_data_object.n_docs_distribution)
            assert isinstance(scored_sparse_matrix, csr_matrix)
    elif method == 'soa' and matrix_form == 'term_freq':
        # getting term-frequency matrix.
        # ATTENTION: the input for TF-IDF MUST be term-frequency matrix. NOT document-frequency matrix
        matrix_data_object = data_converter.DataConverter().labeledMultiDocs2TermFreqMatrix(
            labeled_documents=input_dict,
            ngram=ngram,
            n_jobs=n_jobs
        )
        assert isinstance(matrix_data_object, DataCsrMatrix)

        scored_sparse_matrix = SOA().fit_transform(X=matrix_data_object.csr_matrix_,
                                                   unit_distribution=matrix_data_object.n_docs_distribution)
        assert isinstance(scored_sparse_matrix, csr_matrix)
    elif method == 'bns':
        if not 'positive' in input_dict:
            raise KeyError('input_dict must have "positive" key')
        if not 'negative' in input_dict:
            raise KeyError('input_dict must have "negative" key')
        if len(input_dict.keys()) >= 3:
            raise KeyError('input_dict must not have more than 3 keys if you would like to use BNS.')

        matrix_data_object = data_converter.DataConverter().labeledMultiDocs2TermFreqMatrix(
            labeled_documents=input_dict,
            ngram=ngram,
            n_jobs=n_jobs
        )
        assert isinstance(matrix_data_object, DataCsrMatrix)

        true_class_index = matrix_data_object.label2id_dict['positive']
        scored_sparse_matrix = BNS().fit_transform(
            X=matrix_data_object.csr_matrix_,
            unit_distribution=matrix_data_object.n_term_freq_distribution,
            n_jobs=n_jobs,
            true_index=true_class_index
        )
        assert isinstance(scored_sparse_matrix, csr_matrix)

    else:
        raise Exception()


    return ScoredResultObject(
        scored_matrix=scored_sparse_matrix,
        label2id_dict=matrix_data_object.label2id_dict,
        feature2id_dict=matrix_data_object.vocabulary,
        method=method,
        matrix_form=matrix_form
    )