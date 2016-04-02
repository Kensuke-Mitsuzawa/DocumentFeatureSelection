#! -*- coding: utf-8 -*-
__author__ = 'kensuke-mi'

from document_feature_selection import PMI, TFIDF, DataConverter, DataCsrMatrix
from scipy.sparse.csr import csr_matrix
import logging
logger = logging.Logger(level=logging.DEBUG, name='test')
import pprint

input_dict = {
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

# --------------------------------------------------------------------
# example for getting PMI score

# getting document-frequency matrix.
# ATTENTION: the input for PMI MUST be document-frequency matrix. NOT term-frequency matrix
doc_freq_information = DataConverter().labeledMultiDocs2DocFreqMatrix(
    labeled_documents=input_dict,
    ngram=1,
    n_jobs=5
)
assert isinstance(doc_freq_information, DataCsrMatrix)

pmi_scored_sparse_matrix = PMI().fit_transform(X=doc_freq_information.csr_matrix_, n_docs_distribution=doc_freq_information.n_docs_distribution)
assert isinstance(pmi_scored_sparse_matrix, csr_matrix)
#pprint.pprint(pmi_scored_sparse_matrix.toarray())

# You can show result with dictionary type
pmi_score_result = DataConverter().ScoreMatrix2ScoreDictionary(
    scored_matrix=pmi_scored_sparse_matrix,
    label2id_dict=doc_freq_information.label2id_dict,
    vocaburary2id_dict=doc_freq_information.vocabulary
)
print('-'*30)
print('PMI score')
pprint.pprint(pmi_score_result)


# --------------------------------------------------------------------
# example for getting TF-IDF score


# getting term-frequency matrix.
# ATTENTION: the input for TF-IDF MUST be term-frequency matrix. NOT document-frequency matrix
term_freq_information = DataConverter().labeledMultiDocs2TermFreqMatrix(
    labeled_documents=input_dict,
    ngram=1,
    n_jobs=5
)
assert isinstance(term_freq_information, DataCsrMatrix)

tfIdf_scored_sparse_matrix = TFIDF().fit_transform(X=term_freq_information.csr_matrix_)
assert isinstance(tfIdf_scored_sparse_matrix, csr_matrix)
#pprint.pprint(tfIdf_scored_sparse_matrix.toarray())

# You can show result with dictionary type
tfidf_score_result = DataConverter().ScoreMatrix2ScoreDictionary(
    scored_matrix=tfIdf_scored_sparse_matrix,
    label2id_dict=term_freq_information.label2id_dict,
    vocaburary2id_dict=term_freq_information.vocabulary
)
print('-'*30)
print('TFIDF score')
pprint.pprint(tfidf_score_result)