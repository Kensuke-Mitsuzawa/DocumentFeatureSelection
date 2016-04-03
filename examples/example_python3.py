#! -*- coding: utf-8 -*-
__author__ = 'kensuke-mi'

from DocumentFeatureSelection import PMI, TFIDF, DataConverter, DataCsrMatrix, SOA, BNS
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


# --------------------------------------------------------------------
# example for getting SOA score
# According to original paper "Using Hashtags to Capture Fine Emotion Categories from Tweets", Salif, 2013, he used SOA based on Term-frequency.
# On the contrary, it is normal to use document-frequency for PMI computing. SOA is based on PMI computing.
# I suggest you can try both of Term-frequency based and Doc-frequency based.


# getting term-frequency matrix.
term_freq_information = DataConverter().labeledMultiDocs2TermFreqMatrix(
    labeled_documents=input_dict,
    ngram=1,
    n_jobs=5
)
assert isinstance(term_freq_information, DataCsrMatrix)

doc_freq_information = DataConverter().labeledMultiDocs2DocFreqMatrix(
    labeled_documents=input_dict,
    ngram=1,
    n_jobs=5
)
assert isinstance(doc_freq_information, DataCsrMatrix)

term_freq_based_soa_score = SOA().fit_transform(X=term_freq_information.csr_matrix_,
                    unit_distribution=term_freq_information.n_term_freq_distribution,
                    n_jobs=5)
assert isinstance(term_freq_based_soa_score, csr_matrix)

doc_freq_based_soa_score = SOA().fit_transform(X=doc_freq_information.csr_matrix_,
                    unit_distribution=doc_freq_information.n_docs_distribution,
                    n_jobs=5)
assert isinstance(doc_freq_based_soa_score, csr_matrix)


term_soa_score_dict = DataConverter().ScoreMatrix2ScoreDictionary(
    scored_matrix=term_freq_based_soa_score,
    label2id_dict=term_freq_information.label2id_dict,
    vocaburary2id_dict=term_freq_information.vocabulary
)

doc_soa_score_dict = DataConverter().ScoreMatrix2ScoreDictionary(
    scored_matrix=doc_freq_based_soa_score,
    label2id_dict=doc_freq_information.label2id_dict,
    vocaburary2id_dict=doc_freq_information.vocabulary
)


print('Term-frequency based soa score')
pprint.pprint(term_soa_score_dict)

print('Doc-frequency based soa score')
pprint.pprint(doc_soa_score_dict)


# --------------------------------------------------------------------
# example for getting BNS score
# According to original paper "Lei Tang and Huan Liu, "Bias Analysis in Text Classification for Highly Skewed Data", 2005", BNS is used with term-frequency.
# I suggest you can try both of Term-frequency based and Doc-frequency based.

# Input data must contain 2 labels. Still, any label-names are plausible.
# If you put data having more than 2 labels, BNS class returns Exception error.
input_dict = {
    "positive": [
        ["I", "aa", "aa", "aa", "aa", "aa"],
        ["bb", "aa", "aa", "aa", "aa", "aa"],
        ["I", "aa", "hero", "some", "ok", "aa"]
    ],
    "negative": [
        ["bb", "bb", "bb"],
        ["bb", "bb", "bb"],
        ["hero", "ok", "bb"],
        ["hero", "cc", "bb"],
    ]
}

doc_freq_with_2labels_information = DataConverter().labeledMultiDocs2TermFreqMatrix(
    labeled_documents=input_dict,
    ngram=1,
    n_jobs=5
)
assert isinstance(doc_freq_with_2labels_information, DataCsrMatrix)

true_class_index = doc_freq_with_2labels_information.label2id_dict['positive']
bns_scored_csr_matrix = BNS().fit_transform(
    X=doc_freq_with_2labels_information.csr_matrix_,
    unit_distribution=doc_freq_with_2labels_information.n_term_freq_distribution,
    n_jobs=5,
    true_index=true_class_index
)
assert isinstance(bns_scored_csr_matrix, csr_matrix)

doc_bns_score_dict = DataConverter().ScoreMatrix2ScoreDictionary(
    scored_matrix=bns_scored_csr_matrix,
    label2id_dict=doc_freq_with_2labels_information.label2id_dict,
    vocaburary2id_dict=doc_freq_with_2labels_information.vocabulary
)

print('BNS score')
pprint.pprint(doc_bns_score_dict)