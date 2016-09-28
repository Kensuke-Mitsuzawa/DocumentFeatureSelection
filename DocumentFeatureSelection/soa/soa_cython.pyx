import math
import scipy
cimport numpy as np
from cpython cimport bool

cdef float soa(
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.int64_t, ndim=1] unit_distribution,
    int n_total_docs,
    int feature_index,
    int sample_index, 
    bool verbose):
    # X is either of term-frequency matrix per label or document-frequency per label

    matrix_size = X.shape
    NOT_sample_indexes = [i for i in range(0, matrix_size[0]) if i != sample_index]

    # freq_w_e is term-frequency(or document-frequency) of w in the unit having the specific label e
    cdef float freq_w_e = X[sample_index, feature_index]
    # freq_w_not_e is term-frequency(or document-frequency) of w in units except the specific label e
    cdef float freq_w_not_e = X[NOT_sample_indexes, feature_index].sum()
    # freq_e is the number of the unit having specific label e
    cdef float freq_e = unit_distribution[sample_index]
    # freq_not_e is the number of the unit NOT having the specific label e
    cdef float freq_not_e = n_total_docs - freq_e
    cdef float nominator, denominator, ans, soa_val

    if verbose:
        print('For feature_index:{} sample_index:{}'.format(feature_index, sample_index))
        print('freq_w_e:{} freq_w_not_e:{} freq_e:{} freq_not_e:{}'.format(
            freq_w_e,
            freq_w_not_e,
            freq_e,
            freq_not_e
        ))

    if freq_w_e == 0 or freq_w_not_e == 0 or freq_e == 0 or freq_not_e == 0:
        return 0.0
    else:
        nominator = (float(freq_w_e) * freq_not_e)
        denominator = (float(freq_e) * freq_w_not_e)
        ans = nominator / denominator
        soa_val = math.log(ans, 2)
        return soa_val


def main(X,
        np.ndarray[np.int64_t, ndim=1] n_docs_distribution,
        int n_total_doc,
        sample_range,
        feature_range,
        bool verbose=False):
    """What you can do
    - calculate PMI score based on given data.
    - The function returns list of tuple, whose element is (sample_index, feature_index, score)
    - Your input matrix should be numpy.ndarray or scipy.sparse.csr_matrix. The matrix should represent document-frequency of each feature.
    """

    cdef int n_samples = X.shape[0]

    if isinstance(X, scipy.sparse.csr_matrix):
        X = X.toarray()

    cdef int sample_index, feature_index
    soa_score_csr_source = [
        (
            sample_index,
            feature_index,
            soa(X, n_docs_distribution, n_total_doc, feature_index, sample_index, verbose)
         )
        for sample_index in sample_range
        for feature_index in feature_range
    ]
    non_zero_soa_score_csr_source = [score_tuple for score_tuple in soa_score_csr_source if not score_tuple[2]==0]

    return non_zero_soa_score_csr_source