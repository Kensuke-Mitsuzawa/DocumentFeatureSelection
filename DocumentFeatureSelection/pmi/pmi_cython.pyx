import math
import scipy
cimport numpy as np

cdef float pmi(np.ndarray[np.float64_t, ndim=2] X,
        int n_samples,
        np.ndarray[np.int64_t, ndim=1] n_docs_distribution,
        int n_total_doc,
        int feature_index,
        int sample_index):
    """get PMI score for given feature & sample index
    """
    cdef i
    sample_indexes = [i for i in range(0, n_samples) if i != sample_index]

    # n_11 is #docs having feature(i.e. word) in the specified index(label)
    cdef float n_11 = X[sample_index, feature_index]
    # n_01 is #docs NOT having feature in the specified index(label)
    cdef float n_01 = n_docs_distribution[sample_index] - n_11
    # n_10 is #docs having feature in NOT specified index(indexes except specified index)
    cdef float n_10 = X[sample_indexes, feature_index].sum()
    # n_00 is #docs NOT having feature in NOT specified index(indexes except specified index)
    cdef float n_00 = n_total_doc - (n_10 + n_docs_distribution[sample_index])

    cdef float temp1, temp2, temp3, temp4, score

    if n_11 == 0.0 or n_10 == 0.0 or n_01 == 0.0 or n_00 == 0.0:
        return 0
    else:
        temp1 = n_11/n_total_doc * math.log((n_total_doc*n_11)/((n_10+n_11)*(n_01+n_11)), 2)
        temp2 = n_01/n_total_doc * math.log((n_total_doc*n_01)/((n_00+n_01)*(n_01+n_11)), 2)
        temp3 = n_10/n_total_doc * math.log((n_total_doc*n_10)/((n_10+n_11)*(n_00+n_10)), 2)
        temp4 = n_00/n_total_doc * math.log((n_total_doc*n_00)/((n_00+n_01)*(n_00+n_10)), 2)
        score = temp1 + temp2 + temp3 + temp4

        if score < 0:
            print(score)
            raise Exception('PMI score={}. Score under 0 is detected. Something strange in Input matrix. Check your input matrix.'.format(score))

        return score


def main(X,
        np.ndarray[np.int64_t, ndim=1] n_docs_distribution,
        int n_total_doc,
        sample_range,
        feature_range):
    """What you can do
    - calculate PMI score based on given data.
    - The function returns list of tuple, whose element is (sample_index, feature_index, score)
    - Your input matrix should be numpy.ndarray or scipy.sparse.csr_matrix. The matrix should represent document-frequency of each feature.
    """

    cdef int n_samples = X.shape[0]

    if isinstance(X, scipy.sparse.csr_matrix):
        X = X.toarray()

    cdef int sample_index, feature_index
    pmi_score_csr_source = [
        (
            sample_index,
            feature_index,
            pmi(X, n_samples, n_docs_distribution, n_total_doc, feature_index, sample_index)
         )
        for sample_index in sample_range
        for feature_index in feature_range
    ]
    non_zero_pmi_score_csr_source = [score_tuple for score_tuple in pmi_score_csr_source if not score_tuple[2]==0]

    return non_zero_pmi_score_csr_source