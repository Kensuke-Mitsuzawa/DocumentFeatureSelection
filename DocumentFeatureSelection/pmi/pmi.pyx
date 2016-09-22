import numpy
import math

def pmi(X,
        n_docs_distribution,
        n_total_doc,
        feature_index,
        sample_index, verbose=False):
    """get PMI score for given feature & sample index

    :param X:
    :param feature_index:
    :param sample_index:
    :return:
    """
    matrix_size = X.shape
    sample_indexes = [i for i in range(0, matrix_size[0]) if i != sample_index]

    # n_11 is #docs having feature(i.e. word) in the specified index(label)
    n_11 = X[sample_index, feature_index]
    # n_01 is #docs NOT having feature in the specified index(label)
    n_01 = n_docs_distribution[sample_index] - n_11
    # n_10 is #docs having feature in NOT specified index(indexes except specified index)
    n_10 = X[sample_indexes, feature_index].sum()
    # n_00 is #docs NOT having feature in NOT specified index(indexes except specified index)
    n_00 = n_total_doc - (n_10 + n_docs_distribution[sample_index])

    if verbose:
        print('For feature_index:{} sample_index:{}'.format(feature_index, sample_index))
        print('n_11:{} n_01:{} n_10:{} n_00:{}'.format(
            n_11,
            n_01,
            n_10,
            n_00
        ))

    if n_11 == 0.0 or n_10 == 0.0 or n_01 == 0.0 or n_00 == 0.0:
        return 0
    else:
        temp1 = n_11/n_total_doc * math.log((n_total_doc*n_11)/((n_10+n_11)*(n_01+n_11)), 2)
        temp2 = n_01/n_total_doc * math.log((n_total_doc*n_01)/((n_00+n_01)*(n_01+n_11)), 2)
        temp3 = n_10/n_total_doc * math.log((n_total_doc*n_10)/((n_10+n_11)*(n_00+n_10)), 2)
        temp4 = n_00/n_total_doc * math.log((n_total_doc*n_00)/((n_00+n_01)*(n_00+n_10)), 2)
        score = temp1 + temp2 + temp3 + temp4

        if score < 0:
            raise Exception('score under 0 is detected. Something strange in Input matrix. Check your input matrix.')

        return score