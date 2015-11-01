from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from .pmi import pmi_single_process_main
from .pmi_csr_matrix import make_pmi_matrix
__author__ = 'kensuke-mi'
__all__ = ['get_pmi_score']


def get_pmi_score(labeled_structure, logger, cut_zero=False, ngram=1, outformat='items'):
    csr_matrix, label_group_dict, vocabulary = make_pmi_matrix(labeled_structure, logger, ngram)
    pmi_score_objects = pmi_single_process_main(csr_matrix, vocabulary, label_group_dict, logger, outformat=outformat, cut_zero=cut_zero)

    return pmi_score_objects




