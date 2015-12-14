#! -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from logging import Logger
from ..pmi import pmi_single_process_main
from ..pmi.pmi_csr_matrix import make_pmi_matrix
from . import tf_idf
__author__ = 'kensuke-mi'



class TFIDF(object):
    """This class generates PMI featured matrix.

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


    :param int ngram: n parameter to generate n-gram from given dataset
    :param logging.Logger logger:
    """
    def __init__(self, logger, ngram=1, norm_metric='l2', use_idf_bool=True, smooth_idf_bool=True, sublinear_tf_bool=False):
        assert isinstance(ngram, int)
        assert isinstance(logger, Logger)
        self.n_gram = ngram
        self.logger = logger
        self.transformer = tf_idf.fit_transform
        self.norm_metric = norm_metric
        self.use_idf_bool = use_idf_bool
        self.smooth_idf_bool = smooth_idf_bool
        self.sublinear_tf_bool = sublinear_tf_bool

    def fit_transform(self, labeled_structure):

        csr_matrix, label_group_dict, vocabulary = make_pmi_matrix(labeled_structure, self.logger, self.n_gram)

        self.feature_matrix = csr_matrix
        self.label_group_dict = label_group_dict
        self.vocabulary = vocabulary

        tf_idf_matrix = self.transformer(
            csr_matrix=csr_matrix,
            norm=self.norm_metric,
            use_idf=self.use_idf_bool,
            smooth_idf=self.smooth_idf_bool,
            sublinear_tf=self.sublinear_tf_bool
        )
        self.weighed_matrix = tf_idf_matrix

        return tf_idf_matrix

    def get_feature_dictionary(self, outformat='items', cut_zero=False, n_job=1):
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


        :param string outformat: format type of output dictionary. You can choose 'items' or 'dict'
        :param bool cut_zero: return all result or not. If cut_zero = True, the method cuts zero features.
        """
        if not hasattr(self, "feature_matrix") or not hasattr(self, "label_group_dict") or not hasattr(self, "vocabulary"):
            raise AttributeError("You need to call 'fit_transform()' method before calling this method.")

        pmi_score_objects = pmi_single_process_main(self.feature_matrix,
                                                    self.vocabulary,
                                                    self.label_group_dict,
                                                    self.logger,
                                                    outformat=outformat,
                                                    cut_zero=cut_zero)

        return pmi_score_objects