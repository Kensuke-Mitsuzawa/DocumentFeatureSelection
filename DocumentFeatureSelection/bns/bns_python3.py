from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin
from scipy.stats import norm
from logging import getLogger, StreamHandler
from numpy import ndarray
import numpy as np
import joblib
import logging

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)


class BNS(TransformerMixin):
    def __init__(self):
        pass

    def __check_matrix_form(self, X):
        assert isinstance(X, csr_matrix)
        matrix_size = X.shape
        n_categories = matrix_size[0]
        if n_categories != 2:
            raise Exception('BNS input must be of 2 categories')

    def fit_transform(self, X, y=None, **fit_params):
        assert isinstance(X, csr_matrix)

        if not 'unit_distribution' in fit_params:
            raise Exception('You must put unit_distribution parameter')
        assert isinstance(fit_params['unit_distribution'], (list, ndarray))
        self.__check_matrix_form(X)

        unit_distribution = fit_params['unit_distribution']

        if 'n_jobs' in fit_params:
            n_jobs = fit_params['n_jobs']
        else:
            n_jobs = 1

        if 'true_index' in fit_params:
            true_index = fit_params['true_index']
        else:
            true_index = 0

        if 'verbose' in fit_params:
            verbose = True
        else:
            verbose = False

        matrix_size = X.shape
        sample_range = list(range(0, matrix_size[0]))
        feature_range = list(range(0, matrix_size[1]))

        logger.debug(msg='Start calculating BNS with n(process)={}'.format(n_jobs))
        logger.debug(msg='size(input_matrix)={} * {}'.format(X.shape[0], X.shape[1]))

        bns_score_csr_source = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self.docId_word_BNS)(
                X=X,
                feature_index=feature_index,
                sample_index=sample_index,
                true_index=true_index,
                unit_distribution=unit_distribution,
                verbose=verbose
            )
            for sample_index in sample_range
            for feature_index in feature_range
        )

        row_list = [t[0] for t in bns_score_csr_source]
        col_list = [t[1] for t in bns_score_csr_source]
        data_list = [t[2] for t in bns_score_csr_source]

        bns_featured_csr_matrix = csr_matrix((data_list, (row_list, col_list)),
                                             shape=(X.shape[0],
                                                    X.shape[1]))

        logging.debug(msg='End calculating BNS')

        return bns_featured_csr_matrix

    def docId_word_BNS(self, X, feature_index, sample_index, unit_distribution, true_index, verbose=False):

        assert isinstance(X, csr_matrix)
        assert isinstance(feature_index, int)
        assert isinstance(sample_index, int)

        bns_score = self.bns(
            X=X,
            feature_index=feature_index,
            sample_index=sample_index,
            true_index=true_index,
            unit_distribution=unit_distribution,
            verbose=verbose
        )
        return sample_index, feature_index, bns_score

    def bns(self, X, feature_index, sample_index, unit_distribution, true_index=0, verbose=False):
        if true_index==0:
            false_index = 1
        elif true_index==1:
            false_index = 0
        else:
            raise Exception('true index must be either of 0 or 1')

        # trueラベルで出現した回数
        # tp is frequency of features in the specified positive label
        tp = X[true_index, feature_index]
        # trueラベルで出現しなかった回数
        # fp is frequency of NON-features(expect specified feature) in the specified positive label
        fp = unit_distribution[true_index] - tp

        # negativeラベルで出現した回数
        # fn is frequency of features in the specified negative label
        fn = X[false_index, feature_index]
        # negativeラベルで出現しなかった回数
        # fp is frequency of NON-features(expect specified feature) in the specified negative label
        tn = unit_distribution[false_index] - fn

        if tn < 0.0:
            print('aaaa')

        pos = tp + fn
        neg = fp + tn

        tpr = tp / pos
        fpr = fp / neg

        if verbose:
            logging.debug('For feature_index:{} sample_index:{}'.format(feature_index, sample_index))
            logging.debug('tp:{} fp:{} fn:{} tn:{} pos:{} neg:{} tpr:{} fpr:{}'.format(
                tp,
                fp,
                fn,
                tn,
                pos,
                neg,
                tpr,
                fpr
            ))

        bns_score = np.abs(norm.ppf(norm.cdf(tpr)) - norm.ppf(norm.cdf(fpr)))
        return bns_score