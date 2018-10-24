from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin
from scipy.stats import norm
from numpy import ndarray, memmap
from typing import Union
from DocumentFeatureSelection.init_logger import logger
import numpy as np
import joblib
import logging


def bns(X:Union[memmap, csr_matrix],
        feature_index: int,
        sample_index: int,
        unit_distribution: np.ndarray,
        true_index: int = 0,
        verbose: bool = False):
    if true_index == 0:
        false_index = 1
    elif true_index == 1:
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

    #bns_score = np.abs(norm.ppf(norm.cdf(tpr)) - norm.ppf(norm.cdf(fpr)))
    bns_score = abs(norm.ppf(tpr) - norm.ppf(fpr))
    return bns_score



class BNS(TransformerMixin):
    def __init__(self):
        pass

    def __check_matrix_form(self, X):
        assert isinstance(X, csr_matrix)
        matrix_size = X.shape
        n_categories = matrix_size[0]
        if n_categories != 2:
            raise Exception('BNS input must be of 2 categories')

    def fit_transform(self,
                      X: Union[memmap, csr_matrix],
                      y=None,
                      **fit_params):
        """* What you can do

        * Args
        - X; scipy.csr_matrix or numpy.memmap: Matrix object

        * Params
        - unit_distribution; list or ndarray: The number of document frequency per label. Ex. [10, 20]
        - n_jobs: The number of cores when you use joblib.
        - joblib_backend: "multiprocessing" or "multithreding"
        - true_index: The index number of True label.
        - use_cython; boolean: True, then Use Cython for computation. False, not.
        """
        assert isinstance(X, csr_matrix)

        # --------------------------------------------------------
        # Check parameters
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

        if 'joblib_backend' in fit_params:
            joblib_backend = fit_params['joblib_backend']
        else:
            joblib_backend = 'multiprocessing'

        if 'use_cython' in fit_params:
            is_use_cython = True
        else:
            is_use_cython = False
        # --------------------------------------------------------

        matrix_size = X.shape
        sample_range = list(range(0, matrix_size[0]))
        feature_range = list(range(0, matrix_size[1]))

        logger.debug(msg='Start calculating BNS with n(process)={}'.format(n_jobs))
        logger.debug(msg='size(input_matrix)={} * {}'.format(X.shape[0], X.shape[1]))

        if is_use_cython:
            import pyximport; pyximport.install()
            from DocumentFeatureSelection.bns.bns_cython import main
            logger.warning(msg='n_jobs parameter is invalid when use_cython=True')
            bns_score_csr_source = main(
                X=X,
                unit_distribution=unit_distribution,
                sample_range=sample_range,
                feature_range=feature_range,
                true_index=true_index,
                verbose=verbose
            )
        else:
            bns_score_csr_source = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(
                joblib.delayed(self.docId_word_BNS)(
                X=X,
                feature_index=feature_index,
                sample_index=sample_index,
                true_index=true_index,
                unit_distribution=unit_distribution,
                verbose=verbose
            )
            for sample_index in sample_range
            for feature_index in feature_range)

        row_list = [t[0] for t in bns_score_csr_source]
        col_list = [t[1] for t in bns_score_csr_source]
        data_list = [t[2] for t in bns_score_csr_source]

        bns_featured_csr_matrix = csr_matrix((data_list, (row_list, col_list)),
                                             shape=(X.shape[0],
                                                    X.shape[1]))

        logging.debug(msg='End calculating BNS')

        return bns_featured_csr_matrix

    def docId_word_BNS(self, X:csr_matrix,
                       feature_index:int,
                       sample_index:int,
                       unit_distribution:np.ndarray,
                       true_index:int,
                       verbose=False):

        assert isinstance(X, csr_matrix)
        assert isinstance(feature_index, int)
        assert isinstance(sample_index, int)

        bns_score = bns(
            X=X,
            feature_index=feature_index,
            sample_index=sample_index,
            true_index=true_index,
            unit_distribution=unit_distribution,
            verbose=verbose
        )
        return sample_index, feature_index, bns_score