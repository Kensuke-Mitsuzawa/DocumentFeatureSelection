from scipy.sparse import csr_matrix
from logging import getLogger, StreamHandler
import logging
import joblib
import math

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)

__author__ = 'kensuke-mi'


class SOA(object):
    def __init__(self):
        pass

    def fit_transform(self, X, unit_distribution, n_jobs=1, verbose=False):
        assert isinstance(X, csr_matrix)
        assert isinstance(unit_distribution, list)

        matrix_size = X.shape
        sample_range = list(range(0, matrix_size[0]))
        feature_range = list(range(0, matrix_size[1]))
        n_total_document = sum(unit_distribution)

        logger.debug(msg='Start calculating SOA with n(process)={}'.format(n_jobs))
        logger.debug(msg='size(input_matrix)={} * {}'.format(X.shape[0], X.shape[1]))

        pmi_score_csr_source = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self.docId_word_soa)(
                X=X,
                unit_distribution=unit_distribution,
                feature_index=feature_index,
                sample_index=sample_index,
                n_total_doc=n_total_document,
                verbose=verbose
            )
            for sample_index in sample_range
            for feature_index in feature_range
        )

        row_list = [t[0] for t in pmi_score_csr_source]
        col_list = [t[1] for t in pmi_score_csr_source]
        data_list = [t[2] for t in pmi_score_csr_source]

        soa_featured_csr_matrix = csr_matrix((data_list, (row_list, col_list)),
                                             shape=(X.shape[0],
                                                    X.shape[1]))

        logging.debug(msg='End calculating SOA')

        return soa_featured_csr_matrix


    def docId_word_soa(self, X, unit_distribution, n_total_doc, feature_index, sample_index, verbose=False):
        """
        """
        assert isinstance(X, csr_matrix)
        assert isinstance(unit_distribution, list)
        assert isinstance(feature_index, int)
        assert isinstance(sample_index, int)

        soa_score = self.soa(
            X=X,
            unit_distribution=unit_distribution,
            feature_index=feature_index,
            sample_index=sample_index,
            n_total_docs=n_total_doc,
            verbose=verbose
        )
        return sample_index, feature_index, soa_score

    def soa(self, X, unit_distribution, n_total_docs, feature_index, sample_index, verbose=False):
        # X is either of term-frequency matrix per label or document-frequency per label
        assert isinstance(X, csr_matrix)
        assert isinstance(unit_distribution, list)
        assert isinstance(feature_index, int)
        assert isinstance(sample_index, int)

        matrix_size = X.shape
        NOT_sample_indexes = [i for i in range(0, matrix_size[0]) if i != sample_index]

        # freq_w_e is term-frequency(or document-frequency) of w in the unit having the specific label e
        freq_w_e = X[sample_index, feature_index]
        # freq_w_not_e is term-frequency(or document-frequency) of w in units except the specific label e
        freq_w_not_e = X[NOT_sample_indexes, feature_index].sum()
        # freq_e is the number of the unit having specific label e
        freq_e = unit_distribution[sample_index]
        # freq_not_e is the number of the unit NOT having the specific label e
        freq_not_e = n_total_docs - freq_e

        if verbose:
            logging.debug('For feature_index:{} sample_index:{}'.format(feature_index, sample_index))
            logging.debug('freq_w_e:{} freq_w_not_e:{} freq_e:{} freq_not_e:{}'.format(
                freq_w_e,
                freq_w_not_e,
                freq_e,
                freq_not_e
            ))

        if freq_w_e == 0 or freq_w_not_e == 0 or freq_e == 0 or freq_not_e == 0:
            return 0
        else:
            nominator = (float(freq_w_e) * freq_not_e)
            denominator = (float(freq_e) * freq_w_not_e)
            ans = nominator / denominator
            assert isinstance(ans, float)
            soa_val = math.log(ans, 2)
            return soa_val