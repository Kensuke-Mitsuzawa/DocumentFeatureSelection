from typing import Dict, List, Tuple, Union, Any, TypeVar
from scipy.sparse.csr import csr_matrix
from DocumentFeatureSelection.common import utils
from numpy.core.multiarray import array, ndarray
FeatureType = TypeVar('T', str, Tuple[Any])

class SetDocumentInformation(object):
    __slots__ = ['matrix_object', 'label2id', 'feature2id']

    def __init__(self, matrix_object:Union[csr_matrix, ndarray],
                 label2id:Dict[str,int],
                 feature2id:Dict[str,int]):
        self.matrix_object = matrix_object
        self.label2id = label2id
        self.feature2id = feature2id


class DataCsrMatrix(object):
    __slots__ = ['csr_matrix_', 'label2id_dict', 'vocabulary', 'n_docs_distribution', 'n_term_freq_distribution']

    def __init__(self, csr_matrix_:csr_matrix,
                 label2id_dict:Dict[str,int],
                 vocabulary:Dict[str,int],
                 n_docs_distribution:ndarray,
                 n_term_freq_distribution:ndarray):
        self.csr_matrix_ = csr_matrix_
        self.label2id_dict = label2id_dict
        self.vocabulary = vocabulary
        self.n_docs_distribution = n_docs_distribution
        self.n_term_freq_distribution = n_term_freq_distribution


class ScoredResultObject(object):
    def __init__(self,
                 scored_matrix:csr_matrix,
                 label2id_dict:ndarray,
                 feature2id_dict=ndarray,
                 method:str=None,
                 matrix_form:str=None):
        self.scored_matrix = scored_matrix
        self.label2id_dict = label2id_dict
        self.feature2id_dict = feature2id_dict
        self.method = method
        self.matrix_form = matrix_form

    def __conv_into_dict_format(self, word_score_items):
        out_format_structure = {}
        for item in word_score_items:
            if item['label'] not in out_format_structure :
                out_format_structure[item['label']] = [{'word': item['word'], 'score': item['score']}]
            else:
                out_format_structure[item['label']].append({'word': item['word'], 'score': item['score']})
        return out_format_structure

    def ScoreMatrix2ScoreDictionary(self,
                                    outformat:str='items',
                                    sort_desc:bool=True,
                                    n_jobs:int=1):
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

        """

        scored_objects = utils.get_feature_dictionary(
            weighted_matrix=self.scored_matrix,
            vocabulary=self.feature2id_dict,
            label_group_dict=self.label2id_dict,
            n_jobs=n_jobs
        )

        if sort_desc: scored_objects = \
            sorted(scored_objects, key=lambda x: x['score'], reverse=True)

        if outformat=='dict':
            out_format_structure = self.__conv_into_dict_format(scored_objects)
        elif outformat=='items':
            out_format_structure = scored_objects
        else:
            raise ValueError('outformat must be either of {dict, items}')

        return out_format_structure