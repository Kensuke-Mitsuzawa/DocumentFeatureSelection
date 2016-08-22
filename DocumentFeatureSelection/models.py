from typing import Dict, List, Tuple, Union, Any, TypeVar
from scipy.sparse.csr import csr_matrix
from DocumentFeatureSelection.common import utils
FeatureType = TypeVar('T', str, Tuple[Any])


class SetDocumentInformation(object):
    def __init__(self, feature_frequency:Dict[FeatureType,int],
                 label2id_dict:Dict[str,int],
                 feature2id_dict:Dict[FeatureType, int]):
        self.feature_frequency = feature_frequency
        self.label2id_dict = label2id_dict
        self.feature2id_dict = feature2id_dict


class DataCsrMatrix(object):
    def __init__(self, csr_matrix_:csr_matrix,
                 label2id_dict:Dict[str, int],
                 vocabulary:Dict[FeatureType, int],
                 n_docs_distribution:List[int],
                 n_term_freq_distribution:List[int]):
        self.csr_matrix_ = csr_matrix_
        self.label2id_dict = label2id_dict
        self.vocabulary = vocabulary
        self.n_docs_distribution = n_docs_distribution
        self.n_term_freq_distribution = n_term_freq_distribution


class ScoredResultObject(object):
    def __init__(self,
                 scored_matrix:csr_matrix,
                 label2id_dict:Dict[str,int],
                 feature2id_dict=Dict[FeatureType,int],
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