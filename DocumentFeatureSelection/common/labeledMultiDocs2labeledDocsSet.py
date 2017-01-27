from collections import Counter
from DocumentFeatureSelection.models import SetDocumentInformation, AvailableInputTypes, PersistentDict
from DocumentFeatureSelection import init_logger
from DocumentFeatureSelection.common.utils import init_cache_object
from sklearn.feature_extraction import DictVectorizer
from typing import Dict, List, Tuple, Any, Union
from sqlitedict import SqliteDict
import logging
import joblib
import itertools
import tempfile
logger = init_logger.init_logger(logging.getLogger(init_logger.LOGGER_NAME))
N_FEATURE_SWITCH_STRATEGY = 1000000

'''
def decode_into_utf8(string:str)->bytes:
    """* what you can do
    - convert string into etf-8
    """
    return string.encode('utf-8')'''

def generate_document_dict(document_key:str,
                           documents:List[Union[List[str], Tuple[Any]]])->Tuple[str,Counter]:
    """This function gets Document-frequency count in given list of documents
    """
    assert isinstance(documents, list)
    word_frequencies = [Counter(document) for document in documents]
    document_frequencies = Counter()
    for word_frequency in word_frequencies: document_frequencies.update(word_frequency.keys())

    return (document_key, document_frequencies)


def multiDocs2TermFreqInfo(labeled_documents:AvailableInputTypes,
                           is_use_cache:bool=True,
                           path_work_dir:str=tempfile.mkdtemp()):
    """This function generates information to construct term-frequency matrix
    """
    assert isinstance(labeled_documents, (SqliteDict, dict))

    counted_frequency = [(label, Counter(list(itertools.chain.from_iterable(documents))))
                         for label, documents in labeled_documents.items()]
    feature_documents = [dict(label_freqCounter_tuple[1]) for label_freqCounter_tuple in counted_frequency]


    if is_use_cache:
        dict_matrix_index = init_cache_object('matrix_element_objects', path_work_dir=path_work_dir)
    else:
        dict_matrix_index = {}

    # use sklearn feature-extraction
    vec = DictVectorizer()
    dict_matrix_index['matrix_object'] = vec.fit_transform(feature_documents).tocsr()
    dict_matrix_index['feature2id'] = {feat:feat_id for feat_id, feat in enumerate(vec.get_feature_names())}
    dict_matrix_index['label2id'] = {label_freqCounter_tuple[0]:label_id for label_id, label_freqCounter_tuple in  enumerate(counted_frequency)}

    return SetDocumentInformation(dict_matrix_index)


def judge_feature_type(docs:List[List[Union[str, Tuple[Any]]]])->str:
    type_flag = None
    for document_list in docs:
        assert isinstance(document_list, list)
        for feature in document_list:
            if isinstance(feature, str):
                type_flag = 'str'
            elif isinstance(feature, tuple):
                type_flag = 'tuple'
            else:
                logger.error(msg=docs)
                raise TypeError('Feature object should be either of str or tuple')
    return type_flag


def multiDocs2DocFreqInfo(labeled_documents:AvailableInputTypes,
                          n_jobs:int=1,
                          path_working_dir:str=tempfile.mkdtemp(),
                          is_use_cache: bool = True)->SetDocumentInformation:
    """This function generates information for constructing document-frequency matrix.
    """
    assert isinstance(labeled_documents, (SqliteDict, dict))
    type_flag = set([judge_feature_type(docs) for docs in labeled_documents.values()])
    assert len(type_flag)==1

    counted_frequency = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(generate_document_dict)(key, docs)
        for key, docs in sorted(labeled_documents.items(), key=lambda key_value_tuple: key_value_tuple[0]))

    ### construct [{}] structure for input of DictVectorizer() ###
    seq_feature_documents = (dict(label_freqCounter_tuple[1]) for label_freqCounter_tuple in counted_frequency)

    ### Save index-string dictionary
    if is_use_cache:
        dict_matrix_index = init_cache_object('matrix_element_object', path_working_dir)
    else:
        dict_matrix_index = {}

    # use sklearn feature-extraction
    vec = DictVectorizer()
    dict_matrix_index['matrix_object'] = vec.fit_transform(seq_feature_documents).tocsr()
    dict_matrix_index['feature2id'] = {feat:feat_id for feat_id, feat in enumerate(vec.get_feature_names())}
    dict_matrix_index['label2id'] = {label_freqCounter_tuple[0]:label_id for label_id, label_freqCounter_tuple in enumerate(counted_frequency)}

    return SetDocumentInformation(dict_matrix_index)