from collections import namedtuple
from collections import Counter
from DocumentFeatureSelection.common import utils
from DocumentFeatureSelection.models import SetDocumentInformation
from DocumentFeatureSelection import init_logger
from typing import Dict, List, Tuple, Any, Union
import logging
import joblib
logger = init_logger.init_logger(logging.getLogger(init_logger.LOGGER_NAME))
N_FEATURE_SWITCH_STRATEGY = 1000000

def generate_document_dict(document_key:str,
                           documents:List[Union[List[str], Tuple[Any]]])->Tuple[str,Dict[str, int]]:
    """This function gets Document-frequency count in given list of documents
    """
    assert isinstance(documents, list)
    word_frequencies = [Counter(document) for document in documents]
    document_frequencies = Counter()
    for word_frequency in word_frequencies: document_frequencies.update(word_frequency.keys())
    document_frequency_dict = dict(document_frequencies)
    '''
    V = set([t for d in documents for t in d])
    document_frequency_dict = {}
    for v in V:
        binary_count = [1 for d in documents if v in d]
        document_frequency_dict[v] = sum(binary_count)'''

    assert isinstance(document_frequency_dict, dict)
    return (document_key, document_frequency_dict)


def multiDocs2TermFreqInfo(labeled_documents):
    """This function generates information to construct term-frequency matrix

    :param labeled_structure:
    :return:
    """
    assert isinstance(labeled_documents, dict)

    vocabulary_list = list(set(utils.flatten(labeled_documents.values())))
    vocabulary_list = sorted(vocabulary_list)

    vocaburary2id_dict = {t: index for index, t in enumerate(vocabulary_list)}

    # make label: id dictionary structure
    label2id_dict = {}
    # make list of Term-Frequency
    feature_frequency = []
    document_index = 0

    for key, docs in sorted(labeled_documents.items(), key=lambda key_value_tuple: key_value_tuple[0]):
        words_in_docs = utils.flatten(docs)
        feature_frequency.append(dict(Counter(words_in_docs)))
        label2id_dict.update({key: document_index})
        document_index += 1

    assert isinstance(vocaburary2id_dict, dict)
    assert isinstance(feature_frequency, list)
    assert isinstance(label2id_dict, dict)
    return SetDocumentInformation(feature_frequency, label2id_dict, vocaburary2id_dict)


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
                raise TypeError('Feature object should be either of str or tuple')
    return type_flag


def multiDocs2DocFreqInfo(labeled_documents:Dict[str, List[List[Union[str, Tuple[Any]]]]],
                          joblib_backend:str='auto',
                          n_jobs:int=1)->SetDocumentInformation:
    """This function generates information for constructing document-frequency matrix.
    """
    assert isinstance(labeled_documents, dict)
    type_flag = set([judge_feature_type(docs) for docs in labeled_documents.values()])
    assert len(type_flag)==1

    if type_flag == set(['str']):
        feature_list = list(set(utils.flatten(labeled_documents.values())))
        feature_list = sorted(feature_list)
    elif type_flag == set(['tuple']):
        feature_list = list(set(utils.flatten(labeled_documents.values())))
        feature_list = sorted(feature_list)
    else:
        raise Exception('Your input data has various type of data. Detected types: {}'.format(type_flag))

    feature2id_dict = {t: index for index, t in enumerate(feature_list)}

    # make label: id dictionary structure
    label2id_dict = {}
    # make list of document-frequency
    feature_frequency = []

    if joblib_backend == 'auto' and len(feature2id_dict) >= N_FEATURE_SWITCH_STRATEGY:
        joblib_backend = 'threading'
    if joblib_backend == 'auto' and len(feature2id_dict) < N_FEATURE_SWITCH_STRATEGY:
        joblib_backend = 'multiprocessing'

    counted_frequency = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(
        joblib.delayed(generate_document_dict)(key, docs)
        for key, docs in sorted(labeled_documents.items(), key=lambda key_value_tuple: key_value_tuple[0]))

    document_index = 0
    for doc_key_freq_tuple in counted_frequency:
        label2id_dict.update({doc_key_freq_tuple[0]: document_index})
        document_index += 1
        feature_frequency.append(doc_key_freq_tuple[1])

    return SetDocumentInformation(feature_frequency, label2id_dict, feature2id_dict)