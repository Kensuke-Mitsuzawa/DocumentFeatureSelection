from collections import namedtuple
from collections import Counter
from DocumentFeatureSelection.common import utils
from DocumentFeatureSelection.models import SetDocumentInformation
from DocumentFeatureSelection import init_logger
from typing import Dict, List, Tuple, Any, Union
import logging
import joblib
import numpy
import pickle
logger = init_logger.init_logger(logging.getLogger(init_logger.LOGGER_NAME))
N_FEATURE_SWITCH_STRATEGY = 1000000

def generate_document_dict(document_key:str,
                           documents:List[Union[List[str], Tuple[Any]]])->Tuple[str,Dict[Union[str,bytes], int]]:
    """This function gets Document-frequency count in given list of documents
    """
    assert isinstance(documents, list)
    word_frequencies = [Counter(document) for document in documents]
    document_frequencies = Counter()
    for word_frequency in word_frequencies: document_frequencies.update(word_frequency.keys())
    document_frequency_dict = dict(document_frequencies)

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
    type_flag = set([judge_feature_type(docs) for docs in labeled_documents.values()])
    if type_flag == set(['str']):
        feature_list = list(set(utils.flatten(labeled_documents.values())))
        feature_list = sorted(feature_list)
        max_lenght = max([len(s) for s in feature_list])
    elif type_flag == set(['tuple']):
        # make tuple into string
        feature_list = list(set(utils.flatten(labeled_documents.values())))
        feature_list = [pickle.dumps(feature_tuple) for feature_tuple in sorted(feature_list)]
        max_lenght = max([len(s) for s in feature_list]) + 10
    else:
        raise Exception('Your input data has various type of data. Detected types: {}'.format(type_flag))

    feature2id = numpy.array(list(enumerate(feature_list)), dtype=[('value', 'i8'), ('key','S{}'.format(max_lenght))])  # type: ndarray

    # make label: id dictionary structure
    label2id_dict = {}
    # make list of Term-Frequency
    feature_frequency = []
    document_index = 0

    for key, docs in sorted(labeled_documents.items(), key=lambda key_value_tuple: key_value_tuple[0]):
        label2id_dict.update({key: document_index})
        document_index += 1
        words_in_docs = utils.flatten(docs)
        if type_flag == set(['str']):
            term_freq = Counter(words_in_docs)
        elif type_flag == set(['tuple']):
            term_freq = {pickle.dumps(key): value for key,value in dict(Counter(words_in_docs)).items()}
        else:
            raise Exception()

        feature_frequency.append(
            numpy.array(
                [(index_tuple[1], index_tuple[0]) for index_tuple in enumerate(term_freq)],
                dtype=[('key', 'S{}'.format(max_lenght)), ('value', 'i8')]
            ))

    label_max_length = max([len(label) for label in label2id_dict.keys()]) + 10
    label2id = numpy.array(list(label2id_dict.items()), dtype=[('key', 'S{}'.format(label_max_length)), ('value', 'i8')])
    assert isinstance(feature2id, numpy.ndarray)
    assert isinstance(feature_frequency, list)
    assert isinstance(label2id, numpy.ndarray)
    return SetDocumentInformation(feature_frequency, label2id, feature2id)


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
        max_lenght = max([len(s) for s in feature_list])
    elif type_flag == set(['tuple']):
        # make tuple into string
        feature_list = list(set(utils.flatten(labeled_documents.values())))
        feature_list = [pickle.dumps(feature_tuple) for feature_tuple in sorted(feature_list)]
        max_lenght = max([len(s) for s in feature_list]) + 10
    else:
        raise Exception('Your input data has various type of data. Detected types: {}'.format(type_flag))

    feature2id = numpy.array(list(enumerate(feature_list)), dtype=[('value', 'i8'), ('key','S{}'.format(max_lenght))])  # type: ndarray

    # make label: id dictionary structure
    label2id_dict = {}
    # list of document-frequency array
    feature_frequency = []

    if joblib_backend == 'auto' and len(feature2id) >= N_FEATURE_SWITCH_STRATEGY:
        joblib_backend = 'threading'
    if joblib_backend == 'auto' and len(feature2id) < N_FEATURE_SWITCH_STRATEGY:
        joblib_backend = 'multiprocessing'

    counted_frequency = joblib.Parallel(n_jobs=n_jobs, backend=joblib_backend)(
        joblib.delayed(generate_document_dict)(key, docs)
        for key, docs in sorted(labeled_documents.items(), key=lambda key_value_tuple: key_value_tuple[0]))

    document_index = 0
    for doc_key_freq_tuple in counted_frequency:
        label2id_dict.update({doc_key_freq_tuple[0]: document_index})
        document_index += 1
        if type_flag == set(['str']):
            doc_freq = doc_key_freq_tuple[1]
        elif type_flag == set(['tuple']):
            doc_freq = {pickle.dumps(key): value for key,value in list(doc_key_freq_tuple[1].items())}
        else:
            raise Exception()
        feature_frequency.append(
            numpy.array(
                list(doc_freq.items()),
                dtype=[('key', 'S{}'.format(max_lenght)), ('value', 'i8')]
            ))
    label_max_length = max([len(label) for label in label2id_dict.keys()]) + 10
    label2id = numpy.array(list(label2id_dict.items()), dtype=[('key', 'S{}'.format(label_max_length)), ('value', 'i8')])
    return SetDocumentInformation(feature_frequency, label2id, feature2id)