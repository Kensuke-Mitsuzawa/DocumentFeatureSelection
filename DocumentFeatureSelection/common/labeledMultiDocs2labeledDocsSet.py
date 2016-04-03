from collections import namedtuple
from collections import Counter
from DocumentFeatureSelection.common import utils

SetDocumentInformation = namedtuple('SetDocumentInformation',
                                    ('feature_frequency', 'label2id_dict', 'vocaburary2id_dict'))


def generate_document_dict(documents):
    """This function gets Document-frequency count in given list of documents

    :param documents [[str]]:
    :return dict that represents document frequency:
    """
    assert isinstance(documents, list)
    assert isinstance(documents[0], list)
    V = list(set([t for d in documents for t in d]))
    document_frequency_dict = {}
    for v in V:
        binary_count = [1 for d in documents if v in d]
        document_frequency_dict[v] = sum(binary_count)

    assert isinstance(document_frequency_dict, dict)
    return document_frequency_dict


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
        words_in_docs = utils.flatten(labeled_documents.values())
        feature_frequency.append(dict(Counter(words_in_docs)))
        label2id_dict.update({key: document_index})
        document_index += 1

    assert isinstance(vocaburary2id_dict, dict)
    assert isinstance(feature_frequency, list)
    assert isinstance(label2id_dict, dict)
    return SetDocumentInformation(feature_frequency, label2id_dict, vocaburary2id_dict)


def multiDocs2DocFreqInfo(labeled_documents):
    """This function generates information for constructing document-frequency matrix.


    :param labeled_structure:
    :return:
    """
    assert isinstance(labeled_documents, dict)

    vocabulary_list = list(set(utils.flatten(labeled_documents.values())))
    vocabulary_list = sorted(vocabulary_list)

    vocaburary2id_dict = {t: index for index, t in enumerate(vocabulary_list)}

    # make label: id dictionary structure
    label2id_dict = {}
    # make list of document-frequency
    feature_frequency = []
    document_index = 0

    for key, docs in sorted(labeled_documents.items(), key=lambda key_value_tuple: key_value_tuple[0]):
        feature_frequency.append(generate_document_dict(docs))
        label2id_dict.update({key: document_index})
        document_index += 1

    assert isinstance(vocaburary2id_dict, dict)
    assert isinstance(feature_frequency, list)
    assert isinstance(label2id_dict, dict)
    return SetDocumentInformation(feature_frequency, label2id_dict, vocaburary2id_dict)