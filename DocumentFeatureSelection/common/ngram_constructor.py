from logging import getLogger, StreamHandler
from nltk import ngrams
import logging
import joblib
import sys

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)
logger = getLogger(__name__)
handler = StreamHandler()
logger.addHandler(handler)

python_version = sys.version_info


def SUB_FUNC_ngram_data_conversion(key, docs, n, joiner_string='_'):
    """This function converts list of tokens into list of n_grams tokens

    :param key: key name of document
    :param docs: lits of tokens
    :param n: n of n_gram
    """

    assert isinstance(key, str)
    assert isinstance(docs, list)
    assert isinstance(n, int)

    if python_version > (3, 0, 0):
        assert isinstance(joiner_string, str)
    else:
        assert isinstance(joiner_string, unicode)

    character_joiner = lambda ngram_tuple: joiner_string.join(ngram_tuple)
    generate_nGram = lambda ngram_d: [character_joiner(g) for g in ngram_d]

    new_docs = [
        generate_nGram(ngrams(d, n))
        for d in docs
    ]

    assert isinstance(new_docs, list)
    assert isinstance(new_docs[0], list)

    return (key, new_docs)


def ngram_constructor(labeled_documents, ngram, n_jobs):

    logger.debug(msg='Now making {}-gram data strucutre with n(process) = {}'.format(ngram, n_jobs))
    key_docs_tuples = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(SUB_FUNC_ngram_data_conversion)(
            key=key,
            docs=docs,
            n=ngram,
            joiner_string='_'
        )
        for key, docs in labeled_documents.items()
    )
    reconstructed_labeled_documents = dict(key_docs_tuples)
    logger.debug(msg='Finished making N-gram')

    return reconstructed_labeled_documents