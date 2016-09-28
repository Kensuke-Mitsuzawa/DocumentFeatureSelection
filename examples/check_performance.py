from DocumentFeatureSelection import interface
import nltk
import logging
import time
try:
    import line_profiler
except:
    raise ImportError('Install line_profiler first!')
logger = logging.getLogger('sample usage')
logger.level = logging.DEBUG


#@profile
def pmi_with_parallel(input_corpus):
    logging.debug(msg='With multiprocessing backend')
    start = time.time()
    scored_matrix_obj = interface.run_feature_selection(
        input_dict=input_corpus,
        method='pmi',
        n_jobs=-1,
        joblib_backend='multiprocessing'
    )
    elapsed_time = time.time() - start
    print ("elapsed_time with multiprocess:{} [sec]".format(elapsed_time))


#@profile
def pmi_with_threading(input_corpus):
    start = time.time()
    logging.debug(msg='With threading backend')
    scored_matrix_obj = interface.run_feature_selection(
        input_dict=input_corpus,
        method='pmi',
        n_jobs=-1,
        joblib_backend='threading'
    )
    elapsed_time = time.time() - start
    print ("elapsed_time with multiprocess:{} [sec]".format(elapsed_time))


def pmi_with_cython(input_corpus):
    logging.debug(msg='With cython is True')
    start = time.time()
    scored_matrix_obj = interface.run_feature_selection(
        input_dict=input_corpus,
        method='pmi',
        n_jobs=-1,
        use_cython=True
    )
    elapsed_time = time.time() - start
    print ("elapsed_time with cython:{} [sec]".format(elapsed_time))

from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import genesis
from nltk.corpus import abc

abs_corpus = abc.sents()
genesis_corpus = genesis.sents()
web_corpus = webtext.sents()
gutenberg_corpus = gutenberg.sents()

input_corpus = {
    'abs': list(abs_corpus),
    'genesis': list(genesis_corpus),
    'web': list(web_corpus),
    'gutenberg': list(gutenberg_corpus)
    }

pmi_with_cython(input_corpus)
pmi_with_parallel(input_corpus)
#pmi_with_threading(input_corpus)
