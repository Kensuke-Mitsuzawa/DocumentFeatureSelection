from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import genesis
from nltk.corpus import abc
from DocumentFeatureSelection import interface
from DocumentFeatureSelection.models import PersistentDict
import time

"""This example shows you how to work on huge dataset.
You're supposed to be ready to use following corpora object in nltk
- abc
- genesis
- web
- gutenberg
"""

#----------------------------------------------------------
abc_corpus = abc.sents()
genesis_corpus = genesis.sents()
web_corpus = webtext.sents()
gutenberg_corpus = gutenberg.sents()

#shelve_object = PersistentDict('demo.json', 'c', format='json')
from sqlitedict import SqliteDict
shelve_object = SqliteDict('./my_db.sqlite', autocommit=True)
shelve_object['abc'] = list(abc_corpus)
shelve_object['genesis'] = list(genesis_corpus)
shelve_object['web'] = list(web_corpus)
shelve_object['gutenberg'] = list(gutenberg_corpus)

start = time.time()
scored_matrix_obj = interface.run_feature_selection(
        input_dict=shelve_object,
        method='pmi',
        use_cython=True
    )
elapsed_time = time.time() - start
print ("elapsed_time with cython:{} [sec]".format(elapsed_time))