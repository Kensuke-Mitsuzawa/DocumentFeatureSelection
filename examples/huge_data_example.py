from nltk.corpus import gutenberg
from nltk.corpus import webtext
from nltk.corpus import genesis
from nltk.corpus import abc
from DocumentFeatureSelection import interface
from DocumentFeatureSelection.models import PersistentDict
from sqlitedict import SqliteDict
import time
import os

"""This example shows you how to work on huge dataset.
For persisted-dict object you can choose PersistentDict or SqliteDict
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

# Case of PersistentDict
persistent_dict_obj = PersistentDict('demo.json', 'c', format='json')
persistent_dict_obj['abc'] = list(abc_corpus)
persistent_dict_obj['genesis'] = list(genesis_corpus)
persistent_dict_obj['web'] = list(web_corpus)
persistent_dict_obj['gutenberg'] = list(gutenberg_corpus)

start = time.time()
# If you put is_use_cache=True, it uses cache object for keeping huge objects during computation
# If you put is_use_memmap=True, it uses memmap for keeping matrix during computation
scored_matrix_obj = interface.run_feature_selection(
        input_dict=persistent_dict_obj,
        method='pmi',
        use_cython=True,
        is_use_cache=True,
        is_use_memmap=True
    )
elapsed_time = time.time() - start
print ("elapsed_time with cython:{} [sec]".format(elapsed_time))

# Case of SqliteDict
persisten_sqlite3_dict_obj = SqliteDict('./my_db.sqlite', autocommit=True)
persisten_sqlite3_dict_obj['abc'] = list(abc_corpus)
persisten_sqlite3_dict_obj['genesis'] = list(genesis_corpus)
persisten_sqlite3_dict_obj['web'] = list(web_corpus)
persisten_sqlite3_dict_obj['gutenberg'] = list(gutenberg_corpus)

start = time.time()
scored_matrix_obj_ = interface.run_feature_selection(
        input_dict=persisten_sqlite3_dict_obj,
        method='pmi',
        use_cython=True
    )
elapsed_time = time.time() - start
print ("elapsed_time with cython:{} [sec]".format(elapsed_time))
os.remove('./my_db.sqlite')