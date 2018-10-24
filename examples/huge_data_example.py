#! -*- coding: utf-8 -*-
from DocumentFeatureSelection import interface
from DocumentFeatureSelection.models import PersistentDict
from DocumentFeatureSelection.init_logger import logger
import logging
import time
import os
import nltk
nltk.download('wordnet')
from collections import Counter
from nltk import stem
from typing import List
# make download 20news group file
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
lemmatizer = stem.WordNetLemmatizer()
logger.setLevel(logging.DEBUG)

"""This example shows you how to work on huge dataset.
For persisted-dict object you can choose PersistentDict or SqliteDict
"""

DATA_LIMIT = 100000


def run_nltk_lemma(subject_name: str)->List[str]:
    return [lemmatizer.lemmatize(t).strip(':?!><') for t in subject_name.lower().split()]


category_names = newsgroups_train.target_names
logger.debug("20-news has {} categories".format(len(category_names)))
logger.debug("Now pre-processing on subject text...")
news_lemma = [run_nltk_lemma(d) for d in newsgroups_train.data[:DATA_LIMIT]]

index2category = {i: t for i, t in enumerate(newsgroups_train.target_names)}
dict_index2label = {i: index2category[t_no] for i, t_no in enumerate(newsgroups_train.target[:DATA_LIMIT])}
logger.info("Subject distribution")
for k, v in dict(Counter(dict_index2label.values())).items():
    logger.info("{} is {}, {}%".format(k, v, v / len(dict_index2label) * 100))

# Case of PersistentDict
logger.info("Putting documents into dict object...")
persistent_dict_obj = PersistentDict('demo.json', 'c', format='json')
for i, label in dict_index2label.items():
    if label in persistent_dict_obj:
        persistent_dict_obj[label].append(news_lemma[i])
    else:
        persistent_dict_obj[label] = [news_lemma[i]]
else:
    persistent_dict_obj.sync()

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
logger.info("elapsed_time with cython: {} [sec]".format(elapsed_time))
os.remove('./demo.json')
