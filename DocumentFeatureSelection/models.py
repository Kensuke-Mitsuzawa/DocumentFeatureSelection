from typing import Dict, List, Tuple, Union, Any, TypeVar
from scipy.sparse.csr import csr_matrix
from numpy.core.multiarray import array, ndarray
from numpy import memmap
from sqlitedict import SqliteDict
from tempfile import mkdtemp
from DocumentFeatureSelection import init_logger
from numpy import ndarray, int32, int64
import typing
import pickle, json, csv, os, shutil
import logging
import collections
logger = init_logger.init_logger(logging.getLogger(init_logger.LOGGER_NAME))


# this class is from https://code.activestate.com/recipes/576642/
class PersistentDict(dict):
    ''' Persistent dictionary with an API compatible with shelve and anydbm.

    The dict is kept in memory, so the dictionary operations run as fast as
    a regular dictionary.

    Write to disk is delayed until close or sync (similar to gdbm's fast mode).

    Input file format is automatically discovered.
    Output file format is selectable between pickle, json, and csv.
    All three serialization formats are backed by fast C implementations.

    '''

    def __init__(self, filename, flag='c', mode=None, format='pickle', *args, **kwds):
        self.flag = flag                    # r=readonly, c=create, or n=new
        self.mode = mode                    # None or an octal triple like 0644
        self.format = format                # 'csv', 'json', or 'pickle'
        self.filename = filename
        if flag != 'n' and os.access(filename, os.R_OK):
            fileobj = open(filename, 'rb' if format=='pickle' else 'r')
            with fileobj:
                self.load(fileobj)
        dict.__init__(self, *args, **kwds)

    def sync(self):
        'Write dict to disk'
        if self.flag == 'r':
            return
        filename = self.filename
        tempname = filename + '.tmp'
        fileobj = open(tempname, 'wb' if self.format=='pickle' else 'w')
        try:
            self.dump(fileobj)
        except Exception:
            os.remove(tempname)
            raise
        finally:
            fileobj.close()
        shutil.move(tempname, self.filename)    # atomic commit
        if self.mode is not None:
            os.chmod(self.filename, self.mode)

    def close(self):
        self.sync()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def dump(self, fileobj):
        if self.format == 'csv':
            csv.writer(fileobj).writerows(self.items())
        elif self.format == 'json':
            json.dump(self, fileobj, separators=(',', ':'))
        elif self.format == 'pickle':
            pickle.dump(dict(self), fileobj, 2)
        else:
            raise NotImplementedError('Unknown format: ' + repr(self.format))

    def load(self, fileobj):
        # try formats from most restrictive to least restrictive
        for loader in (pickle.load, json.load, csv.reader):
            fileobj.seek(0)
            try:
                return self.update(loader(fileobj))
            except Exception:
                pass
        raise ValueError('File not in a supported format')


class SetDocumentInformation(object):
    __slots__ = ['matrix_object', 'label2id', 'feature2id']

    def __init__(self, dict_matrix_index:Union[Dict[str,Any], SqliteDict, PersistentDict]):
        """
        * Keys
        - matrix_object:Union[csr_matrix, ndarray]
        - label2id: Dict[str, str]
        - feature2id: Dict[str, str]
        """
        if not "matrix_object" in dict_matrix_index:
            raise Exception("dict_matrix_index must have key='matrix_object'")
        if not "label2id" in dict_matrix_index:
            raise Exception("dict_matrix_index must have key='label2id'")
        if not "feature2id" in dict_matrix_index:
            raise Exception("dict_matrix_index must have key='feature2id'")

        self.matrix_object = dict_matrix_index['matrix_object']
        self.label2id = dict_matrix_index['label2id']
        self.feature2id = dict_matrix_index['feature2id']


class DataCsrMatrix(object):
    """* What you can do
    - You can keep information for keeping matrix object.

    * Attributes
    - vocaburary is, dict object with token: feature_id
    >>> {'I_aa_hero': 4, 'xx_xx_cc': 1, 'I_aa_aa': 2, 'bb_aa_aa': 3, 'cc_cc_bb': 8}
    - label_group_dict is, dict object with label_name: label_id
    >>> {'label_b': 0, 'label_c': 1, 'label_a': 2}
    - csr_matrix is, sparse matrix from scipy.sparse
    """
    __slots__ = ['cache_backend', 'csr_matrix_',
                 'label2id_dict', 'vocabulary',
                 'n_docs_distribution', 'n_term_freq_distribution', 'path_working_dir']

    def __init__(self, csr_matrix_:csr_matrix,
                 label2id_dict:Dict[str,int],
                 vocabulary:Dict[str,int],
                 n_docs_distribution:ndarray,
                 n_term_freq_distribution:ndarray,
                 is_use_cache:bool=False,
                 is_use_memmap:bool=False,
                 cache_backend:str='PersistentDict',
                 path_working_dir:str=None):

        self.n_docs_distribution = n_docs_distribution
        self.n_term_freq_distribution = n_term_freq_distribution
        self.cache_backend = cache_backend

        if path_working_dir is None: self.path_working_dir = mkdtemp()
        else: self.path_working_dir = path_working_dir

        if is_use_cache:
            """You use disk-drive for keeping object.
            """
            path_vocabulary_cache_obj = os.path.join(self.path_working_dir, 'vocabulary.cache')
            path_label_2_dict_cache_obj = os.path.join(self.path_working_dir, 'label_2_dict.cache')
            self.vocabulary = self.initialize_cache_dict_object(path_vocabulary_cache_obj)
            self.vocabulary = vocabulary

            self.label2id_dict = self.initialize_cache_dict_object(path_label_2_dict_cache_obj)
            self.label2id_dict = label2id_dict
        else:
            """Keep everything on memory
            """
            self.label2id_dict = label2id_dict
            self.vocabulary = vocabulary

        if is_use_memmap:
            """You use disk-drive for keeping object
            """
            path_memmap_obj = os.path.join(self.path_working_dir, 'matrix.memmap')
            self.csr_matrix_ = self.initialize_memmap_object(csr_matrix_, path_memmap_object=path_memmap_obj)
        else:
            self.csr_matrix_ = csr_matrix_

    def initialize_cache_dict_object(self, path_cache_file):
        if self.cache_backend == 'PersistentDict':
            return PersistentDict(path_cache_file, flag='c', format='json')
        elif self.cache_backend == 'SqliteDict':
            return SqliteDict(path_cache_file, autocommit=True)
        else:
            raise Exception('No such cache_backend option named {}'.format(self.cache_backend))

    def initialize_memmap_object(self, matrix_object:csr_matrix, path_memmap_object:str)->memmap:
        fp = memmap(path_memmap_object, dtype='float64', mode='w+', shape=matrix_object.shape)
        fp[:] = matrix_object.todense()[:]
        return fp

    def __str__(self):
        return """matrix-type={}, matrix-size={}, path_working_dir={}""".format(type(self.csr_matrix_),
                   self.csr_matrix_.shape,
                   self.path_working_dir)


class ScoredResultObject(object):
    def __init__(self,
                 scored_matrix:csr_matrix,
                 label2id_dict:Union[Dict[str,Any], ndarray],
                 feature2id_dict=Union[Dict[str,Any], ndarray],
                 method:str=None,
                 matrix_form:str=None):
        self.scored_matrix = scored_matrix
        self.label2id_dict = label2id_dict
        self.feature2id_dict = feature2id_dict
        self.method = method
        self.matrix_form = matrix_form
        self.ROW_COL_VAL = collections.namedtuple('ROW_COL_VAL', 'row col val')

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
        """* What you can do
        - Get dictionary structure from weighted-featured scores.

        * Output
        - You can choose 'dict' or 'items' for ```outformat``` parameter.
        - If outformat='dict', you get
        >>> {label_name:{feature: score}}
        Else if outformat='items', you get
        >>> [{feature: score}]

        """
        scored_objects = self.get_feature_dictionary(
            weighted_matrix=self.scored_matrix,
            vocabulary=self.feature2id_dict,
            label_group_dict=self.label2id_dict,
            n_jobs=n_jobs)

        if sort_desc: scored_objects = \
            sorted(scored_objects, key=lambda x: x['score'], reverse=True)

        if outformat=='dict':
            out_format_structure = self.__conv_into_dict_format(scored_objects)
        elif outformat=='items':
            out_format_structure = scored_objects
        else:
            raise ValueError('outformat must be either of {dict, items}')

        return out_format_structure

    def __get_value_index(self, row_index, column_index, weight_csr_matrix, verbose=False):
        assert isinstance(row_index, (int, int32))
        assert isinstance(column_index, (int, int32))
        assert isinstance(weight_csr_matrix, csr_matrix)

        value = weight_csr_matrix[row_index, column_index]

        return value

    def make_non_zero_information(self, weight_csr_matrix: csr_matrix):
        """Construct Tuple of matrix value. Return value is array of ROW_COL_VAL namedtuple.

        :param weight_csr_matrix:
        :return:
        """
        assert isinstance(weight_csr_matrix, csr_matrix)

        row_col_index_array = weight_csr_matrix.nonzero()
        row_indexes = row_col_index_array[0]
        column_indexes = row_col_index_array[1]
        assert len(row_indexes) == len(column_indexes)

        value_index_items = [
            self.ROW_COL_VAL(
                row_indexes[i],
                column_indexes[i],
                self.__get_value_index(row_indexes[i], column_indexes[i], weight_csr_matrix)
            )
            for i
            in range(0, len(row_indexes))]

        return value_index_items

    def SUB_FUNC_feature_extraction(self,
                                    row_col_val_tuple: typing.Tuple[int, int, int],
                                    dict_index_information:Dict[str,Dict[str,str]]):
        """This function returns weighted score between label and words.

        Input csr matrix must be 'document-frequency' matrix, where records #document that word appears in document set.
        [NOTE] This is not TERM-FREQUENCY.

        For example,
        If 'iPhone' appears in 5 documents of 'IT' category document set, value must be 5.
        Even if 10 'iPhone' words in 'IT' category document set, value is still 5.
        """
        assert isinstance(row_col_val_tuple, tuple)
        assert isinstance(row_col_val_tuple, self.ROW_COL_VAL)

        return {
            'score': row_col_val_tuple.val,
            'label': self.get_label(row_col_val_tuple, dict_index_information['id2label']),
            'word': self.get_word(row_col_val_tuple, dict_index_information['id2vocab'])
        }

    def get_feature_dictionary(self,
                               weighted_matrix,
                               vocabulary,
                               label_group_dict,
                               n_jobs=1,
                               cache_backend: str = 'PersistentDict',
                               is_use_cache:bool=True):
        """* What you can do
        - Get dictionary structure from weighted-featured scores.

        * Output
        - You can choose 'dict' or 'items' for ```outformat``` parameter.
        - If outformat='dict', you get
        >>> {label_name:{feature: score}}
        Else if outformat='items', you get
        >>> [{feature: score}]

        * Args
        :param string outformat: format type of output dictionary. You can choose 'items' or 'dict'
        :param bool cut_zero: return all result or not. If cut_zero = True, the method cuts zero features.
        """
        assert isinstance(weighted_matrix, csr_matrix)
        assert isinstance(vocabulary, dict)
        assert isinstance(label_group_dict, dict)
        assert isinstance(n_jobs, int)

        logger.debug(msg='Start making scored dictionary object from scored matrix')
        logger.debug(msg='Input matrix size= {} * {}'.format(weighted_matrix.shape[0], weighted_matrix.shape[1]))

        value_index_items = self.make_non_zero_information(weighted_matrix)
        if is_use_cache:
            dict_index_information = self.initialize_cache_dict_object(cache_backend, file_name='dict_index_information')
        else:
            dict_index_information = {}

        dict_index_information['id2label'] = {value:key for key, value in label_group_dict.items()}
        dict_index_information['id2vocab'] = {value:key for key, value in vocabulary.items()}

        # TODO cython化を検討
        seq_score_objects = [
            self.SUB_FUNC_feature_extraction(
                row_col_val_tuple,
                dict_index_information) for row_col_val_tuple in value_index_items
            ]
        logger.debug(msg='Finished making scored dictionary')

        return seq_score_objects

    def get_label(self, row_col_val_tuple, label_id)->str:
        assert isinstance(row_col_val_tuple, self.ROW_COL_VAL)
        assert isinstance(label_id, dict)

        label = label_id[row_col_val_tuple.row]

        return label

    def get_word(self, row_col_val_tuple, vocabulary)->str:
        assert isinstance(row_col_val_tuple, self.ROW_COL_VAL)
        assert isinstance(vocabulary, dict)
        vocab = vocabulary[row_col_val_tuple.col]

        return vocab

    def initialize_cache_dict_object(self, cache_backend:str, file_name:str, path_cache_file=mkdtemp()):
        if cache_backend == 'PersistentDict':
            return PersistentDict(os.path.join(path_cache_file, file_name), flag='c', format='json')
        elif cache_backend == 'SqliteDict':
            return SqliteDict(os.path.join(path_cache_file, file_name), autocommit=True)
        else:
            raise Exception('No such cache_backend option named {}'.format(cache_backend))



FeatureType = TypeVar('T', str, Tuple[Any])
AvailableInputTypes = TypeVar('T', PersistentDict,
                              SqliteDict,
                              Dict[str,List[List[Union[str,Tuple[Any]]]]])