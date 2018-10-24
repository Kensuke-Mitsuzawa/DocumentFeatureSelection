from typing import Dict, List, Tuple, Union, Any, TypeVar
from scipy.sparse.csr import csr_matrix
from numpy import memmap
from sqlitedict import SqliteDict
from tempfile import mkdtemp
from DocumentFeatureSelection.init_logger import logger
from numpy import ndarray, int32, int64
import pickle
import json
import csv
import os
import shutil


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

        if isinstance(dict_matrix_index, dict):
            pass
        elif isinstance(dict_matrix_index, PersistentDict):
            dict_matrix_index.sync()
        elif isinstance(dict_matrix_index, SqliteDict):
            dict_matrix_index.sync()
        else:
            raise Exception()


class DataCsrMatrix(object):
    """* What you can do
    - You can keep information for keeping matrix object.
    """
    __slots__ = ['cache_backend', 'csr_matrix_',
                 'label2id_dict', 'vocabulary',
                 'n_docs_distribution', 'n_term_freq_distribution', 'path_working_dir']

    def __init__(self,
                 csr_matrix_: csr_matrix,
                 label2id_dict: Dict[str, int],
                 vocabulary: Dict[str, int],
                 n_docs_distribution: ndarray,
                 n_term_freq_distribution: ndarray,
                 is_use_cache: bool=False,
                 is_use_memmap: bool=False,
                 cache_backend: str='PersistentDict',
                 path_working_dir: str=None):
        """* Parameters
        -----------------
        - csr_matrix_: Matrix object which saves term frequency or document frequency
        - label2id_dict: Dict object whose key is label-name, value is row-index of the given matrix.
            >>> {'label_b': 0, 'label_c': 1, 'label_a': 2}
        -  vocabulary: Dict object whose key is feature-name, value is column-index of the given matrix.
            >>> {'label_b': 0, 'label_c': 1, 'label_a': 2}
        - n_docs_distribution: Sequence object(list,ndarray). It saves a distribution of N(docs) in each label.
        - n_term_freq_distribution: Sequence object(list,ndarray). It saves a distribution of N(all terms) in each label.
        - is_use_cache: boolean. It True; the matrix object is saved on the disk. It saves memory of your machine.
        - is_use_memmap: boolean. It True; the matrix object is saved on the disk. It saves memory of your machine.
        - cache_backend: str. {PersistentDict, SqliteDict}, backend to save this object on the disk.
        - path_working_dir: str. Path to save temporary cache objects.
        """

        self.n_docs_distribution = n_docs_distribution
        self.n_term_freq_distribution = n_term_freq_distribution
        self.cache_backend = cache_backend

        if (is_use_memmap or is_use_cache) and path_working_dir is None:
            self.path_working_dir = mkdtemp()
            logger.info("Temporary files are at {}".format(self.path_working_dir))
        else:
            self.path_working_dir = path_working_dir

        if is_use_cache:
            """You use disk-drive for keeping object.
            """
            path_vocabulary_cache_obj = os.path.join(self.path_working_dir, 'vocabulary.cache')
            path_label_2_dict_cache_obj = os.path.join(self.path_working_dir, 'label_2_dict.cache')
            self.vocabulary = self.initialize_cache_dict_object(path_vocabulary_cache_obj)
            self.vocabulary = vocabulary

            self.label2id_dict = self.initialize_cache_dict_object(path_label_2_dict_cache_obj)
            logger.info("Now saving into local file...")
            for k, v in label2id_dict.items():
                self.label2id_dict[k] = v
            if isinstance(self.label2id_dict, PersistentDict):
                self.label2id_dict.sync()

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

    def initialize_memmap_object(self, matrix_object: csr_matrix, path_memmap_object: str)->memmap:
        fp = memmap(path_memmap_object, dtype='float64', mode='w+', shape=matrix_object.shape)
        fp[:] = matrix_object.todense()[:]
        return fp

    def __str__(self):
        return """matrix-type={}, matrix-size={}, path_working_dir={}""".format(type(self.csr_matrix_),
                   self.csr_matrix_.shape,
                   self.path_working_dir)


class ROW_COL_VAL(object):
    """Data class to keep value of one item in CSR-matrix"""
    __slots__ = ('row', 'col', 'val')
    def __init__(self, row: int, col:int, val:int):
        self.row = row
        self.col = col
        self.val = val


class ScoredResultObject(object):
    """"""

    def __init__(self,
                 scored_matrix:csr_matrix,
                 label2id_dict:Union[Dict[str,Any], ndarray],
                 feature2id_dict=Union[Dict[str,Any], ndarray],
                 method:str=None,
                 matrix_form:str=None,
                 frequency_matrix:csr_matrix=None):
        """*Parameters
        ------------
        - scored_matrix: Matrix object which saves result of feature-extraction
        - label2id_dict: Dict object whose key is label-name, value is row-index of the matrix.
        - feature2id_dict: Dict object whose key is feature-name, value is column-index of the matrix.
        - method: a name of feature-extraction method.
        - matrix_form: a type of the given matrix for feature-extraction computation. {term_freq, doc_freq}
        - frequency_matrix: Matrix object(term-frequency or document-frequency). The matrix is data-source of feature-extraction computation.
        """
        self.scored_matrix = scored_matrix
        self.label2id_dict = label2id_dict
        self.feature2id_dict = feature2id_dict
        self.method = method
        self.matrix_form = matrix_form
        self.frequency_matrix = frequency_matrix
        # For keeping old version
        self.ScoreMatrix2ScoreDictionary = self.convert_score_matrix2score_record

    def __conv_into_dict_format(self, word_score_items):
        out_format_structure = {}
        for item in word_score_items:
            if item['label'] not in out_format_structure :
                out_format_structure[item['label']] = [{'feature': item['word'], 'score': item['score']}]
            else:
                out_format_structure[item['label']].append({'feature': item['word'], 'score': item['score']})
        return out_format_structure

    def convert_score_matrix2score_record(self,
                                    outformat:str='items',
                                    sort_desc:bool=True):
        """* What you can do
        - Get dictionary structure from weighted-featured scores.
        - You can choose 'dict' or 'items' for ```outformat``` parameter.

        * Output
        ---------------------
        - If outformat='dict', you get
        >>> {label_name:{feature: score}}
        Else if outformat='items', you get
        >>> [{feature: score}]

        """
        scored_objects = self.get_feature_dictionary(
            weighted_matrix=self.scored_matrix,
            vocabulary=self.feature2id_dict,
            label_group_dict=self.label2id_dict,
            frequency_matrix=self.frequency_matrix
        )

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
        assert isinstance(row_index, (int, int32, int64))
        assert isinstance(column_index, (int, int32, int64))
        assert isinstance(weight_csr_matrix, (ndarray,csr_matrix))

        value = weight_csr_matrix[row_index, column_index]

        return value

    def make_non_zero_information(self, weight_csr_matrix: csr_matrix)->List[ROW_COL_VAL]:
        """Construct Tuple of matrix value. Return value is array of ROW_COL_VAL namedtuple.

        :param weight_csr_matrix:
        :return:
        """
        assert isinstance(weight_csr_matrix, (csr_matrix, ndarray))

        row_col_index_array = weight_csr_matrix.nonzero()
        row_indexes = row_col_index_array[0]
        column_indexes = row_col_index_array[1]
        assert len(row_indexes) == len(column_indexes)

        value_index_items = [None] * len(row_indexes)  # type: List[ROW_COL_VAL]
        for i in range(0, len(row_indexes)):
            value_index_items[i] = ROW_COL_VAL(row_indexes[i],
                                               column_indexes[i],
                                               self.__get_value_index(row_indexes[i], column_indexes[i], weight_csr_matrix))
        return value_index_items

    def SUB_FUNC_feature_extraction(self,
                                    weight_row_col_val_obj: ROW_COL_VAL,
                                    dict_index_information: Dict[str, Dict[str, str]],
                                    dict_position2value: Dict[Tuple[int, int], float]=None)->Dict[str, Any]:
        """This function returns weighted score between label and words.

        Input csr matrix must be 'document-frequency' matrix, where records #document that word appears in document set.
        [NOTE] This is not TERM-FREQUENCY.

        For example,
        If 'iPhone' appears in 5 documents of 'IT' category document set, value must be 5.
        Even if 10 'iPhone' words in 'IT' category document set, value is still 5.
        """
        assert isinstance(weight_row_col_val_obj, ROW_COL_VAL)
        feature_score_record = {
            'score': weight_row_col_val_obj.val,
            'label': self.get_label(weight_row_col_val_obj, dict_index_information['id2label']),
            'feature': self.get_word(weight_row_col_val_obj, dict_index_information['id2vocab'])
        }
        if not dict_position2value is None:
            if (weight_row_col_val_obj.col,weight_row_col_val_obj.row) in dict_position2value:
                frequency = dict_position2value[tuple([weight_row_col_val_obj.col,weight_row_col_val_obj.row])]
            else:
                """When a feature-extraction method is BNS, frequency=0 is possible."""
                frequency = 0

            feature_score_record.update({"frequency": frequency})

        return feature_score_record

    def get_feature_dictionary(self,
                               weighted_matrix: csr_matrix,
                               vocabulary:Dict[str, int],
                               label_group_dict:Dict[str, int],
                               cache_backend: str = 'PersistentDict',
                               is_use_cache: bool=True,
                               frequency_matrix: csr_matrix=None)->List[Dict[str, Any]]:
        """* What you can do
        - Get dictionary structure from weighted-featured scores.
        """
        assert isinstance(weighted_matrix, csr_matrix)
        assert isinstance(vocabulary, dict)
        assert isinstance(label_group_dict, dict)

        logger.debug(msg='Start making scored dictionary object from scored matrix')
        logger.debug(msg='Input matrix size= {} * {}'.format(weighted_matrix.shape[0], weighted_matrix.shape[1]))

        weight_value_index_items = self.make_non_zero_information(weighted_matrix)
        if not frequency_matrix is None:
            frequency_value_index_items = self.make_non_zero_information(frequency_matrix)
            dict_position2value = {(t_col_row.col,t_col_row.row): t_col_row.val for t_col_row in frequency_value_index_items}
        else:
            dict_position2value = None

        if is_use_cache:
            dict_index_information = self.initialize_cache_dict_object(cache_backend, file_name='dict_index_information')
        else:
            dict_index_information = {}

        dict_index_information['id2label'] = {value:key for key, value in label_group_dict.items()}
        dict_index_information['id2vocab'] = {value:key for key, value in vocabulary.items()}
        if isinstance(dict_index_information, SqliteDict):
            dict_index_information.commit()
        elif isinstance(dict_index_information, PersistentDict):
            dict_index_information.sync()
        else:
            pass

        # TODO may be this func takes too much time. consider cython.
        seq_score_objects = [None] * len(weight_value_index_items)  # type: List[Dict[str,Any]]
        for i, weight_row_col_val_tuple in enumerate(weight_value_index_items):
            seq_score_objects[i] = self.SUB_FUNC_feature_extraction(
                weight_row_col_val_tuple,
                dict_index_information,
                dict_position2value)

        logger.debug(msg='Finished making scored dictionary')

        return seq_score_objects

    def get_label(self, row_col_val_tuple, label_id)->str:
        assert isinstance(row_col_val_tuple, ROW_COL_VAL)
        assert isinstance(label_id, dict)

        label = label_id[row_col_val_tuple.row]

        return label

    def get_word(self, row_col_val_tuple:ROW_COL_VAL, vocabulary:Dict[int,str])->Union[str,List[str],Tuple[str,...]]:
        """* what u can do
        - It gets feature name from the given matrix object.
        - A feature is json serialized, thus this method tries to de-serialize json string into python object.
            - Original feature object is possibly string(word), list of str, list of str.
        """
        assert isinstance(row_col_val_tuple, ROW_COL_VAL)
        assert isinstance(vocabulary, dict)
        vocab = vocabulary[row_col_val_tuple.col]
        try:
            feature_object = json.loads(vocab)
            if len(feature_object)==1:
                # When feature is word, the length is 1 #
                feature_object = feature_object[0]
        except:
            feature_object = vocab


        return feature_object

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