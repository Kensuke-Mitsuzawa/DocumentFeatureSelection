#! - coding: utf-8 -*-
## flask package
from flask import Flask, render_template, request, redirect, url_for, jsonify
## typing
from typing import List, Dict, Any, Union
## DocumentFeature selection
from DocumentFeatureSelection import interface
from celery import Celery
## to generate job-queue-id
import uuid
## to save job result
# else
import pkg_resources
import os
import traceback
import json
import apsw
from datetime import datetime


class Sqlite3Handler(object):
    def __init__(self,
                 path_sqlite_file,
                 table_name_text="text",
                 is_close_connection_end=True):
        """* Args
        - is_close_connection_end
            - If True, it deleted DB when a process is done. False; don't.
        """
        # type: (str,str,bool)->None
        self.table_name_text = table_name_text
        self.is_close_connection_end = is_close_connection_end
        self.path_sqlite_file = path_sqlite_file
        if not os.path.exists(self.path_sqlite_file):
            #self.db_connection = sqlite3.connect(database=self.path_sqlite_file)
            #self.db_connection.text_factory = str
            self.db_connection = apsw.Connection(filename=self.path_sqlite_file)
            self.create_db()
            self.db_connection = apsw.Connection(filename=self.path_sqlite_file)
        else:
            #self.db_connection = sqlite3.connect(database=self.path_sqlite_file)
            #self.db_connection.text_factory = str
            self.db_connection = apsw.Connection(filename=self.path_sqlite_file)

    def __del__(self):
        if self.is_close_connection_end and hasattr(self, 'db_connection'):
            self.db_connection.close()

    def create_db(self):
        #cur = self.db_connection.cursor()
        cur = self.db_connection.cursor()
        sql = """create table if not exists {table_name} (
        record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT,
        result_json TEXT
        created_at DATETIME,
        updated_at DATETIME)"""
        cur.execute(sql.format(table_name=self.table_name_text))
        #self.db_connection.commit()
        self.db_connection.execute('commit')

    def insert_record(self, job_id:str, result_obj:List[Dict[str,Any]]):
        """* What you can do
        - You initialize record for input
        """
        sql_check = "SELECT count(job_id) FROM {} WHERE job_id = ?".format(self.table_name_text)
        cur = self.db_connection.cursor()
        cur.execute(sql_check, (job_id,))
        if cur.fetchone()[0] >= 1:
            cur.close()
            return False
        else:
            result_json = json.dumps(result_obj, ensure_ascii=False)
            sql_insert = """INSERT INTO {}(job_id, result_json)
            values (?, ?)""".format(self.table_name_text)
            cur = self.db_connection.cursor()
            try:
                cur.execute(sql_insert, (job_id,
                                         result_json))
                self.db_connection.execute('commit')
                cur.close()
            except:
                #logger.error(traceback.format_exc())
                # self.db_connection.rollback()
                print(traceback.format_exc())
                return False

    def get_one_record(self, record_id)->Union[None, Dict[str,Any]]:
        sql_ = """SELECT result_json FROM {} WHERE job_id = ?""".format(self.table_name_text)

        cur = self.db_connection.cursor()
        cur.execute(sql_, (record_id, ))
        fetched_record = cur.fetchone()
        if fetched_record is None:
            return False
        else:
            return json.loads(fetched_record[0])


def make_celery(app):
    '''
    celery = Celery(app.import_name,
                    backend=app.config['CELERY_RESULT_BACKEND'],
                    broker=app.config['CELERY_BROKER_URL'])'''
    celery = Celery('app',
                    backend=app.config['CELERY_RESULT_BACKEND'],
                    broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    return celery


flask_app = Flask(__name__)
############ You change following config depending on your app ############
flask_app.config.from_object('config.DevelopmentConfig')
flask_app.config.from_object('celeryconfig')
flask_app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379')
celery = make_celery(flask_app)

### Connect with backend DB
path_backend_db = os.path.join(flask_app.config['PATH_WORKING_DIR'], flask_app.config['FILENAME_BACKEND_SQLITE3'])
backend_database_handler = Sqlite3Handler(path_sqlite_file=path_backend_db)


@celery.task(bind=True)
def run_interface(self,
                  job_id:str,
                  input_dict:Dict[str,Any],
                  method:str,
                  use_cython:bool=True,
                  is_use_cache:bool=True,
                  is_use_memmap:bool=True):
    """* What you can do
    - Background task that runs a long function with progress reports.
    - It saves result into background DB

    """
    ###
    started_at = datetime.now()
    self.update_state(state='PROGRESS', meta={'method': method,
                                              'started_at': started_at.strftime('%Y-%m-%d %H:%M:%S')})

    scored_result_obj = interface.run_feature_selection(
        input_dict=input_dict,
        method=method,
        use_cython=use_cython,
        is_use_cache=is_use_cache,
        is_use_memmap=is_use_memmap,
        path_working_dir=flask_app.config['PATH_WORKING_DIR']
    )

    backend_database_handler.insert_record(job_id=job_id,
                                           result_obj=scored_result_obj.ScoreMatrix2ScoreDictionary())

    return {'job_id': job_id,
            'status': 'completed',
            'method': method,
            'started_at': started_at.strftime('%Y-%m-%d %H:%M:%S')}


@flask_app.route('/')
def index():
    package_version = pkg_resources.get_distribution("DocumentFeatureSelection").version
    # index.html をレンダリングする
    return render_template('index.html', version_info=package_version)


@flask_app.route('/status/<task_id>', methods=['Get'])
def taskstatus(task_id):
    task = run_interface.AsyncResult(task_id)
    if task.state == 'PENDING':
        # job did not start yet
        response = {
            'state': task.state,
            'started_at': task.info.get('started_at'),
            'method': task.method,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'method': task.info.get('method'),
            'started_at': task.info.get('started_at'),
            'status': task.info.get('status', '')
        }
        if 'result' in task.info:
            response['result'] = task.info['result']
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'method': task.info.get('method'),
            'started_at': task.info.get('started_at'),
            'status': str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@flask_app.route('/run_job_api', methods=['POST', 'Get'])
def run_job_api():
    """* What you can do
    - You start document feature selection with your data
    - This process takes looooong time, so this API saves does not return result itself.
    - Instead, this API saves result into DB. And the API returns key of record.

    * Format
    - body json must have following fields
        - method
        - input_data
    >>> {"method": "soa", "input_data": {"label1": [["I", "aa", "aa", "aa", "aa", "aa"],["bb", "aa", "aa", "aa", "aa", "aa"], ["I", "aa", "hero", "some", "ok", "aa"]], "label2": [ ["bb", "bb", "bb"], ["bb", "bb", "bb"], ["hero", "ok", "bb"], ["hero", "cc", "bb"]]}}
    """
    if request.method == 'GET':
        return render_template('index.html')

    body_object = request.get_json()
    try:
        method = body_object['method']
        input_data = body_object['input_data']

        started_at = datetime.now()
        date = started_at.strftime('%Y-%m-%d')
        job_id = '{}-{}'.format(date, str(uuid.uuid4()))

        task = run_interface.apply_async(args=[job_id, input_data, method])
        response_body = {'message': 'your job is started',
                         'job_id': job_id,
                         'task_id': task.id}
        return jsonify(response_body), 202, {'Location': url_for('taskstatus', task_id=task.id)}
    except:
        response_body = {'message': 'Internal server error.',
                         'traceback': traceback.format_exc()}
        return jsonify(response_body), 500


@flask_app.route('/get_result_api', methods=['POST', 'GET'])
def get_result_api():
    """* What you can do
    - You can get processed result

    * Format
    - body json must have following fields
        - job_id
    >>> {"job_id" : "2017-02-25-3599a066-e590-4a0c-8b4d-4804f608a11d"}
    """
    try:
        body_object = request.get_json()
        job_id = body_object['job_id']
        result_object = backend_database_handler.get_one_record(record_id=job_id)
        response_body = {
            'job_id': job_id,
            'result': result_object
        }
        return jsonify(response_body), 200
    except:
        response_body = {'message': 'Internal server error.',
                         'traceback': traceback.format_exc()}
        return jsonify(response_body), 500


if __name__ == '__main__':
    flask_app.debug = True # デバッグモード有効化
    flask_app.run(host='0.0.0.0') # どこからでもアクセス可能に