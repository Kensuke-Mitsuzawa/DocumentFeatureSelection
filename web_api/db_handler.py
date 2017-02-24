#! -*- coding: utf-8 -*-

"""This is sqlite3 db-hander to save computation result"""

import os
import sqlite3
import logging
import traceback
import json
from typing import List, Dict, Any
logger = logging.getLogger(__file__)


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
            self.db_connection = sqlite3.connect(database=self.path_sqlite_file)
            self.db_connection.text_factory = str
            self.create_db()
        else:
            self.db_connection = sqlite3.connect(database=self.path_sqlite_file)
            self.db_connection.text_factory = str

    def __del__(self):
        if self.is_close_connection_end and hasattr(self, 'db_connection'):
            self.db_connection.close()

    def create_db(self):
        cur = self.db_connection.cursor()
        sql = """create table if not exists {table_name} (
        record_id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT,
        result_json TEXT
        created_at DATETIME,
        updated_at DATETIME)"""
        cur.execute(sql.format(table_name=self.table_name_text))
        self.db_connection.commit()

    def insert_record(self, job_id:int, result_obj:List[Dict[str,Any]]):
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
                self.db_connection.commit()
                cur.close()
            except:
                logger.error(traceback.format_exc())
                self.db_connection.rollback()
                return False


    def get_one_record(self, record_id):
        sql_ = """SELECT result_json FROM {} WHERE job_id = ?""".format(self.table_name_text)

        cur = self.db_connection.cursor()
        cur.execute(sql_, (record_id, ))
        fetched_record = cur.fetchone()
        return json.loads(fetched_record[0])