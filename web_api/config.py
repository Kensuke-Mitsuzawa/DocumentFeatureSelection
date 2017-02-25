class BaseConfig(object):
    PATH_WORKING_DIR = None
    FILENAME_BACKEND_SQLITE3 = None


class DevelopmentConfig(BaseConfig):
    PATH_WORKING_DIR = './tests'
    FILENAME_BACKEND_SQLITE3 = 'backend.sqlite3'


class TestingConfig(BaseConfig):
    PATH_WORKING_DIR = None
    FILENAME_BACKEND_SQLITE3 = None