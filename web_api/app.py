#! - coding: utf-8 -*-
## flask package
from flask import Flask, render_template, request, redirect, url_for
## DocumentFeature selection
from DocumentFeatureSelection import interface
from celery import Celery
## to generate job-queue-id
import uuid
## to save job result
from sqlitedict import SqliteDict


def make_celery(app):
    celery = Celery(app.import_name, backend=app.config['CELERY_RESULT_BACKEND'],
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


flask_app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)
celery = make_celery(flask_app)


@flask_app.route('/')
def index():
    title = "Welcome to DocumentFeatureSelction API"
    # index.html をレンダリングする
    #return render_template('index.html', message=message, title=title)
    return None


@flask_app.route('/post', methods=['GET', 'POST'])
def post():
    """* What you can do
    - You start document feature selection with your data
    - This process takes looooong time, so this API saves does not return result itself.
    - Instead, this API saves result into DB. And the API returns key of record.
    """
    pass