#!/usr/bin/env bash
BASE_DIR=`pwd`

cd ./web_api
redis-server redis.conf
nohup celery worker -A app --loglevel=info > celery.log &
python app.py
