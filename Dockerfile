# Anaconda3 python distributionをベースContainerに利用
FROM continuumio/anaconda3
MAINTAINER kensuke-mi <kensuke.mit@gmail.com>

ENV REDIS_VERSION 3.2.8
ENV REDIS_HOME /opt/redis
RUN mkdir -p /opt
RUN mkdir -p ${REDIS_HOME}

## use apt-get for install
RUN apt-get update
RUN apt-get install -y software-properties-common --fix-missing
RUN apt-get update  # update again after adding repository
# install gcc, make
RUN apt-get install -y gcc --fix-missing
RUN apt-get install -y g++ --fix-missing
RUN apt-get install -y swig2.0 --fix-missing
RUN apt-get install -y make --fix-missing
# install other tools
RUN apt-get install -y vim wget lsof curl sqlite3 pandoc

## Install redis server
RUN wget -q http://download.redis.io/releases/redis-${REDIS_VERSION}.tar.gz && \
    tar -zxvf redis-${REDIS_VERSION}.tar.gz && \
    mv redis-${REDIS_VERSION} redis-src && \
    cd redis-src && \
    make

## Install packages for python with conda
RUN conda install -y numpy scipy scikit-learn cython psycopg2
RUN pip install --user https://github.com/rogerbinns/apsw/releases/download/3.17.0-r1/apsw-3.17.0-r1.zip \
--global-option=fetch --global-option=--version --global-option=3.17.0 --global-option=--all \
--global-option=build --global-option=--enable-all-extensions
RUN pip install celery

RUN mkdir /codes
ADD . /codes/DocumentFeatureSelection
RUN cd /codes/DocumentFeatureSelection && python setup.py install

RUN mkdir /var/log/redis/
RUN mkdir /var/run/redis

EXPOSE 6379
EXPOSE 5000
WORKDIR /codes/DocumentFeatureSelection
CMD ["/bin/bash", "start_web_service.sh"]