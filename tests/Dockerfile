FROM frolvlad/alpine-glibc:alpine-3.6
MAINTAINER kensuke-mi <kensuke.mit@gmail.com>

# apk update
RUN apk update
# general
RUN apk --no-cache add vim \
wget \
lsof \
curl \
bash \
swig \
gcc \
build-base \
make \
python-dev \
py-pip \
jpeg-dev \
zlib-dev \
git \
linux-headers
ENV LIBRARY_PATH=/lib:/usr/lib

ENV PATH=/opt/conda/bin:$PATH \
    LANG=C.UTF-8 \
    MINICONDA=Miniconda3-latest-Linux-x86_64.sh

# Python
RUN apk add --no-cache bash wget && \
    wget -q --no-check-certificate https://repo.continuum.io/miniconda/$MINICONDA && \
    bash /Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    ln -s /opt/conda/bin/* /usr/local/bin/ && \
    rm -rf /root/.[acpw]* /$MINICONDA /opt/conda/pkgs/*

RUN conda config --add channels conda-forge --system
RUN conda install Cython \
scikit-learn \
scipy \
numpy

RUN pip install more_itertools joblib nltk pypandoc sqlitedict nose
RUN conda create -y -n p36 python=3.6
RUN conda create -y -n p37 python=3.7

CMD ["/bin/bash"]