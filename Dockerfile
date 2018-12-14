#
# Facebook FAIR SentEval
# A python tool for evaluating the quality of sentence embeddings.
#
# @see https://github.com/facebookresearch/SentEval
# Based on the SentEval image from https://github.com/loretoparisi/docker (Loreto Parisi <loretoparisi@gmail.com>)
#

FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

MAINTAINER Tom Oberhauser <toberhauser@beuth-hochschule.de>

# Install base packages
RUN apt-get update && apt-get install -y \
  git \
  software-properties-common \
  python-dev \
  python-pip \
  cabextract \
  sudo \
  curl \
  unzip

# install dependencies
RUN pip install numpy scipy scikit-learn sklearn torch requests theano lasagne

WORKDIR /root/

# SentEval
# RUN git clone https://github.com/devfoo-one/SentEval.git

COPY . /

# download dataset and models

# test gloVe
# RUN python examples/bow.py

# test infersent tasks
# RUN python infersent.py

# test nvidia docker
CMD nvidia-smi -q

# defaults command
CMD ["bash"]
