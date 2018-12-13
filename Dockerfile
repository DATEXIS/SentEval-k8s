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
RUN git clone https://github.com/devfoo-one/SentEval.git

# download dataset and models
RUN \
    cd SentEval/data/downstream && \
    bash get_transfer_data.bash && \
    cd ../../examples/ && \
    mkdir infersent && \
    curl -Lo infersent/infersent.allnli.pickle http://141.64.184.124:8080/infersent.allnli.pickle && \
    curl -Lo infersent/infersent.snli.pickle http://141.64.184.124:8080/infersent.snli.pickle && \
    curl -Lo glove.840B.300d.zip http://141.64.184.124:8080/glove.840B.300d.zip && \
    mkdir glove && \
    unzip glove.840B.300d.zip -d glove && \
    rm glove.840B.300d.zip && \
    curl -Lo crawl-300d-2M.vec.zip http://141.64.184.124:8080/crawl-300d-2M.vec.zip && \
    mkdir fasttext && \
    unzip crawl-300d-2M.vec.zip -d fasttext && \
    rm crawl-300d-2M.vec.zip

# RUN \
#    cd SentEval/data/downstream && \
#    bash get_transfer_data.bash && \
#    cd ../../examples/ && \
#    mkdir infersent && \
#    curl -Lo infersent/infersent.allnli.pickle https://s3.amazonaws.com/senteval/infersent/infersent.allnli.pickle && \
#    curl -Lo infersent/infersent.snli.pickle https://s3.amazonaws.com/senteval/infersent/infersent.snli.pickle && \
#    curl -Lo glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip && \
#    mkdir glove && \
#    unzip glove.840B.300d.zip -d glove && \
#    rm glove.840B.300d.zip && \
#    curl -Lo crawl-300d-2M.vec.zip https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip && \
#    mkdir fasttext && \
#    unzip crawl-300d-2M.vec.zip -d fasttext && \
#    rm crawl-300d-2M.vec.zip


# test gloVe
# RUN python examples/bow.py

# test infersent tasks
# RUN python infersent.py

# test nvidia docker
CMD nvidia-smi -q

# defaults command
CMD ["bash"]
