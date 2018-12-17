#!/usr/bin/env bash

# remove SentEval data if already existing (otherwise download script fails)
find SentEval/data/downstream/ -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} \;

# Download SentEval downstream task data
cd SentEval/data/downstream
bash get_transfer_data.bash
cd ../../examples/
mkdir infersent
curl -Lo infersent/infersent.allnli.pickle https://s3.amazonaws.com/senteval/infersent/infersent.allnli.pickle && \
curl -Lo infersent/infersent.snli.pickle https://s3.amazonaws.com/senteval/infersent/infersent.snli.pickle && \
curl -Lo glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip && \
mkdir glove && \
unzip glove.840B.300d.zip -d glove && \
rm glove.840B.300d.zip && \
curl -Lo crawl-300d-2M.vec.zip https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip && \
mkdir fasttext && \
unzip crawl-300d-2M.vec.zip -d fasttext && \
rm crawl-300d-2M.vec.zip
cd ../../

# build image
IMAGE=senteval-gpu
docker build -t $IMAGE .
