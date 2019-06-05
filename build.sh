#!/usr/bin/env bash

# remove SentEval data if already existing (otherwise download script fails)
find SentEval/data/downstream/ -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} \;

# Store current git hash to identify build
git rev-parse --short HEAD > commit-hash.txt

# Download SentEval downstream task data
cd SentEval/data/downstream
bash get_transfer_data.bash
bash get_pubmed-wikisection_data.bash
cd ../../../

# build image
IMAGE=senteval-gpu
build=`cat commit-hash.txt`
version=`cat VERSION`
docker build -t $IMAGE -t $IMAGE:latest -t $IMAGE:$version -t $IMAGE:$build .
echo "version: $version"
echo "build: $build"