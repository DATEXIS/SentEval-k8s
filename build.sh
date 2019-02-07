#!/usr/bin/env bash

# remove SentEval data if already existing (otherwise download script fails)
find SentEval/data/downstream/ -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} \;

# Store current git hash to identify build
git rev-parse HEAD > commit-hash.txt

# Download SentEval downstream task data
cd SentEval/data/downstream
bash get_transfer_data.bash
cd ../../../

# build image
IMAGE=senteval-gpu
docker build -t $IMAGE .
version=`cat VERSION`
echo "version: $version"
docker tag $IMAGE:latest $IMAGE:$version