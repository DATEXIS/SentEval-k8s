#!/usr/bin/env bash

./build.sh

# build image
IMAGE=registry.datexis.com/toberhauser/senteval-gpu
docker build -t $IMAGE .
version=`cat VERSION`
echo "version: $version"

docker build -t $IMAGE .
docker tag $IMAGE:latest $IMAGE:$version
docker push $IMAGE
