#!/usr/bin/env bash

./build.sh

# build image
IMAGE=registry.datexis.com/toberhauser/senteval-gpu
version=`cat VERSION`
echo "version: $version"
githash=`git rev-parse --short HEAD`

docker build -t $IMAGE -t $IMAGE:$githash -t $IMAGE:$version .
docker push $IMAGE
docker push $IMAGE:$githash
docker push $IMAGE:$version
echo "Done deploying image for build $githash"