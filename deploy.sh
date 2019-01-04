#!/usr/bin/env bash

./build.sh
docker build -t registry.beuth-hochschule.de/toberhauser/senteval-k8s .
docker push registry.beuth-hochschule.de/toberhauser/senteval-k8s
