#!/usr/bin/env bash

# docker run -v $(pwd)/eval:/root/SentEval/eval -it --runtime=nvidia senteval-gpu bash
docker run -t --runtime=nvidia -e ENCODERURL='http://141.64.184.124:5000/embed/sentences' \
-e ENCODERTYPE='TOKEN' \
senteval-gpu
