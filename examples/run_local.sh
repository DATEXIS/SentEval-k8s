#!/usr/bin/env bash

export NV_GPU='0'
#export NV_GPU='6,7'
docker run --runtime=nvidia \
-e ENCODERURL='http://192.168.0.78:8080/embed/sentences' \
-e ENCODERTYPE='SENTENCE' \
-e TOKENAGGREGATION='AVG' \
-e SENTEVAL_KFOLD='10' \
-e SENTEVAL_CLASSIFIER_NHID='0' \
-e SENTEVAL_CLASSIFIER_OPTIM='adam' \
-e SENTEVAL_CLASSIFIER_BATCHSIZE='64' \
-e SENTEVAL_CLASSIFIER_TENACITY='5' \
-e SENTEVAL_CLASSIFIER_EPOCHSIZE='4' \
-e SENTEVAL_CLASSIFIER_DROPOUT='0.0' \
-e LOGLEVEL='DEBUG' \
-e TIMEOUT='60.0' \
-v $(pwd)/results:/root/results senteval-gpu:latest
