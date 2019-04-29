#!/usr/bin/env bash

export NV_GPU='0'
#export NV_GPU='6,7'
docker run --runtime=nvidia \
-e ENCODERURL='http://141.64.184.124:2342/embed/sentences' \
-e ENCODERTYPE='SENTENCE' \
-e TOKENAGGREGATION='NONE' \
-e SENTEVAL_KFOLD='10' \
-e SENTEVAL_CLASSIFIER_NHID='0' \
-e SENTEVAL_CLASSIFIER_OPTIM='adam' \
-e SENTEVAL_CLASSIFIER_BATCHSIZE='64' \
-e SENTEVAL_CLASSIFIER_TENACITY='5' \
-e SENTEVAL_CLASSIFIER_EPOCHSIZE='4' \
-e SENTEVAL_CLASSIFIER_DROPOUT='0.0' \
-v $(pwd)/results:/root/results senteval-gpu
