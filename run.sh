#!/usr/bin/env bash

export NV_GPU='0'
#export NV_GPU='6,7'
docker run --runtime=nvidia \
-e ENCODERURL='http://datexis-worker15:31022/v2/embed/sentences/top' \
-e ENCODERTYPE='TOKEN' \
-e TOKENAGGREGATION='AVG' \
-e SENTEVAL_KFOLD='10' \
-e SENTEVAL_CLASSIFIER_NHID='0' \
-e SENTEVAL_CLASSIFIER_OPTIM='adam' \
-e SENTEVAL_CLASSIFIER_BATCHSIZE='64' \
-e SENTEVAL_CLASSIFIER_TENACITY='5' \
-e SENTEVAL_CLASSIFIER_EPOCHSIZE='4' \
-e SENTEVAL_CLASSIFIER_DROPOUT='0.0' \
-v $(pwd)/results:/root/results senteval-gpu
