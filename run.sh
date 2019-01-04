#!/usr/bin/env bash

export NV_GPU='0'
#export NV_GPU='6,7'
nvidia-docker run \
-e ENCODERURL='http://datexis-worker15:31022/v2/embed/sentences/top' \
-e ENCODERTYPE='TOKEN' \
-v $(pwd)/results:/root/results senteval-gpu