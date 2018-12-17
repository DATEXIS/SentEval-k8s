#!/usr/bin/env bash

docker run -t --runtime=nvidia \
-e ENCODERURL='http://141.64.184.124:5000/embed/sentences' \    # SET ENCODER URL
-e ENCODERTYPE='TOKEN' \                                        # SET ENCODER MODE (TOKEN, SENTENCE)
-v $(pwd)/results:/root/results \                               # MOUNT RESULTS PATH
senteval-gpu
