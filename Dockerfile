#
# Facebook FAIR SentEval
# A python tool for evaluating the quality of sentence embeddings.
#
# @see https://github.com/facebookresearch/SentEval
# Based on the SentEval image from https://github.com/loretoparisi/docker (Loreto Parisi <loretoparisi@gmail.com>)
#

FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

MAINTAINER Tom Oberhauser <toberhauser@beuth-hochschule.de>

# Install base packages
RUN apt-get update && apt-get install -y git

WORKDIR /root/

COPY . /root/

ENV LANG C.UTF-8

# install dependencies
RUN pip install -r requirements.txt


# download dataset and models

# test gloVe
# RUN python examples/bow.py

# test infersent tasks
# RUN python infersent.py

# test nvidia docker
CMD nvidia-smi -q

# defaults command
# CMD ["bash"]



CMD ["python3", "EvaluateRestEncoder.py"]