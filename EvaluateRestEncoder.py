from __future__ import absolute_import, division, unicode_literals

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import logging

# Set PATHs
import requests

from SIF.src import SIF_embedding
from senteval import utils

""" ---------- CONFIG ------------ """
PATH_TO_SENTEVAL = 'SentEval'
PATH_TO_DATA = 'SentEval/data'
REQUEST_HEADERS = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
MAX_CONNECTION_RETRIES = 30

logger = logging.getLogger(__name__)

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def arora(word_vectors, term_frequencies, a=.001):
    """
    Aggregates a bag of word vectors to a single vector using
    “A Simple but Tough-to-Beat Baseline for Sentence Embeddings.” - Arora et al. - 2017
    Since word_vectors contain all vectors in a 2d array, the sentence split information must be performed by
    the term_frequencies array.
    :param a:
    :param word_vectors: list[n, dim]: ordered word vectors word n (for all sentences)
    :param term_frequencies: list[i, n]: ordered term frequencies for sentence i and token n within that sentence
    :return: [i, :] sentence embeddings for sentence i
    """

    if type(word_vectors) is not list:
        raise TypeError('word_vectors must be a list of shape [n, dim]')

    if type(term_frequencies) is not list:
        raise TypeError('term_frequencies must be a list of shape [i, n]')

    num_sentences = len(term_frequencies)
    longest_sentence_count = max([len(sentence) for sentence in term_frequencies])
    term_weights = np.zeros((num_sentences, longest_sentence_count))

    indices = np.zeros((num_sentences, longest_sentence_count), dtype=np.int)

    index = 0
    for s_i, sentence_term_frequencies in enumerate(term_frequencies):
        for t_i, tf in enumerate(sentence_term_frequencies):
            term_weights[s_i, t_i] = a / (a + tf)
            indices[s_i, t_i] = index
            index += 1

    params = utils.dotdict({'rmpc': 1})  # remove 1st principal component

    # Arora expects ONE LARGE WORD VECTOR ARRAY and the INDICES MUST BE OUT OF THIS ARRAY FOR ALL SENTENCES!
    word_vectors = np.asarray(word_vectors, dtype=np.float64)
    embeddings = SIF_embedding.SIF_embedding(word_vectors, indices, term_weights, params)
    return embeddings


def tf_dictionary(samples):
    """
    Returns a dictionary with the term frequencies for all terms of a set of samples
    :param samples: Array of arrays of tokenized words
    :return: Dictionary with term frequencies
    """
    term_counter = defaultdict(int)
    term_frequencies = dict()
    total_terms = 0
    for sample in samples:
        for token in sample:
            term_counter[token] += 1
            total_terms += 1
    for term, count in term_counter.items():
        term_frequencies[term] = count / total_terms
    return term_frequencies


# SentEval method prepare
def prepare(params, samples):
    """
    SentEval code that gets called by SentEval before a task.
    It is used to determine the term frequencies.
    :return: None
    """
    params.tf_dictionary = tf_dictionary(samples)
    return


def get_vectors_from_encoder(batch):
    """
    Queries the REST encoder and returns embedding vectors in the correct order
    :param batch: Array of arrays of tokens
    :return: Embedding vectors
    """
    should_connect = True
    retries = 0
    while should_connect:
        retries += 1
        if retries > MAX_CONNECTION_RETRIES:
            logger.error("Giving up on connecting to encoder {}".format(ENCODER_URL))
            exit(2)
        try:
            r = requests.post(ENCODER_URL, headers=REQUEST_HEADERS, data=json.dumps(batch))
        except requests.exceptions.ConnectionError:
            time.sleep(5)
            logger.error("Error while connecting to encoder {}, attempt {}/{}".format(ENCODER_URL, retries,
                                                                                      MAX_CONNECTION_RETRIES))
            should_connect = True
        else:
            should_connect = False
    vectors = r.json()
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors


# SentEval method batcher
def batcher(params, batch):
    """
    SentEval code that gets called by SentEval during a task with a current batch of text.
    """
    batch = [sent if sent != [] else ['.'] for sent in batch]
    batch_term_frequencies = []
    for sent in batch:
        sent_tf = []
        for token in sent:
            sent_tf.append(params.tf_dictionary[token])
        batch_term_frequencies.append(sent_tf)
    batch_term_frequencies = np.asarray(batch_term_frequencies, dtype=np.float32)
    batch_word_vectors = get_vectors_from_encoder(batch)
    batch_word_vectors = batch_word_vectors[:, 1]  # use second layer elmo layer  # TODO THIS IS ELMO SPECIFIC, REMOVE!

    sentence_embeddings = arora(batch_word_vectors, batch_term_frequencies, a=0.001)

    return sentence_embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    try:
        ENCODER_URL = os.environ['ENCODERURL']
    except KeyError:
        logger.error("Encoder URL must be specified using the ENCODERURL environment variable!")
        exit(1)
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    outfile = open("RESTEncoderResults.json", "w")  # TODO BETTER PERSISTENCE
    outfile.write(json.dumps(results))
    outfile.close()
    print(results)
