# -*- coding: UTF-8 -*-

from __future__ import absolute_import, division, unicode_literals

import datetime
import json
import os
import re
import sys
import time
from collections import defaultdict
from enum import Enum

import numpy as np
import logging

# Set PATHs
import requests
import urllib3

from SIF.src import SIF_embedding

""" ---------- CONFIG ------------ """
PATH_TO_SENTEVAL = 'SentEval/'
PATH_TO_DATA = 'SentEval/data'
REQUEST_HEADERS = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
PATH_TO_RESULTS = 'results/'
MAX_CONNECTION_RETRIES = 300

logger = logging.getLogger(__name__)

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


class EncoderType(Enum):
    TOKEN = 1
    SENTENCE = 2


class TokenAggregationMode(Enum):
    AVG = 1
    ARORA = 2
    NOAGG = 3


def arora(word_vectors, term_frequencies, a=.001):
    """
    Aggregates a bag of word vectors to a single vector using
    “A Simple but Tough-to-Beat Baseline for Sentence Embeddings.” - Arora et al. - 2017
    Since word_vectors contain all vectors in a 2d array, the sentence split information must be performed by
    the term_frequencies array.
    :param a: Smoothing parameter a (default is 0.001)
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
    # Arora expects ONE LARGE WORD VECTOR ARRAY and the INDICES MUST BE OUT OF THIS ARRAY FOR ALL SENTENCES!
    indices = np.zeros((num_sentences, longest_sentence_count), dtype=np.int)
    index = 0
    for sentence_index, sentence_term_frequencies in enumerate(term_frequencies):
        for token_index, token_frequency in enumerate(sentence_term_frequencies):
            term_weights[sentence_index, token_index] = a / (a + token_frequency)
            indices[sentence_index, token_index] = index
            index += 1
    params = senteval.utils.dotdict({'rmpc': 1})  # remove 1st principal component
    word_vectors = np.asarray(word_vectors, dtype=np.float64)
    embeddings = SIF_embedding.SIF_embedding(word_vectors, indices, term_weights, params)
    return embeddings


def average(word_vectors, term_frequencies):
    """
        Aggregates a bag of word vectors to a single vector through averaging
        Since word_vectors contain all vectors in a 2d array, the sentence split information must be performed by
        the term_frequencies array.
        :param word_vectors: list[n, dim]: ordered word vectors word n (for all sentences)
        :param term_frequencies: list[i, n]: ordered term frequencies for sentence i and token n within that sentence
        :return: [i, :] sentence embeddings for sentence i
        """

    if type(word_vectors) is not list:
        raise TypeError('word_vectors must be a list of shape [n, dim]')
    if type(term_frequencies) is not list:
        raise TypeError('term_frequencies must be a list of shape [i, n]')
    embeddings = []
    token_index = 0
    for sentence_term_frequencies in term_frequencies:
        sentence_word_vectors = []
        for _ in sentence_term_frequencies:
            sentence_word_vectors.append(word_vectors[token_index])
            token_index += 1
        embeddings.append(np.average(sentence_word_vectors, axis=0))
    return np.asarray(embeddings)


def aggregate_token_vectors_to_sentence_vector(word_vectors, term_frequencies):
    if config['TOKEN_AGGREGATION_MODE'] == TokenAggregationMode.ARORA:
        return arora(word_vectors, term_frequencies, a=0.001)
    if config['TOKEN_AGGREGATION_MODE'] == TokenAggregationMode.AVG:
        return average(word_vectors, term_frequencies)


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
    vectors = []
    while should_connect:
        retries += 1
        if retries > MAX_CONNECTION_RETRIES:
            logger.error("Giving up on connecting to encoder {}".format(config['ENCODER_URL']))
            exit(2)
        try:
            r = requests.post(config['ENCODER_URL'], headers=REQUEST_HEADERS, data=json.dumps(batch), timeout=config['TIMEOUT'])
            if r.status_code != 200:
                time.sleep(5)
                logger.error("Got status code {} from encoder {}, attempt {}/{}".format(r.status_code, config['ENCODER_URL'], retries,
                                                                                      MAX_CONNECTION_RETRIES))
                continue
            vectors = r.json()
        except (requests.exceptions.ConnectionError, urllib3.exceptions.ProtocolError,
                requests.exceptions.ChunkedEncodingError, urllib3.exceptions.ReadTimeoutError) as e:
            time.sleep(5)
            logger.error("Error while connecting to encoder {}, attempt {}/{}".format(config['ENCODER_URL'], retries,
                                                                                      MAX_CONNECTION_RETRIES))
        except ValueError:
            time.sleep(5)
            logger.error("Got invalid JSON from encoder {}, attempt {}/{}".format(config['ENCODER_URL'], retries,
                                                                                      MAX_CONNECTION_RETRIES))
        else:
            should_connect = False
    if len(batch) == 1:  # TODO https://github.com/SchmaR/ELMo-Rest/issues/8
        vectors = [vectors]
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
    batch_vectors = get_vectors_from_encoder(batch)
    batch_word_vectors_flattened = []
    for sentence_word_vectors in batch_vectors:
        for word_vector in sentence_word_vectors:
            batch_word_vectors_flattened.append(np.asarray(word_vector, dtype=np.float64))
    if config['ENCODER_TYPE'] == EncoderType.TOKEN:
        return aggregate_token_vectors_to_sentence_vector(batch_word_vectors_flattened, batch_term_frequencies)
    if config['ENCODER_TYPE'] == EncoderType.SENTENCE:
        return np.asarray(batch_vectors, dtype=np.float64)


def generate_filename(encoder_url, encoder_mode, aggregation_mode):
    fn = re.sub(r'http[s]*://', ' ', encoder_url)
    fn = fn.replace(':', '-')
    fn = fn.replace('/', '-')
    fn = ''.join([c for c in fn if re.match(r'\w', c) or c == '-' or c == '_'])
    if encoder_mode == EncoderType.TOKEN:
        fn = datetime.datetime.now().strftime(
            '%Y%m%d_%H%M%S') + '_' + encoder_mode.name + '_' + aggregation_mode.name + '_' + fn + '.json'
    if encoder_mode == EncoderType.SENTENCE:
        fn = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + encoder_mode.name + '_' + fn + '.json'
    fn = PATH_TO_RESULTS + fn
    return fn


def serialization_helper(results):
    """
    Prepares a result object for serialization
    :param results: Results object as returned by SentEval
    :return: Serializable results object
    """
    if type(results) == dict:
        return all_ndarrays_in_dict_2_lists(results)


def all_ndarrays_in_dict_2_lists(d):
    """
    Recursively iterates over a dict and transforms all ndarrays to lists
    :param d: dict
    :return: dict
    """
    for k, v in d.items():
        if type(v) == np.ndarray:
            d[k] = v.tolist()
        if type(v) == dict:
            d[k] = all_ndarrays_in_dict_2_lists(v)
    return d


def add_evaluation_parameter_infos_to_result(senteval_results, senteval_params, general_params):
    senteval_results['evaluation-parameters'] = {}
    with open('commit-hash.txt', 'r') as f:
        results['evaluation-parameters']['build'] = f.read().split()[0]
    senteval_results['evaluation-parameters']['datetime-finished'] = str(datetime.datetime.now())
    senteval_results['evaluation-parameters']['encoder-url'] = general_params['ENCODER_URL']
    senteval_results['evaluation-parameters']['encoder-type'] = general_params['ENCODER_TYPE'].name
    senteval_results['evaluation-parameters']['token-aggregation-type'] = general_params['TOKEN_AGGREGATION_MODE'].name
    senteval_results['evaluation-parameters']['senteval'] = senteval_params
    return senteval_results


if __name__ == "__main__":
    config = {}

    try:
        config['ENCODER_URL'] = os.environ['ENCODERURL']
    except KeyError:
        logger.error("Encoder URL must be specified using the ENCODERURL environment variable!")
        exit(1)

    try:
        if os.environ['ENCODERTYPE'] == 'TOKEN':
            config['ENCODER_TYPE'] = EncoderType.TOKEN
        if os.environ['ENCODERTYPE'] == 'SENTENCE':
            config['ENCODER_TYPE'] = EncoderType.SENTENCE
            config['TOKEN_AGGREGATION_MODE'] = TokenAggregationMode.NOAGG
    except KeyError:
        logger.error("Encoder type (TOKEN or SENTENCE) must be specified using the ENCODERTYPE environment variable!")
        exit(1)

    if 'ENCODER_TYPE' not in config.keys():
        raise RuntimeError('ENCODERTYPE has not been specified or is invalid!')
    if config['ENCODER_TYPE'] == EncoderType.TOKEN:
        try:
            config['TOKEN_AGGREGATION_MODE'] = {
                'AVG': TokenAggregationMode.AVG,
                'ARORA': TokenAggregationMode.ARORA}[os.environ['TOKENAGGREGATION']]
        except KeyError:
            logger.error("Aggregation mode (TOKENAGGREGATION) must be specified for token encoders (AVG or ARORA)!")
            exit(1)
    if not os.path.isdir(PATH_TO_RESULTS):
        raise RuntimeError('Result path {} not found!'.format(PATH_TO_RESULTS))

    try:
        # Set params for SentEval
        params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': int(os.environ['SENTEVAL_KFOLD']),
                           'classifier': {
                               'nhid': int(os.environ['SENTEVAL_CLASSIFIER_NHID']),
                               'optim': os.environ['SENTEVAL_CLASSIFIER_OPTIM'],
                               'batch_size': int(os.environ['SENTEVAL_CLASSIFIER_BATCHSIZE']),
                               'tenacity': int(os.environ['SENTEVAL_CLASSIFIER_TENACITY']),
                               'epoch_size': int(os.environ['SENTEVAL_CLASSIFIER_EPOCHSIZE']),
                               'dropout': float(os.environ['SENTEVAL_CLASSIFIER_DROPOUT'])
                           }}
    except KeyError as e:
        logger.error('Invalid parameter config for {}!'.format(e.args[0]))
        exit(1)

    config['LOGLEVEL'] = os.getenv('LOGLEVEL', 'ERROR')
    logging.basicConfig(level=config['LOGLEVEL'])

    config['TASKS'] = os.getenv('TASKS', 'STS12, STS13, STS14, STS15, STS16, MR, CR, MPQA, SUBJ, SST2, SST5, TREC,'
                                         'MRPC, SICKEntailment, SICKRelatedness, STSBenchmark, Length, WordContent, '
                                         'Depth,TopConstituents, BigramShift, Tense, SubjNumber, ObjNumber, '
                                         'OddManOut, CoordinationInversion, PubMedSection, WikiSection')
    config['TASKS'] = [x.strip() for x in config['TASKS'].split(',')]
    config['TIMEOUT'] = os.getenv('TIMEOUT', 30)

    logger.info('Starting SentEval for {}'.format(config['ENCODER_URL']))
    logger.info('Tasks: {}'.format(', '.join(config['TASKS'])))
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = config['TASKS']
    results = se.eval(transfer_tasks)
    results = add_evaluation_parameter_infos_to_result(results, params_senteval, config)
    filename = generate_filename(config['ENCODER_URL'], config['ENCODER_TYPE'], config['TOKEN_AGGREGATION_MODE'])
    outfile = open(filename, "w")
    outfile.write(json.dumps(serialization_helper(results)))
    outfile.close()
    logger.info('Evaluation of {} finished.'.format(config['ENCODER_URL']))
