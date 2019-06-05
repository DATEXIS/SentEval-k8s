# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

'''
PubMed-Section Topic-classification
'''

from __future__ import absolute_import, division, unicode_literals

import json
import logging
import os

import numpy as np

from senteval.tools.validation import KFoldClassifier


def getAnnotatedSentencesFromArticle(article, tgt2idx, samples, labels):
    for sentence in article['Sentences']:
        label = sentence['Label']
        tokens = []
        for t in sentence['Tokens']:
            tokens.append(t)
        samples.append(tokens)
        labels.append(tgt2idx[label])


class PubMedSectionEval(object):
    def __init__(self, task_path, seed=1111, evalType = ''):
        self.evalType = evalType

        logging.info('***** Transfer task : ' + self.evalType + ' *****\n\n')
        self.seed = seed
        self.train = self.loadFile(os.path.join(task_path, 'train_tokenized.json'))
        self.test = self.loadFile(os.path.join(task_path, 'test_tokenized.json'))

    def do_prepare(self, params, prepare):
        samples = self.train['X'] + self.test['X']
        return prepare(params, samples)

    def loadFile(self, fpath):
        pubmed_data = {'X': [], 'y': []}

        tgt2idx = {'treatment': 0, 'other': 1, 'infection': 2, 'pathology': 3, 'fauna': 4, 'etymology': 5,
                   'diagnosis': 6, 'cause': 7, 'history': 8, 'classification': 9, 'prognosis': 10, 'research': 11,
                   'epidemiology': 12, 'symptom': 13, 'genetics': 14, 'management': 15,
                   'culture': 16, 'pathophysiology': 17, 'risk': 18, 'mechanism': 19, 'prevention': 20, 'surgery': 21,
                   'tomography': 22, 'geography': 23, 'medication': 24, 'complication': 25, 'screening': 26}

        with open(fpath, 'r') as f:
            pubmedjson = json.load(f)
            for article in pubmedjson:
                getAnnotatedSentencesFromArticle(article, tgt2idx, pubmed_data['X'], pubmed_data['y'])
        return pubmed_data

    def run(self, params, batcher):
        train_embeddings, test_embeddings = [], []

        train_samples = self.train['X']
        train_labels = self.train['y']

        test_samples = self.test['X']
        test_labels = self.test['y']

        # Get train embeddings
        for ii in range(0, len(train_labels), params.batch_size):
            batch = train_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            train_embeddings.append(embeddings)

        train_embeddings = np.vstack(train_embeddings)
        logging.info('Computed train embeddings')

        # Get test embeddings
        for ii in range(0, len(test_labels), params.batch_size):
            batch = test_samples[ii:ii + params.batch_size]
            embeddings = batcher(params, batch)
            test_embeddings.append(embeddings)
            # for e in embeddings:
            #    test_embeddings.append(e)
        test_embeddings = np.vstack(test_embeddings)
        logging.info('Computed test embeddings')

        config_classifier = {'nclasses': 27, 'seed': self.seed,
                             'usepytorch': params.usepytorch,
                             'classifier': params.classifier,
                             'kfold': params.kfold}
        clf = KFoldClassifier({'X': train_embeddings,
                               'y': np.array(train_labels)},
                              {'X': test_embeddings,
                               'y': np.array(test_labels)},
                              config_classifier)
        devacc, testacc, _ = clf.run()
        logging.debug('\nDev acc : {0} Test acc : {1} \
            for ' + self.evalType + " ".format(devacc, testacc))
        return {'devacc': devacc, 'acc': testacc,
                'ndev': len(self.train['X']), 'ntest': len(self.test['X'])}
