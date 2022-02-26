# ================================================================================
# Copyright (C) 2020 Alessio Netti
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.If not, see <https://www.gnu.org/licenses/>.
# ================================================================================

from testCases.testCaseInterface import TestCaseInterface
from testCases.datasetBuilder import DatasetBuilder
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score
from time import perf_counter
import statistics as stats
import numpy as np
import logging


class CrossArchitectureTestCase(TestCaseInterface):

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger('CrossArchitect')
        self.sensorsPath = "./datasets/cross_architecture/sensors"
        self.responsesPath = "./datasets/cross_architecture/responses"
        self.window = 30
        self.step = 2
        self.tests = 5
        self.importances = True
        self.numImportances = 40
        self.classifier = RandomForestClassifier(n_estimators=50)
        # self.classifier = SVC(kernel='rbf')
        # self.classifier = DecisionTreeClassifier(criterion='gini')
        # self.classifier = MLPClassifier(activation='relu', hidden_layer_sizes=(100, 100))

    def runTest(self, signatureMethod, errorMethods=None):
        self._logger.info('Cross-architecture test case')
        self._logger.info('Classifier: %s' % self.classifier.__class__.__name__)
        self._logger.info('Signature method: %s' % signatureMethod.name)
        if errorMethods is None or len(errorMethods) == 0:
            self._logger.info('Error methods: None')
        else:
            self._logger.info('Error methods: %s' % ', '.join([err.name for err in errorMethods]))
        self._logger.info('Dataset building phase:')

        self._logger.info('Building KNL dataset:')
        builder = DatasetBuilder()
        builder.window = self.window
        builder.step = self.step
        builder.stringResponses = True
        builder.subdirMode = False
        builder.loadDataset(self.sensorsPath + "/knl-node", self.responsesPath + "/knl-node", signatureMethod, errorMethods)

        self._logger.info('Building Skylake dataset:')
        builder2 = DatasetBuilder()
        builder2.window = self.window
        builder2.step = self.step
        builder2.stringResponses = True
        builder2.subdirMode = False
        builder2.loadDataset(self.sensorsPath + "/skylake-node", self.responsesPath + "/skylake-node", signatureMethod, errorMethods)

        self._logger.info('Building AMD dataset:')
        builder3 = DatasetBuilder()
        builder3.window = self.window
        builder3.step = self.step
        builder3.stringResponses = True
        builder3.subdirMode = False
        builder3.loadDataset(self.sensorsPath + "/amd-node", self.responsesPath + "/amd-node", signatureMethod, errorMethods)

        origSize = builder.features.shape[0]
        builder.features = np.vstack((builder.features, builder2.features, builder3.features))
        builder.responses = np.concatenate((builder.responses, builder2.responses, builder3.responses))

        # Separate folds, one for each architecture
        folds = [(list(range(0, origSize + builder2.features.shape[0])), list(range(origSize + builder2.features.shape[0], builder.features.shape[0]))),
                 (list(range(origSize, builder.features.shape[0])), list(range(0, origSize))),
                 (list(range(0, origSize)) + list(range(origSize + builder2.features.shape[0], builder.features.shape[0])), list(range(origSize, origSize + builder2.features.shape[0])))]

        self._logger.info('Final dataset statistics:')
        self._logger.info("    Built a dataset of %s feature sets" % builder.features.shape[0])
        self._logger.info("    Each feature set contains %s elements" % builder.features.shape[1])

        builder.shuffleDataset()
        globalScores = None
        times = []

        self._logger.info('Evaluation phase:')
        if self.tests > 0:
            for testID in range(self.tests):
                scorers = self._getScorerObjects(builder.responses)
                t_start = perf_counter()
                scores = cross_validate(self.classifier, builder.features, builder.responses, cv=5, scoring=scorers, n_jobs=-1)
                times.append(perf_counter() - t_start)
                self._logger.info('    Test %s completed...' % testID)
                if globalScores is None:
                    globalScores = scores
                else:
                    globalScores['test_weighted'] = np.concatenate((globalScores['test_weighted'], scores['test_weighted']))
                    for k, v in scores.items():
                        if 'test_' in k and k != 'test_weighted':
                            globalScores[k] = np.concatenate((globalScores[k], scores[k]))
            mean = globalScores['test_weighted'].mean()
            confidence = globalScores['test_weighted'].std() * 1.96 / np.sqrt(len(globalScores['test_weighted']))
            self._logger.info('    Global F1-Score : %s (+/- %s)' % ('{:.6f}'.format(mean), '{:.6f}'.format(confidence)))
            for k, v in globalScores.items():
                if 'test_' in k and k != 'test_weighted':
                    mean = globalScores[k].mean()
                    confidence = globalScores[k].std() * 1.96 / np.sqrt(len(globalScores[k]))
                    self._logger.info('    %s F1-Score : %s (+/- %s)' % (k.split('_')[1].rstrip(), '{:.6f}'.format(mean), '{:.6f}'.format(confidence)))
            self._logger.info("    Average time per test : %s" % stats.mean(times))

        if self.importances and (isinstance(self.classifier, RandomForestClassifier) or isinstance(self.classifier, DecisionTreeClassifier)):
            self.classifier.fit(builder.features, builder.responses)
            importances = self.classifier.feature_importances_
            indices = np.argsort(importances)[::-1]
            self._logger.info("    Most important features:")
            for idx in range(self.numImportances if self.numImportances < len(indices) else len(indices)):
                self._logger.info("        %s: %s" % (builder.featureNames[indices[idx]], importances[indices[idx]]))

    def _getScorerObjects(self, labels):
        labelSet = set(labels)
        scorers = {}
        for label in labelSet:
            scorers[label] = make_scorer(f1_score, average=None, labels=[label])
        scorers['weighted'] = make_scorer(f1_score, average='weighted')
        return scorers
