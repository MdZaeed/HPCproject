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
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import make_scorer, mean_squared_error
from time import perf_counter
import statistics as stats
import numpy as np
import logging


class InfrastructureManagementTestCase(TestCaseInterface):

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger('Infrastructure')
        self.sensorsPath = "./datasets/infrastructure_management/sensors"
        self.responsesPath = "./datasets/infrastructure_management/responses"
        self.window = 30
        self.step = 6
        self.tests = 5
        self.importances = True
        self.numImportances = 40
        self.classifier = RandomForestRegressor(n_estimators=50)
        # self.classifier = SVC(kernel='rbf')
        # self.classifier = DecisionTreeClassifier(criterion='gini')
        # self.classifier = MLPClassifier(activation='relu', hidden_layer_sizes=(1000, 1000))

    def runTest(self, signatureMethod, errorMethods=None):
        self._logger.info('Infrastructure Management test case')
        self._logger.info('Regressor: %s' % self.classifier.__class__.__name__)
        self._logger.info('Signature method: %s' % signatureMethod.name)
        if errorMethods is None or len(errorMethods) == 0:
            self._logger.info('Error methods: None')
        else:
            self._logger.info('Error methods: %s' % ', '.join([err.name for err in errorMethods]))
        self._logger.info('Dataset building phase:')
        builder = DatasetBuilder()
        builder.window = self.window
        builder.step = self.step
        builder.shiftResponses = 1
        builder.stringResponses = False
        builder.subdirMode = True
        builder.loadDataset(self.sensorsPath, self.responsesPath, signatureMethod, errorMethods)
        builder.shuffleDataset()
        scorer = {"relerr": make_scorer(self._nrmse, greater_is_better=False)}
        globalScores = None
        times = []

        self._logger.info('Evaluation phase:')
        if self.tests > 0:
            for testID in range(self.tests):
                t_start = perf_counter()
                scores = cross_validate(self.classifier, builder.features, builder.responses, cv=5, scoring=scorer, n_jobs=-1)
                times.append(perf_counter() - t_start)
                self._logger.info('    Test %s completed...' % testID)
                if globalScores is None:
                    globalScores = scores
                else:
                    globalScores['test_relerr'] = np.concatenate((globalScores['test_relerr'], scores['test_relerr']))
            mean = -globalScores['test_relerr'].mean()
            confidence = globalScores['test_relerr'].std() * 1.96 / np.sqrt(len(globalScores['test_relerr']))
            self._logger.info('    Normalized root mean squared error : %s (+/- %s)' % ('{:.6f}'.format(mean), '{:.6f}'.format(confidence)))
            self._logger.info("    Average time per test : %s" % stats.mean(times))

        if self.importances and (isinstance(self.classifier, RandomForestRegressor) or isinstance(self.classifier, DecisionTreeRegressor)):
            self.classifier.fit(builder.features, builder.responses)
            importances = self.classifier.feature_importances_
            indices = np.argsort(importances)[::-1]
            self._logger.info("    Most important features:")
            for idx in range(self.numImportances if self.numImportances < len(indices) else len(indices)):
                self._logger.info("        %s: %s" % (builder.featureNames[indices[idx]], importances[indices[idx]]))

    def _relativeError(self, y_true, y_pred):
        return np.mean(np.divide(np.abs(np.subtract(y_true, y_pred)), y_true, where=y_true != 0))

    def _nrmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))

