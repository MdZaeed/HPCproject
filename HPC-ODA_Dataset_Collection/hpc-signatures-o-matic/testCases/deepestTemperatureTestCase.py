# ================================================================================
# Copyright (C) 2021 Alessio Netti
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

from methods.csMethod import CSMethod
from testCases.testCaseInterface import TestCaseInterface
from testCases.datasetBuilder import DatasetBuilder
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from random import uniform
# from sklearn.neural_network import MLPRegressor
from time import perf_counter
import statistics as stats
import numpy as np
import logging
import os


class DeepestTemperatureTestCase(TestCaseInterface):

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger('DeepestTempera')
        self.segment = "cn"  # can be "cn", "esb" or "gpu"
        self.sensorsPath = "./datasets/deepest_temperature_cn/sensors"
        self.responsesPath = "./datasets/deepest_temperature_cn/responses"
        self.mix = "None"  # can be "None", "Full", "Node" or "Socket"
        self.fileFilter = ""
        self.window = 6
        self.step = 1
        self.shiftResponses = 6
        self.tests = 5
        self.importances = True
        self.numImportances = 40
        self.classifier = RandomForestRegressor(n_estimators=50)
        # self.classifier = SVC(kernel='rbf')
        # self.classifier = DecisionTreeClassifier(criterion='gini')
        # self.classifier = MLPClassifier(activation='relu', hidden_layer_sizes=(1000, 1000))

        self.sockets = ["socket0", "socket1"]
        self.nodes = ["deepest.cn.s" + str("{:02d}".format(1+nodeId)) + "." for nodeId in range(16)]

        # self.errFile = open("err.csv", "a+")

    def runTest(self, signatureMethod, errorMethods=None):
        if self.segment == "cn":
            self.fileFilter = ""
            self.sockets = ["socket0", "socket1"]
            self.nodes = ["deepest.cn.s" + str("{:02d}".format(1 + nodeId)) + "." for nodeId in range(16)]
            self.sensorsPath = "./datasets/deepest_temperature_cn/sensors"
            self.responsesPath = "./datasets/deepest_temperature_cn/responses"
        elif self.segment == "esb":
            self.fileFilter = ".ib0."
            self.sockets = [""]
            self.nodes = ["deepest.esb.s" + str("{:02d}".format(26 + nodeId)) for nodeId in range(16)]
            self.sensorsPath = ["./datasets/deepest_temperature_esb/sensors"]
            self.responsesPath = ["./datasets/deepest_temperature_esb/responses"]
        elif self.segment == "gpu":
            self.fileFilter = ".ib0."
            self.sockets = ["gpu0"]
            self.nodes = ["deepest.esb.s" + str("{:02d}".format(26 + nodeId)) + "." for nodeId in range(16)]
            self.sensorsPath = ["./datasets/deepest_temperature_gpu/sensors"]
            self.responsesPath = ["./datasets/deepest_temperature_gpu/responses"]

        self._logger.info('DEEP-EST temperature prediction test case')
        self._logger.info('Regressor: %s' % self.classifier.__class__.__name__)
        self._logger.info('Signature method: %s' % signatureMethod.name)
        self._logger.info('System segment: %s' % self.segment)
        if errorMethods is None or len(errorMethods) == 0:
            self._logger.info('Error methods: None')
        else:
            self._logger.info('Error methods: %s' % ', '.join([err.name for err in errorMethods]))
        self._logger.info('Dataset building phase:')

        # We iterate over a set of datasets depending if we want to mix data from compute nodes or not
        datasetSet, responsesSet, namesSet, hostsSet, filesSet = self._buildMixedDataset(signatureMethod, errorMethods)
        errResults = []

        for dt in range(len(datasetSet)):
            datasetNow = datasetSet[dt]
            responsesNow = responsesSet[dt]
            namesNow = namesSet[dt]
            hostsNow = hostsSet[dt]
            filesNow = filesSet[dt]

            scorer = {"relerr": make_scorer(self._nrmse, greater_is_better=False)}
            globalScores = None
            times = []
            self._logger.info('-- %s - evaluation phase:' % hostsNow)
            if self.tests > 0:
                for testID in range(self.tests):
                    t_start = perf_counter()
                    scores = cross_validate(self.classifier, datasetNow, responsesNow, cv=5, scoring=scorer, n_jobs=-1)
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
                errResults.append(mean)

            # Saving model data for further use
            if isinstance(signatureMethod, CSMethod):
                signatureMethod.saveToFile("./results/deepest_cs_model_" + self.mix + "_" + hostsNow + ".json")
                fNameOut = open("./results/deepest_cs_names_" + self.mix + "_" + hostsNow + ".csv", "w")
                for n in filesNow:
                    fNameOut.write(n + "\n")
                fNameOut.close()
            self._trainAndSaveOpenCV("./results/deepest_opencv_model_" + self.mix + "_" + hostsNow + ".yml", datasetNow, responsesNow, 80)

            if self.importances and (isinstance(self.classifier, RandomForestRegressor) or isinstance(self.classifier, DecisionTreeRegressor)):
                self.classifier.fit(datasetNow, responsesNow)
                importances = self.classifier.feature_importances_
                indices = np.argsort(importances)[::-1]
                self._logger.info("    Most important features:")
                for idx in range(self.numImportances if self.numImportances < len(indices) else len(indices)):
                    self._logger.info("        %s: %s" % (namesNow[indices[idx]], importances[indices[idx]]))

        self._logger.info("-- Final results in array form: %s" % errResults)
        return errResults

    def _relativeError(self, y_true, y_pred):
        return np.mean(np.divide(np.abs(np.subtract(y_true, y_pred)), y_true, where=y_true != 0))

    def _nrmse(self, y_true, y_pred):
        err = np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))
        # for idx in range(len(y_true)):
        #     self.errFile.write(str(y_true[idx]) + "," + str(y_pred[idx]) + "\n")
        return err

    def _buildMixedDataset(self, signatureMethod, errorMethods):
        datasetSet = []
        responsesSet = []
        namesSet = []
        hostsSet = []
        filesSet = []

        if (self.mix == "Node" or self.mix == "Socket") and self.segment != "cn":
            self._logger.info('-- Invalid combination of mix type %s and segment %s. Falling back to None.' % (self.mix, self.segment))

        if self.mix == "Full":
            self._logger.info('-- Global - dataset building phase:')
            ft, rs, ftn, fsn = self._buildMixedDatasetInt(signatureMethod, errorMethods, self.sensorsPath, self.responsesPath, True, "")
            datasetSet.append(ft)
            responsesSet.append(rs)
            namesSet.append(ftn)
            filesSet.append(fsn)
            hostsSet.append("Global")
        elif self.mix == "None":
            for el in os.listdir(self.sensorsPath):
                if os.path.isdir(os.path.join(self.sensorsPath, el)) and "rcu" not in el:
                    sensorsNow = os.path.join(self.sensorsPath, el)
                    responsesNow = os.path.join(self.responsesPath, el)
                    self._logger.info('-- %s - dataset building phase:' % el)
                    ft, rs, ftn, fsn = self._buildMixedDatasetInt(signatureMethod, errorMethods, sensorsNow, responsesNow, False, "")
                    datasetSet.append(ft)
                    responsesSet.append(rs)
                    namesSet.append(ftn)
                    filesSet.append(fsn)
                    hostsSet.append(el)
        elif self.mix == "Socket" or self.mix == "Node":
            dtList = self.sockets if self.mix == "Socket" else self.nodes
            for el in dtList:
                self._logger.info('-- %s - dataset building phase:' % el)
                ft, rs, ftn, fsn = self._buildMixedDatasetInt(signatureMethod, errorMethods, self.sensorsPath, self.responsesPath, True, el)
                datasetSet.append(ft)
                responsesSet.append(rs)
                namesSet.append(ftn)
                filesSet.append(fsn)
                hostsSet.append(el)
        return datasetSet, responsesSet, namesSet, hostsSet, filesSet

    def _buildMixedDatasetInt(self, signatureMethod, errorMethods, sensorsPath, responsesPath, subdirMode, subdirFilter):
        builder = DatasetBuilder()
        builder.window = self.window
        builder.step = self.step
        builder.subdirMode = subdirMode
        builder.subdirFilter = subdirFilter
        builder.fileFilter = self.fileFilter
        builder.shiftResponses = self.shiftResponses
        builder.smoothResponses = True
        builder.stringResponses = False
        builder.loadDataset(sensorsPath, responsesPath, signatureMethod, errorMethods)
        builder.shuffleDataset()
        # builder.features, builder.responses = self._underSample(builder.features, builder.responses)

        return builder.features, builder.responses, builder.featureNames, builder.fileNames

    def _underSample(self, features, responses):
        nBins = 10
        rMin = min(responses)
        rMax = max(responses)
        wBin = (rMax - rMin) / nBins
        hist, edges = np.histogram(responses, bins=nBins, density=False)
        histCoeff = len(responses) / nBins
        histProbs = [histCoeff / h for h in hist]
        toDelete = []

        for idx, val in enumerate(responses):
            hVal = histProbs[int((val - rMin) // wBin) if val < rMax else (nBins-1)]
            if uniform(0, 1) > hVal:
                toDelete.append(idx)

        self._logger.info("Balancing dataset by removing %s elements" % len(toDelete))

        features = np.delete(features, toDelete, axis=0)
        responses = np.delete(responses, toDelete, axis=0)
        return features, responses

    # Dummy method to be implemented in a different class
    def _trainAndSaveOpenCV(self, outPath, samples, responses, split=100):
        pass





