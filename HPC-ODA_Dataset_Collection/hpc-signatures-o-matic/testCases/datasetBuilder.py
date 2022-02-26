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

from methods.signatureInterface import SignatureInterface
from metrics.metricInterface import MetricInterface
from testCases.datasetThread import DatasetThread
from os import listdir
from os.path import isfile, isdir, join
from copy import copy
from random import randint
import logging
import numpy as np
from time import perf_counter


class DatasetBuilder:
    """

    Loads a dataset at a given path and builds signatures for it.

    """

    def __init__(self):
        self.subdirMode = False
        self.subdirFilter = ""
        self.fileFilter = ""
        self.stringResponses = False
        self.shiftResponses = 0
        self.smoothResponses = False
        self.step = 10
        self.window = 60
        self.numThreads = 4
        self._logger = logging.getLogger('DatasetBuilder')

        self.responses = None
        self.features = None
        self.sensorReadings = None
        self.featureIdx = None
        self.featureNames = None
        self.fileNames = None

    def loadDataset(self, sensorsPath, responsesPath, signatureMethod, errorMethods=None):
        """

        Loads a dataset fetching and aligning all sensors in the directory of "sensorPath", and loading a responses file
        from the directory of "responsesPath". The latter file must be named "responses.csv".

        :param sensorsPath:     Path in which to look for sensor CSV files.
        :param responsesPath:   Path in which to look for a "responses.csv" responses file.
        :param signatureMethod: Object implementing SignatureInterface, to create feature sets from sensor data.
        :param errorMethods:     Optional. List of objects implementing MetricInterface, to quantify information loss.

        """
        if signatureMethod is None or not isinstance(signatureMethod, SignatureInterface):
            raise TypeError
        if errorMethods is not None and not all(isinstance(err, MetricInterface) for err in errorMethods):
            raise TypeError
        if (isinstance(sensorsPath, list) or isinstance(sensorsPath, tuple)) and len(sensorsPath) == 0:
            raise TypeError
        if (isinstance(responsesPath, list) or isinstance(responsesPath, tuple)) and len(responsesPath) == 0:
            raise TypeError

        if not isinstance(sensorsPath, list) and not isinstance(sensorsPath, tuple):
            sensorsPath = [sensorsPath]
        if not isinstance(responsesPath, list) and not isinstance(responsesPath, tuple):
            responsesPath = [responsesPath]

        buildTime = 0
        # Resetting state of the signature method
        signatureMethod.clear()
        if not self.subdirMode:
            buildTime = self._loadDataset(sensorsPath, responsesPath, signatureMethod)
        else:
            subDirs = [d for d in listdir(responsesPath[0]) if isdir(join(responsesPath[0], d)) and (self.subdirFilter == "" or self.subdirFilter in d)]
            totalResponses = None
            totalFeatures = None
            totalReadings = None
            totalFeatureIdx = None
            for subDir in subDirs:
                self._logger.info("Building sub-dataset " + subDir + ":")
                subSensorPaths = [join(s, subDir) for s in sensorsPath]
                subResponsePaths = [join(r, subDir) for r in responsesPath]
                buildTime = buildTime + self._loadDataset(subSensorPaths, subResponsePaths, signatureMethod)
                totalFeatures = self.features if totalFeatures is None else np.vstack((totalFeatures, self.features))
                totalResponses = self.responses if totalResponses is None else np.concatenate((totalResponses, self.responses))
                totalFeatureIdx = self.featureIdx if totalFeatureIdx is None else np.concatenate((totalFeatureIdx, np.add(self.featureIdx, totalReadings.shape[1])))
                totalReadings = self.sensorReadings if totalReadings is None else np.hstack((totalReadings, self.sensorReadings))

            self.responses = totalResponses
            self.features = totalFeatures
            self.sensorReadings = totalReadings
            self.featureIdx = totalFeatureIdx

        # Error computation
        if errorMethods is not None:
            self._logger.info("    Error metrics:")
            for errorMet in errorMethods:
                errors = errorMet.calculateMetric(self.sensorReadings, self.features, self.featureIdx, self.window)
                self._logger.info("        Average %s: %s" % (errorMet.name, np.mean(errors)))

        self._logger.info("    Processed a total of %s sensor entries" % self.sensorReadings.shape[1])
        self._logger.info("    Window size is equal to %s" % self.window)
        self._logger.info("    Step value is equal to %s" % self.step)
        self._logger.info("    Built a dataset of %s feature sets" % self.features.shape[0])
        self._logger.info("    Each feature set contains %s elements" % self.features.shape[1])
        self._logger.info("    Average time per feature set : %s" % (buildTime/self.features.shape[0]))
        self._logger.info("    Total time elapsed : %s" % buildTime)

        self.sensorReadings = None
        self.featureIdx = None

    def shuffleDataset(self):
        """
ƒ
        Shuffles a dataset that has been stored in memory.

        """
        if self.features is not None and self.responses is not None:
            totalLen = self.features.shape[0]
            numSwaps = int(totalLen / 2)
            for i in range(numSwaps):
                idx1 = randint(0, totalLen - 1)
                idx2 = randint(0, totalLen - 1)
                tmpResponse = self.responses[idx2]
                self.responses[idx2] = self.responses[idx1]
                self.responses[idx1] = tmpResponse
                self.features[[idx1, idx2], :] = self.features[[idx2, idx1], :]

    def truncateDataset(self, size):
        """

        Truncates a loaded dataset to a specified size.

        :param size: Target size.
        """
        if self.features is not None and self.responses is not None and size < self.features.shape[0]:
            self.features = self.features[0:size, :]
            self.responses = self.responses[0:size]

    def _loadDataset(self, sensorsPath, responsesPath, signatureMethod):
        responses = []
        self.sensorReadings = []
        sensorFiles = {}

        for rdx, currPath in enumerate(sensorsPath):
            rawResponses = []
            rawSensorReadings = []

            # Loading responses
            self._logger.info("    Starting dataset building in directory %s..." % currPath)
            responseFile = open(responsesPath[rdx] + "/responses.csv", 'r')
            responseFile.readline()
            for l in responseFile:
                lineTuple = l.split(',')
                if len(lineTuple) != 2:
                    raise IOError
                else:
                    rawResponses.append(lineTuple[1] if self.stringResponses else int(lineTuple[1]))
            responseFile.close()
            self._logger.debug("    Loaded responses file: %s" % responseFile.name)

            if self.shiftResponses > 0:
                if not self.smoothResponses:
                    rawResponses = rawResponses[self.shiftResponses:]
                else:
                    newResp = []
                    newLen = len(rawResponses) - self.shiftResponses
                    for idx in range(1, newLen):
                        newResp.append(np.average(rawResponses[idx:(idx + self.shiftResponses)]))
                    rawResponses = newResp

            # Loading sensor data in a row-major Python matrix
            sensorFiles = {}
            for f in sorted([f for f in listdir(sensorsPath[0]) if isfile(join(sensorsPath[0], f)) and ".csv" in f]):
                if self.fileFilter == "" or self.fileFilter not in f:
                    sensorFiles[f] = open(join(currPath, f), 'r')
                    sensorFiles[f].readline()
                    sensorVector = []
                    for l in sensorFiles[f]:
                        lineTuple = l.split(',')
                        if len(lineTuple) != 2:
                            raise IOError
                        else:
                            sensorVector.append(float(lineTuple[1]))
                    sensorFiles[f].close()
                    rawSensorReadings.append(copy(sensorVector))
                    self._logger.debug("    Loaded sensor file: %s" % sensorFiles[f].name)

            # Trimming excess sensor readings to have uniform shape
            minSize = min([len(l) for l in rawSensorReadings])
            if minSize > len(rawResponses):
                minSize = len(rawResponses)
            else:
                rawResponses = rawResponses[0:minSize]

            for idx, v in enumerate(rawSensorReadings):
                if len(v) > minSize:
                    rawSensorReadings[idx] = v[0:minSize]

            # Concatenating all sensor readings and responses from "twin" directories
            responses = responses + rawResponses
            if len(self.sensorReadings) == 0:
                [self.sensorReadings.append(copy(v)) for v in rawSensorReadings]
            else:
                for rr, v in enumerate(rawSensorReadings):
                    self.sensorReadings[rr] = self.sensorReadings[rr] + copy(v)

        # Converting to Numpy format
        self.sensorReadings = np.asarray(self.sensorReadings)
        responses = np.asarray(responses, dtype='object')

        self._logger.info("    Loaded %s sensors" % len(self.sensorReadings))
        self._logger.info("    Each sensor has %s entries" % len(self.sensorReadings[0]))

        # Generating threads and assigning slices
        self.featureIdx = [(self.window + self.step*v) for v in range(int((self.sensorReadings.shape[1]-self.window)/self.step))]
        sliceLen = int(len(self.featureIdx) / self.numThreads)

        threads = []
        for idx in range(self.numThreads):
            tSliceLen = sliceLen if idx < self.numThreads-1 else (len(self.featureIdx) - sliceLen*(self.numThreads-1))
            tBaseIdx = sliceLen*idx
            threads.append(DatasetThread(signatureMethod=signatureMethod, inputSensors=self.sensorReadings, inputResponses=responses,
                                         indexes=self.featureIdx[tBaseIdx:(tBaseIdx+tSliceLen)], window=self.window))

        startTime = perf_counter()
        # Computing signatures
        [t.start() for t in threads]
        self._logger.info("    Started %s threads." % self.numThreads)
        [t.join() for t in threads]
        self._logger.info("    Threads have terminated.")
        buildTime = perf_counter() - startTime

        # Collecting signatures from joined threads
        self.featureNames = signatureMethod.computeNames(list(sensorFiles.keys()))
        self.fileNames = list(sensorFiles.keys())
        self.features = None
        self.responses = None

        for t in threads:
            self.features = np.vstack((self.features, t.features)) if self.features is not None else t.features
            self.responses = np.concatenate((self.responses, t.responses)) if self.responses is not None else t.responses

        if not self.stringResponses:
            self.responses = np.asarray(self.responses, dtype='int')

        # Forcing memory de-allocation
        threads = None
        responses = None

        return buildTime
