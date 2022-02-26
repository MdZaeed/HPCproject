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
from time import perf_counter
import statistics as stats
import numpy as np
import logging


class ScalabilityTestCase(TestCaseInterface):

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger('ScalabilityTes')
        self.sensorsPath = ""
        self.responsesPath = ""
        self.windowWise = False
        self.steps = np.linspace(10, 10000, 200)
        self.fixedAxis = 100
        self.trials = 20

    def runTest(self, signatureMethod, errorMethods=None):
        self._logger.info('Scalability Evaluation test case')
        self._logger.info('Signature method: %s' % signatureMethod.name)
        self._logger.info('Error methods: None')
        self._logger.info('Testing mode: windows' if self.windowWise else 'Testing mode: dimensions')

        timingsTot = []
        for st in self.steps:
            dataset = np.random.randint(10000, size=(self.fixedAxis, int(st)+100)) if self.windowWise else np.random.randint(10000, size=(int(st), self.fixedAxis+100))
            timingsPart = []
            signatureMethod.clear()
            signatureMethod.computeSignature(dataset, dataset.shape[1] - 1, dataset.shape[1] - 100)
            for trial in range(self.trials):
                t_start = perf_counter()
                signatureMethod.computeSignature(dataset, dataset.shape[1]-1, dataset.shape[1]-100)
                timingsPart.append(perf_counter() - t_start)
            try:
                timingsTot.append(np.median(timingsPart))
            except:
                timingsTot.append(np.mean(timingsPart))

        print("\nEvaluation steps:\n")
        print(self.steps)
        print("\nResulting timings:\n")
        print(timingsTot)
