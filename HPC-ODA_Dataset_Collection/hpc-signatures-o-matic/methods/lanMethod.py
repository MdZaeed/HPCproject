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
import numpy as np


class LanMethod(SignatureInterface):

    def __init__(self):
        super().__init__()
        self.name = "Lan Method"
        self._numPoints = 10

    def computeSignature(self, samples, index, window=60):
        if index-window < 0:
            raise ArithmeticError
        numPoints = self._numPoints if window >= self._numPoints else window
        divisor = int(window/numPoints)
        featureSet = np.zeros(samples.shape[0]*numPoints)
        for rowIdx in range(samples.shape[0]):
            featureSet[(rowIdx*numPoints):((rowIdx+1)*numPoints)] = self._subSample(samples[rowIdx, (index-window+1):(index+1)], divisor)
        return featureSet

    def computeNames(self, metricNames):
        names = []
        for metric in metricNames:
            mName = metric.replace(".csv", "")
            names += [mName + "_p" + str(idx) for idx in range(self._numPoints)]
        return names

    def _subSample(self, arr, n):
        rest = len(arr) - n*int(len(arr)/n)
        return np.mean(arr[rest:].reshape(-1, n), 1)
