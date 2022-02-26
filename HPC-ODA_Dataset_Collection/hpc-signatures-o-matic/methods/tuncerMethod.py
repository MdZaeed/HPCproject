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


class TuncerMethod(SignatureInterface):

    def __init__(self):
        super().__init__()
        self.percentiles = [0, 5, 25, 50, 75, 95, 100]
        self.name = "Tuncer Method"
        self._numFeatures = 11

    def computeSignature(self, samples, index, window=60):
        if index-window < 0:
            raise ArithmeticError
        featureSet = np.zeros(samples.shape[0]*self._numFeatures)
        for rowIdx in range(samples.shape[0]):
            featureSet[self._numFeatures*rowIdx] = np.mean(samples[rowIdx, (index-window+1):(index+1)])
            featureSet[self._numFeatures*rowIdx + 1] = np.std(samples[rowIdx, (index-window+1):(index+1)])
            diffs = np.diff(samples[rowIdx, (index-window+1):(index+1)], 1)
            featureSet[self._numFeatures*rowIdx + 2] = np.sum(diffs)
            featureSet[self._numFeatures*rowIdx + 3] = np.sum(np.abs(diffs))
            featureSet[(self._numFeatures*rowIdx + 4):(self._numFeatures*rowIdx + 11)] = np.percentile(samples[rowIdx, (index-window+1):(index+1)], self.percentiles)
        return featureSet

    def computeNames(self, metricNames):
        names = []
        for metric in metricNames:
            mName = metric.replace(".csv", "")
            names.append(mName + "_mean")
            names.append(mName + "_std")
            names.append(mName + "_soc")
            names.append(mName + "_asoc")
            names.append(mName + "_min")
            names.append(mName + "_perc5")
            names.append(mName + "_perc25")
            names.append(mName + "_perc50")
            names.append(mName + "_perc75")
            names.append(mName + "_perc95")
            names.append(mName + "_max")
        return names

