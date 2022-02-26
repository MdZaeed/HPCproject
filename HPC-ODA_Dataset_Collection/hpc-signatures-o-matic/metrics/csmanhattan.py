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
# These scripts reproduce the experiments carried out using the Correlation-wise Smoothing (CS) method in the paper
# "Correlation-wise Smoothing: Lightweight Knowledge Extraction for HPC Monitoring Data" by Alessio Netti et al.

from metrics.metricInterface import MetricInterface
from methods.csMethod import CSMethod
from scipy.stats import entropy
import numpy as np
import logging


# This divergence metric is intended to be used with data processed via the CS method only.
# It is based on the Jensen-Shannon divergence.
class CSManhattan(MetricInterface):

    def __init__(self):
        super().__init__()
        self.name = "Correlation-wise Smoothing Manhattan Distance"

        self.method = None
        self._logger = logging.getLogger('ManhattanDista')

    def calculateMetric(self, samples, signatures, indexes, window=60):
        samples = self._normalize(np.copy(samples))
        signatures = np.copy(signatures.T)

        # Building the complex uncompressed data - every row is a separate variable
        sortingVec = CSMethod(100000).correlationSortingWalk(samples) if self.method is None else self.method.sortingVec
        samples = samples[sortingVec, 1:] + np.diff(samples[sortingVec, :], 1, axis=1)*1j

        realDiv = self._distance(samples.real, signatures.real)
        imagDiv = self._distance(samples.imag, signatures.imag)
        noImagDiv = self._distance(samples.imag, np.zeros(signatures.shape))
        self._logger.info("        Real: %s - Imag: %s - NoImag: %s" % (realDiv, imagDiv, noImagDiv))
        return (realDiv + imagDiv) / 2

    def _normalize(self, features):
        # Min-max normalization: we do not use the Z-score to avoid messing with the variance
        fMax = np.amax(features, axis=1) if self.method is None else self.method.fMax
        fMin = np.amin(features, axis=1) if self.method is None else self.method.fMin
        newFeatures = np.zeros(features.shape)
        for ctr in range(features.shape[0]):
            newFeatures[ctr, :] = (features[ctr, :] - fMin[ctr]) / (fMax[ctr] - fMin[ctr] + 0.0001)
        return newFeatures.clip(0, 1)

    def _distance(self, features, features2):
        # Nearest neighbor interpolation
        constIdx = (features2.shape[0]-1)/(features.shape[0]-1)
        constIdx2 = (features2.shape[1]-1)/(features.shape[1]-1)

        # Two-pass interpolation: first the rows, then the columns
        features2NNCols = np.zeros((features.shape[0], features2.shape[1]))
        for ctr in range(features.shape[0]):
            features2NNCols[ctr, :] = features2[int(np.round(constIdx * ctr)), :]

        features2NN = np.zeros(features.shape)
        for ctr in range(features.shape[1]):
            features2NN[:, ctr] = features2NNCols[:, int(np.round(constIdx2*ctr))]

        return np.average(np.abs(np.subtract(features, features2NN)))
