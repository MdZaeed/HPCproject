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
class CSDivergence(MetricInterface):

    def __init__(self):
        super().__init__()
        self.bins = 100
        self.method = None
        self.reNormalize = False
        self.name = "Correlation-wise Smoothing JS Divergence" if not self.reNormalize else "Correlation-wise Smoothing JS Divergence (norm)"
        self._logger = logging.getLogger('CSDivergenceAl')

    def calculateMetric(self, samples, signatures, indexes, window=60):
        # Re-normalization is useful with datasets that have mostly static or slowly-changing data, in which
        # first-order derivatives would be close to 0 and hence not captured by a normal histogram
        if not self.reNormalize:
            realDiv, imagDiv, noImagDiv = self._calculateMetric(samples, signatures)
        else:
            realDiv, imagDiv, noImagDiv = self._calculateMetricRenormalize(samples, signatures)

        self._logger.info("        Real: %s - Imag: %s - NoImag: %s" % (realDiv, imagDiv, noImagDiv))
        return (realDiv + imagDiv) / 2

    def _calculateMetric(self, samples, signatures):
        samples = self._normalize(np.copy(samples))
        signatures = np.copy(signatures.T)

        # Building the complex uncompressed data - every row is a separate variable
        sortingVec = CSMethod(100000).correlationSortingWalk(samples) if self.method is None else self.method.sortingVec
        samples = samples[sortingVec, 1:] + np.diff(samples[sortingVec, :], 1, axis=1)*1j

        realDiv = self._divergence(samples.real, signatures.real, 0, 1, self.bins)
        imagDiv = self._divergence(samples.imag, signatures.imag, -1, 1, self.bins * 2)
        noImagDiv = self._divergence(samples.imag, np.zeros(signatures.shape), -1, 1, self.bins * 2)

        return realDiv, imagDiv, noImagDiv

    def _calculateMetricRenormalize(self, samples, signatures):
        samples = np.copy(samples)
        signatures = np.copy(signatures.T)

        # Building the complex uncompressed data - every row is a separate variable
        sortingVec = CSMethod(100000).correlationSortingWalk(samples) if self.method is None else self.method.sortingVec
        samples = self._normalize(samples[sortingVec, 1:]) + self._normalize(np.diff(samples[sortingVec, :], 1, axis=1))*1j
        signatures = self._normalize(signatures.real) + self._normalize(signatures.imag)*1j

        realDiv = self._divergence(samples.real, signatures.real, 0, 1, self.bins)
        imagDiv = self._divergence(samples.imag, signatures.imag, 0, 1, self.bins)
        noImagDiv = self._divergence(samples.imag, np.zeros(signatures.shape), 0, 1, self.bins)

        return realDiv, imagDiv, noImagDiv

    def _normalize(self, features):
        # Min-max normalization: we do not use the Z-score to avoid messing with the variance
        fMax = np.amax(features, axis=1) if self.method is None or self.reNormalize else self.method.fMax
        fMin = np.amin(features, axis=1) if self.method is None or self.reNormalize else self.method.fMin
        newFeatures = np.zeros(features.shape)
        for ctr in range(features.shape[0]):
            newFeatures[ctr, :] = (features[ctr, :] - fMin[ctr]) / (fMax[ctr] - fMin[ctr] + 0.0001)
        return newFeatures.clip(0, 1)

    def _getHistMatrix(self, features, fMin, fMax, nb):
        binVec = np.linspace(fMin, fMax, nb+1)
        histM = np.zeros((features.shape[0], nb))

        for idx in range(features.shape[0]):
            histM[idx, :] = np.multiply(np.histogram(features[idx, :], bins=binVec)[0], 1 / features.shape[1])

        return histM

    def _divergence(self, features, features2, b1, b2, nb):
        # Giving the same weight to the samples and the signatures
        histM = self._getHistMatrix(features, b1, b2, nb)
        histM2 = self._getHistMatrix(features2, b1, b2, nb)

        # Nearest neighbor interpolation
        histM2NN = np.zeros(histM.shape)
        for ctr in range(histM.shape[0]):
            roundIdx = int(np.round((histM2.shape[0]-1)*ctr/(histM.shape[0]-1)))
            histM2NN[ctr, :] = histM2[roundIdx, :]

        # Re-scaling of matrices
        histM = np.multiply(histM, 1 / histM.shape[0])
        histM2NN = np.multiply(histM2NN, 1 / histM2NN.shape[0])

        # Final matrix containing the two flattened histograms
        distMatrix = np.hstack((histM.reshape((histM.size, 1), order="C"), histM2NN.reshape((histM2NN.size, 1), order="C")))
        weights = [0.5, 0.5]

        entropySum = entropy(np.sum(np.multiply(distMatrix, weights), axis=1), base=2)
        sumEntropy = np.sum([entropy(distMatrix[:, idx], base=2)*weights[idx] for idx in range(distMatrix.shape[1])])

        return entropySum - sumEntropy
