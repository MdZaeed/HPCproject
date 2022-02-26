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

from metrics.metricInterface import MetricInterface
from scipy.stats import entropy, zscore
import numpy as np


class JensenShannonDivergence(MetricInterface):

    def __init__(self):
        super().__init__()
        self.name = "Jensen-Shannon Divergence"

        self.cross = False
        self.bins = 100
        self._samplesHist = None
        self._signaturesHist = None

    def calculateMetric(self, samples, signatures, indexes, window=60):
        samples = np.copy(samples)
        signatures = np.copy(signatures)

        # In this case no flattening is required - the distributions of the different time lags are identical
        samples = self._normalize(samples.T)
        signatures = self._normalize(signatures)

        return self._divergence(samples, signatures) if not self.cross else self._crossDivergence(samples, signatures)

    def _normalize(self, features):
        # Z-score normalization - this is required to make the distributions comparable
        # In case of constant time series the zscore will be nan - we can replace those with zeros
        features = np.nan_to_num(zscore(features, axis=0))
        return features

    def _getHistMatrix(self, features, fMin, fMax):
        binVec = np.linspace(fMin, fMax, self.bins+1)
        histM = np.zeros((self.bins, features.shape[1]))

        for idx in range(features.shape[1]):
            histM[:, idx], _ = np.histogram(features[:, idx], bins=binVec, density=True)

        return histM

    def _divergence(self, features, features2):
        # Computing global maxima and minima over both matrices to define histogram boundaries
        fMax = np.amax([np.amax(features), np.amax(features2)])
        fMin = np.amin([np.amin(features), np.amin(features2)])

        # Giving the same weight to the samples and the signatures
        histM = np.hstack((self._getHistMatrix(features, fMin, fMax), self._getHistMatrix(features2, fMin, fMax)))
        weights = np.asarray([0.5 / features.shape[1]] * features.shape[1] + [0.5 / features2.shape[1]] * features2.shape[1])

        # Weighted sum of the single PDFs
        entropySum = entropy(np.sum(np.multiply(histM, weights), axis=1))
        sumEntropy = np.sum([entropy(histM[:, idx])*weights[idx] for idx in range(histM.shape[1])])

        return entropySum - sumEntropy

    # A take on the JS divergence which tries to eliminate the intrinsic divergence of the datasets - do not use
    def _crossDivergence(self, features, features2):
        fMax = np.amax([np.amax(features), np.amax(features2)])
        fMin = np.amin([np.amin(features), np.amin(features2)])

        histM = self._getHistMatrix(features, fMin, fMax)
        histM2 = self._getHistMatrix(features2, fMin, fMax)
        weight = 0.5
        divs = []

        for i in range(histM.shape[1]):
            for j in range(histM2.shape[1]):
                pairM = np.column_stack((histM[:, i], histM2[:, j]))
                entropySum = entropy(np.sum(pairM*weight, axis=1))
                sumEntropy = np.sum([entropy(pairM[:, idx])*weight for idx in range(pairM.shape[1])])
                divs.append(entropySum - sumEntropy)
        return np.average(divs)
