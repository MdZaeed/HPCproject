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
import numpy as np


class CumulativePercentVariance(MetricInterface):

    def __init__(self):
        super().__init__()
        self.name = "Cumulative Percent Variance"

    def calculateMetric(self, samples, signatures, indexes, window=60):
        signatures = np.copy(signatures)
        flattenedSamples = np.zeros((len(indexes), samples.shape[0]*window))
        for ctr, idx in enumerate(indexes):
            flattenedSamples[ctr, :] = np.reshape(samples[:, (idx-window+1):(idx+1)], samples.shape[0]*window)

        # Performing a Z-score instead of simply dividing by the max will not work for data with 0 or very small std
        flattenedSamples = self._normalize(flattenedSamples)
        signatures = self._normalize(signatures)

        # Using a cov matrix instead of correlation because - once again - metrics with
        # very small std may show strange behaviors
        covMatSamples = np.cov(flattenedSamples.T)
        covMatSamples[covMatSamples == np.inf] = 0
        covMatSamples[covMatSamples == np.NINF] = 0
        covMatSamples = np.nan_to_num(covMatSamples)
        eigSamples = np.linalg.eig(covMatSamples)[0]
        covMatSignatures = np.cov(signatures.T)
        covMatSignatures[covMatSignatures == np.inf] = 0
        covMatSignatures[covMatSignatures == np.NINF] = 0
        covMatSignatures = np.nan_to_num(covMatSignatures)
        eigSignatures = np.linalg.eig(covMatSignatures)[0]

        # For real data complex eigenvalues are always in conjugate pairs, so we can simply drop the
        # imaginary part upon summing them
        cpv = 100 * np.real(np.sum(eigSignatures)) / np.real(np.sum(eigSamples))
        return [cpv]

    def _normalize(self, features):
        # While normalizing, we also truncate most of the decimal digits to "quantize" the single metrics and
        # mitigate the combinatorial explosion. This does NOT account for the actual distributions of the metrics
        fMax = np.amax(features, axis=0)
        fMin = np.amin(features, axis=0)
        for col in range(features.shape[1]):
            features[:, col] = (features[:, col] - fMin[col]) / (fMax[col] - fMin[col] + 0.0001)
        return features
