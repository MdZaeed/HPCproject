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
from scipy.stats import entropy
import numpy as np


class KullbackLeiblerDivergence(MetricInterface):

    def __init__(self):
        super().__init__()
        self.name = "Kullback-Leibler Divergence"

        self.bins = 5
        self._samplesHist = None
        self._signaturesHist = None
        self._mapping = None

    def calculateMetric(self, samples, signatures, indexes, window=60):
        signatures = np.copy(signatures)
        flattenedSamples = np.zeros((len(indexes), samples.shape[0]*window))
        for ctr, idx in enumerate(indexes):
            flattenedSamples[ctr, :] = np.reshape(samples[:, (idx-window+1):(idx+1)], samples.shape[0]*window)

        flattenedSamples = self._normalize(flattenedSamples)
        signatures = self._normalize(signatures)

        self._samplesHist = {}
        self._signaturesHist = {}
        self._mapping = {}

        # This is brutally slow
        for ctr in range(flattenedSamples.shape[0]):
            self._addToHist(flattenedSamples[ctr, :], signatures[ctr, :])

        sampleProb = []
        signatureProb = []

        for el in self._mapping.keys():
            sampleProb.append(self._samplesHist[el])
            signatureProb.append(self._signaturesHist[self._mapping[el]])

        # Mind that since sampleProb and signatureProb do not have the same support, KL divergence is unbounded
        kldiv = entropy(sampleProb, signatureProb, base=2)
        return [kldiv]

    def _normalize(self, features):
        # While normalizing, we also truncate most of the decimal digits to "quantize" the single metrics and
        # mitigate the combinatorial explosion. This does NOT account for the actual distributions of the metrics
        fMax = np.amax(features, axis=0)
        fMin = np.amin(features, axis=0)
        binWidth = 1 / self.bins
        # Using chars instead of ints to prevent Python from imploding on large datasets
        convFeatures = np.zeros(features.shape, dtype=str)
        for col in range(features.shape[1]):
            features[:, col] = (features[:, col] - fMin[col]) / (fMax[col] - fMin[col] + 0.0001)
            features[:, col] = features[:, col] // binWidth
            for idx in range(features[:, col].shape[0]):
                convFeatures[idx, col] = chr(97 + int(features[idx, col]))
        return convFeatures

    def _addToHist(self, sample, signature):
        sampleTuple = ''.join(sample)
        signatureTuple = ''.join(signature)

        # Populating hist of original features
        if sampleTuple not in self._samplesHist:
            self._samplesHist[sampleTuple] = 1
        else:
            self._samplesHist[sampleTuple] = self._samplesHist[sampleTuple] + 1

        # Populating hist of processed signatures
        if signatureTuple not in self._signaturesHist:
            self._signaturesHist[signatureTuple] = 1
        else:
            self._signaturesHist[signatureTuple] = self._signaturesHist[signatureTuple] + 1

        # Assigning mapping
        self._mapping[sampleTuple] = signatureTuple
