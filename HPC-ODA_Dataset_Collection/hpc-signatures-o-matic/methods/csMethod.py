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

from methods.signatureInterface import SignatureInterface
import numpy as np
import json
from threading import Event, Lock


class CSMethod(SignatureInterface):

    def __init__(self, blocks=20):
        super().__init__()
        self.blocks = blocks
        # Should be set to 1000000000 to ensure compatibility with the DCDB implementation
        self.scalingFactor = 1
        self.realOnly = False
        self.complex = False
        self.reuseMatrix = True
        self.name = "Correlation-wise Smoothing Method" if not self.realOnly else "Correlation-wise Smoothing Method (real)"
        self.sortingVec = []
        self.fMax = []
        self.fMin = []
        self._once = Event()
        self._taken = Lock()

    def clear(self):
        self.sortingVec = []
        self.fMax = []
        self.fMin = []
        self._once = Event()
        self._taken = Lock()

    def saveToFile(self, outPath):
        if not self._once.is_set():
            return False
        dataDict = {}
        for idx in range(len(self.sortingVec)):
            dataDict[idx] = {"idx": str(int(self.sortingVec[idx])), "min": str(int(self.fMin[self.sortingVec[idx]])), "max": str(int(self.fMax[self.sortingVec[idx]]))}
        dataDict = {str(len(self.sortingVec)): dataDict}
        f = open(outPath, "w")
        json.dump(dataDict, f, indent=2)

    def loadFromFile(self, inPath):
        f = open(inPath, "r")
        dataDict = json.load(f)
        dataLen = list(dataDict.keys())[0]
        self.fMax = [0] * int(dataLen)
        self.fMin = [0] * int(dataLen)
        self.sortingVec = [0] * int(dataLen)
        for k, v in dataDict[dataLen].items():
            currIdx = int(k)
            self.sortingVec[currIdx] = int(v["idx"])
            self.fMin[self.sortingVec[currIdx]] = int(v["min"])
            self.fMax[self.sortingVec[currIdx]] = int(v["max"])
        self._once.set()

    def computeSignature(self, samples, index, window=60):
        if index-window < 0:
            raise ArithmeticError

        # If we compute the correlation matrix only once on the entire sample
        if not self._once.is_set():
            if self._taken.acquire(blocking=False):
                self.sortingVec = self.correlationSortingWalk(samples)
                self.fMax = np.amax(samples, axis=1)
                self.fMin = np.amin(samples, axis=1)
                # self.loadFromFile("./results/deepest_cs_model_Full_Global_gpu.json")
                self._once.set()
                self._taken.release()
            else:
                self._once.wait()
        # If we do it at each iteration
        privateSortingVec = self.correlationSortingWalk(samples[:, (index - window + 1):(index + 1)]) if not self.reuseMatrix else self.sortingVec

        # if index < window*2:
        #     self._adjustMinMax(samples)

        # Corner case: we are at index 0 and we cannot look back for the first-order derivatives
        sortedMat = self._normalize(samples[:, (index - window):(index + 1)] if index > 0 else samples[:, (index - window + 1):(index + 1)])
        sortedMat = self._blockSmoothingReal(sortedMat[privateSortingVec, :]) if self.realOnly else self._blockSmoothing(sortedMat[privateSortingVec, :])
        return sortedMat

    def computeNames(self, metricNames):
        # Be careful! If the correlation matrix is not being re-used the order of features will change from
        # signature to signature. In this case, the names of the features returned by this method are not reliable.
        nameVec = [metricNames[idx] for idx in self.sortingVec] if self.blocks > len(metricNames) else ["signature-block" + str(idx) for idx in range(self.blocks)]
        return nameVec if self.realOnly or self.complex else [el + "-avg" for el in nameVec] + [el + "-der" for el in nameVec]

    #  Sorting of metrics based on "correlation walking"
    def correlationSortingWalk(self, samples):
        indexVec = []
        indexSet = set(range(samples.shape[0]))
        corrMat = np.nan_to_num(np.corrcoef(samples))
        corrMat[:, :] += 1
        corrVec = np.average(corrMat, axis=0)
        np.fill_diagonal(corrMat, -1)
        # First index is the one with the highest total correlation in the metric set
        indexVec.append(np.argmax(corrVec))
        indexSet.remove(indexVec[-1])
        while indexSet:
            nextIdx = np.argmax([corrMat[indexVec[-1], idx] * corrVec[idx] if idx in indexSet else -np.inf for idx in range(samples.shape[0])])
            indexVec.append(nextIdx)
            indexSet.remove(nextIdx)
        return np.asarray(indexVec)

    def _adjustMinMax(self, samples):
        newMax = np.amax(samples, axis=1)
        newMin = np.amin(samples, axis=1)
        for idx in range(samples.shape[0]):
            if self.fMin[idx] > newMin[idx]:
                self.fMin[idx] = newMin[idx]
            if self.fMax[idx] < newMax[idx]:
                self.fMax[idx] = newMax[idx]

    # Simple min-max normalization
    def _normalize(self, features):
        newSamples = np.zeros(features.shape)
        for row in range(features.shape[0]):
            newSamples[row, :] = (features[row, :] - self.fMin[row]) / (self.fMax[row] - self.fMin[row] + 0.0001)
        return newSamples.clip(0, 1)

    # Performs smoothing in blocks both on the time axis and across the sorted metrics
    def _blockSmoothing(self, samples):
        numBlocks = self.blocks if samples.shape[0] >= self.blocks else samples.shape[0]
        blockLen = samples.shape[0] / numBlocks
        signatureAvg = np.zeros(numBlocks)
        signatureDiff = np.zeros(numBlocks)
        samplesAvg = np.average(samples[:, 1:] if samples.shape[1] > 1 else samples[:, :], axis=1)
        samplesDiff = np.average(np.diff(samples[:, :], axis=1), axis=1) if samples.shape[1] > 1 else np.zeros(samples.shape[0])
        for idx in range(numBlocks):
            blockStart = int(np.floor(blockLen * idx))
            blockEnd = int(np.ceil(blockLen * (idx+1)))
            signatureAvg[idx] = np.average(samplesAvg[blockStart:blockEnd]) * self.scalingFactor
            signatureDiff[idx] = np.average(samplesDiff[blockStart:blockEnd]) * self.scalingFactor
        return np.concatenate((signatureAvg, signatureDiff)) if not self.complex else signatureAvg + signatureDiff*1j

    def _blockSmoothingReal(self, samples):
        numBlocks = self.blocks if samples.shape[0] >= self.blocks else samples.shape[0]
        blockLen = samples.shape[0] / numBlocks
        signatureAvg = np.zeros(numBlocks)
        samplesAvg = np.average(samples[:, 1:] if samples.shape[1] > 1 else samples[:, :], axis=1)
        for idx in range(numBlocks):
            blockStart = int(np.floor(blockLen * idx))
            blockEnd = int(np.ceil(blockLen * (idx+1)))
            signatureAvg[idx] = np.average(samplesAvg[blockStart:blockEnd]) * self.scalingFactor
        return signatureAvg

    # Performs a simplified version of "correlation walking" to sort the metrics
    def _correlationSortingWalkSimple(self, samples):
        indexVec = []
        indexSet = set(range(samples.shape[0]))
        corrMat = np.nan_to_num(np.corrcoef(samples))
        np.fill_diagonal(corrMat, -1000000)
        # First index is the one with the highest total correlation in the metric set
        indexVec.append(np.argmax(np.sum(corrMat, axis=0)))
        indexSet.remove(indexVec[-1])
        while indexSet:
            nextIdx = np.argmax([corrMat[indexVec[-1], idx] if idx in indexSet else -np.inf for idx in range(samples.shape[0])])
            indexVec.append(nextIdx)
            indexSet.remove(nextIdx)
        return np.asarray(indexVec)

    # Returns a vector of indexes to sort the metrics
    def _correlationSortingSum(self, samples):
        corrMat = np.nan_to_num(np.corrcoef(samples))
        return np.flip(np.argsort(np.sum(corrMat, axis=0)))

    # Linear interpolation, easy peasy
    def _interpolateSumPoints(self, vals, bound):
        lB = int(np.floor(bound))
        rB = int(np.ceil(bound))
        return np.average(vals[lB, :] + (vals[rB, :] - vals[lB, :])*(bound - lB)) if rB - bound > 1e-9 else 0
