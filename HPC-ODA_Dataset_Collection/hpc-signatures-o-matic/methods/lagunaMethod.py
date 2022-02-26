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


class LagunaMethod(SignatureInterface):

    def __init__(self):
        super().__init__()
        self.name = "Laguna Method"

    def computeSignature(self, samples, index, window=60):
        if index-window < 0:
            raise ArithmeticError
        corrMat = np.corrcoef(samples[:, (index-window+1):(index+1)])
        corrMat[corrMat == np.inf] = 0
        corrMat[corrMat == np.NINF] = 0
        corrMat = np.nan_to_num(corrMat)
        return corrMat[np.triu_indices(corrMat.shape[0], k=1)]

    def computeNames(self, metricNames):
        names = []
        for idx in range(len(metricNames)-1):
            mName = metricNames[idx].replace(".csv", "")
            names += [mName + "_X_" + el.replace(".csv", "") for el in metricNames[(idx+1):]]
        return names
