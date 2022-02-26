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

from threading import Thread
import numpy as np


class DatasetThread(Thread):

    def __init__(self, **kwargs):
        super().__init__()

        self._signatureMethod = kwargs.get('signatureMethod')
        self._inputSensors = kwargs.get('inputSensors')
        self._inputResponses = kwargs.get('inputResponses')
        self._indexes = kwargs.get('indexes', 0)
        self._window = kwargs.get('window', 60)

        self.features = None
        self.responses = None

    def run(self):
        self.features = None
        self.responses = None

        for idx in self._indexes:
            currFeature = self._signatureMethod.computeSignature(self._inputSensors, idx, self._window)
            currResponse = self._inputResponses[idx]
            self.features = np.vstack((self.features, currFeature)) if self.features is not None else currFeature
            self.responses = np.append(self.responses, currResponse) if self.responses is not None else np.asarray(currResponse, dtype='object')
