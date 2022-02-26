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

from abc import abstractmethod, ABC


class SignatureInterface(ABC):
    """

    The base abstract interface all signature methods must comply to.

    """

    def __init__(self):
        self.name = "Signature Method"

    """
    
    Re-initializes the signatures object
    
    """
    def clear(self):
        pass

    @abstractmethod
    def computeSignature(self, samples, index, window=60):
        """

        Abstract method. Must be implemented by the subclass.
        Must return a list containing the computed signature.

        :param samples: Dequeue of lists containing successive observations of the monitoring data.
        :param index:   Head index from which to start computing the signature in the dequeue.
        :param window:  Size (in samples) of the aggregation window.
        :return:        A list containing the computed signature.

        """
        raise NotImplementedError

    @abstractmethod
    def computeNames(self, metricNames):
        """

        Abstract method. Must be implemented by the subclass.
        Must return a list of the string identifiers for each element in computed signatures.

        :param metricNames: The list of input sensor names.
        :return:            The resulting list of identifiers for the signature.
        """
        raise NotImplementedError