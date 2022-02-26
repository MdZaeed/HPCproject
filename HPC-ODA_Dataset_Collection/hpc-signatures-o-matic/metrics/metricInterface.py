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


class MetricInterface(ABC):
    """

    The base abstract interface all information loss methods must comply to.

    """

    def __init__(self):
        self.name = "Information Loss Method"

    @abstractmethod
    def calculateMetric(self, samples, signatures, indexes, window=60):
        """

        Abstract method. Must be implemented by the subclass.
        Must return a list of scalars quantifying information loss.

        :param samples:     Dequeue of lists containing successive observations of the monitoring data.
        :param signatures:  Vector containing a signature computed from the input sample.
        :param indexes:     Vector of indexes mapping the computed features to the original metrics.
        :param window:      Size (in samples) of the aggregation window.
        :return:            A list of information loss metrics.

        """
        raise NotImplementedError
