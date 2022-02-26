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
from methods.signatureInterface import SignatureInterface
from metrics.metricInterface import MetricInterface


class TestCaseInterface(ABC):
    """

    The base abstract interface all test case implementations must comply to.

    """

    def __init__(self):
        self.sensorsPath = "./"
        self.responsesPath = "./"
        self.tests = 10

    @abstractmethod
    def runTest(self, signatureMethod, errorMethods=None):
        """

        Abstract method. Must be implemented by the subclass.
        Runs an instance of the implemented test scenario.

        :param signatureMethod: Object of type SignatureInterface, with which signatures will be computed.
        :param errorMethods:    Optional. List of objects implementing MetricInterface, to quantify information loss.

        """
        if not isinstance(signatureMethod, SignatureInterface):
            raise TypeError
        if errorMethods is not None and not all(isinstance(err, MetricInterface) for err in errorMethods):
            raise TypeError
        raise NotImplementedError
