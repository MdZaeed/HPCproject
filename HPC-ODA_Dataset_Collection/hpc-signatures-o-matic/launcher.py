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

from methods.tuncerMethod import TuncerMethod
from methods.bodikMethod import BodikMethod
from methods.lanMethod import LanMethod
from methods.csMethod import CSMethod
from metrics.jensenshannondivergence import JensenShannonDivergence
from testCases.anomalyTestCase import AnomalyDetectionTestCase
from testCases.applicationTestCase import ApplicationClassificationTestCase
from testCases.crossArchitectureTestCase import CrossArchitectureTestCase
from testCases.powerTestCase import PowerPredictionTestCase
from testCases.infrastructureTestCase import InfrastructureManagementTestCase
from testCases.signaturePrinter import SignaturePrinter
import logging, sys
import warnings

signatureMethods = [TuncerMethod(), BodikMethod(), LanMethod(), CSMethod()]
testCases = [ApplicationClassificationTestCase()]
errorMethods = [JensenShannonDivergence()]
computeImportances = True
numTests = 1

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s:%(name)s:%(asctime)s] %(message)s')
    warnings.filterwarnings('ignore')

    for sig in signatureMethods:
        for ts in testCases:
            ts.importances = computeImportances
            ts.tests = numTests
            ts.runTest(sig, errorMethods)
