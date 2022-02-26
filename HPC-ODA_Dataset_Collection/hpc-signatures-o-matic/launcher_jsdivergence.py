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

from methods.csMethod import CSMethod
from metrics.csdivergence import CSDivergence
from testCases.anomalyTestCase import AnomalyDetectionTestCase
from testCases.applicationTestCase import ApplicationClassificationTestCase
from testCases.powerTestCase import PowerPredictionTestCase
from testCases.infrastructureTestCase import InfrastructureManagementTestCase
import logging, sys
import warnings

# Remember to reduce the step value between feature sets in the source file of each test case to obtain
# reliable JS numbers. Suggested step values are 5, 1, 2 and 1 for the four test cases in the order above
# Also, remember to keep the "complex" attribute in CSMethod to true
signatureMethods = [CSMethod(5), CSMethod(10), CSMethod(20), CSMethod(40), CSMethod(200)]
testCases = [AnomalyDetectionTestCase(), PowerPredictionTestCase(), ApplicationClassificationTestCase(), InfrastructureManagementTestCase()]
errorMethods = [CSDivergence()]
computeImportances = False
numTests = 0

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s:%(name)s:%(asctime)s] %(message)s')
    warnings.filterwarnings('ignore')

    for sig in signatureMethods:
        # Forcing the "complex" attribute to true
        sig.complex = True
        for ts in testCases:
            ts.importances = computeImportances
            ts.tests = numTests
            errorMethods[0].method = sig
            errorMethods[0].reNormalize = True
            ts.runTest(sig, errorMethods)
