# ================================================================================
# Copyright (C) 2021 Alessio Netti
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

from methods.csMethod import CSMethod
from metrics.csdivergence import CSDivergence
from testCases.deepestTemperatureTestCase import DeepestTemperatureTestCase
from testCases.deepestOpenCVTestCase import DeepestOpenCVTestCase
import logging, sys
import warnings

signatureMethods = [CSMethod(5), CSMethod(10), CSMethod(20), CSMethod(40)]
signatureStrings = ["CS5", "CS10", "CS20", "CS40"]
testCase = DeepestTemperatureTestCase()  # Use DeepestOpenCVTestCase() to train an OpenCV model instead
errorMethods = [CSDivergence()]
computeImportances = False
numTests = 1

predictionInterval = [1, 3, 6, 12]
windowInterval = [1, 3, 6, 12]
# "Full" means that we combine data from all nodes in a single model, "Node" means
# that we combine data from the same compute node, and so on
mixType = "Full"  # can also be "Node" "Socket" "None"
segment = "cn"  # can also "esb" "gpu"

dfMethods = []
dfInterval = []
dfWindows = []
dfMix = []
dfErr = []
# NRMSE results are printed on screen and written to a file
# Requires a "results" directory to be created beforehand
outFile = "./results/deepest_results_" + mixType + "_" + segment + ".csv"

# !!Attention!! If using the OpenCV version, the MaxDepth parameter of the random forest has a significant impact
# on NRMSE and on the storage required by the final model file. Here we use 10 as a compromise, which yields worse
# results compared to the scikit-learn tests, but the parameter can be tuned at will depending on storage constraints.
# See deepestOpenCVTestCase.py for more details.
# Also, please set scalingFactor in csMethod.py to 1000000000 to ensure compatibility with the DCDB implementation.
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s:%(name)s:%(asctime)s] %(message)s')
    warnings.filterwarnings('ignore')
    _logger = logging.getLogger('DeepestTempera')

    for idx, sig in enumerate(signatureMethods):
        for p in predictionInterval:
            for w in windowInterval:
                testCase.shiftResponses = p
                testCase.window = w
                testCase.mix = mixType
                testCase.segment = segment
                testCase.importances = computeImportances
                testCase.tests = numTests
                _logger.info("---- Running test with prediction %s, window %s, mix type %s, method %s" % (p, w, mixType, signatureStrings[idx]))
                res = testCase.runTest(sig, errorMethods)
                for val in res:
                    dfMethods.append(signatureStrings[idx])
                    dfInterval.append(p)
                    dfWindows.append(w)
                    dfMix.append(mixType)
                    dfErr.append(val)

    of = open(outFile, "w")
    of.write("Method,Prediction,Window,Mix,NRMSE\n")
    for idx in range(len(dfMethods)):
        ofStr = dfMethods[idx] + "," + str(dfInterval[idx]) + "," + str(dfWindows[idx]) + "," + dfMix[idx] + "," + str(dfErr[idx]) + "\n"
        of.write(ofStr)
    of.close()


