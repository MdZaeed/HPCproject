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
from testCases.scalabilityTestCase import ScalabilityTestCase
import logging, sys
import warnings

signatureMethods = [TuncerMethod(), BodikMethod(), LanMethod(), CSMethod(5), CSMethod(10), CSMethod(20), CSMethod(40), CSMethod(100000)]

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s:%(name)s:%(asctime)s] %(message)s')
    warnings.filterwarnings('ignore')

    ts = ScalabilityTestCase()
    ts.windowWise = True
    for sig in signatureMethods:
        ts.runTest(sig)
