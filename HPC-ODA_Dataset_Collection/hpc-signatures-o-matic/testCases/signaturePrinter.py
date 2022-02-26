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

from testCases.testCaseInterface import TestCaseInterface
from testCases.datasetBuilder import DatasetBuilder
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import logging
import numpy as np


class SignaturePrinter(TestCaseInterface):

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger('SignaturePrint')
        self.sensorsPath = "./datasets/application_classification/sensors"
        self.responsesPath = "./datasets/application_classification/responses"
        self.printPath = "./datasets/plots"
        self.window = 60
        self.step = 10
        self.tests = 0
        self.importances = False
        self.numImportances = 0
        self.printSize = 60
        self.numPrints = 10

    def runTest(self, signatureMethod, errorMethods=None):
        self._logger.info('Signature printer test case')
        self._logger.info('Signature method: %s' % signatureMethod.name)
        self._logger.info('Dataset building phase:')
        builder = DatasetBuilder()
        builder.window = self.window
        builder.step = self.step
        builder.stringResponses = True
        builder.subdirMode = False
        builder.loadDataset(self.sensorsPath, self.responsesPath, signatureMethod, errorMethods)

        isComplex = np.any(np.iscomplex(builder.features[0, :]))
        matplotlib.pyplot.switch_backend('Agg')
        self._logger.info('Printing phase:')
        for ctr in range(self.numPrints):
            sigIdx = (1+ctr)*self.printSize
            sample = builder.features[(sigIdx - self.printSize):(sigIdx + 1), :].T
            if not isComplex:
                self._makePlot(sample, sigIdx, "all")
            else:
                # Printing real and imaginary parts
                self._makePlot(sample, sigIdx, "real")
                self._makePlot(sample, sigIdx, "imag")
            self._logger.info('    Printed signature plot %s' % ctr)

    def _makePlot(self, sample, sigIdx, part):
        colorMap = "inferno_r" if part != "imag" else "viridis_r"
        toPlot = sample.real if part == "real" else sample.imag if part == "imag" else sample
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(toPlot, annot=False, ax=ax, cmap=colorMap, linewidths=0)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_xlabel("Time")
        ax.set_ylabel("Sensors")
        plt.tight_layout()
        plt.show()
        pp = PdfPages(self.printPath + '/signature_show_' + part + '_' + str(sigIdx) + '.pdf')
        pp.savefig(fig)
        pp.close()
        plt.close()
