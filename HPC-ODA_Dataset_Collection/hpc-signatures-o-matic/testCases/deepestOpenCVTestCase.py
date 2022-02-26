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

from testCases.deepestTemperatureTestCase import DeepestTemperatureTestCase
import numpy as np
import cv2
import logging


class DeepestOpenCVTestCase(DeepestTemperatureTestCase):

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger('DeepestOpenCVT')

        # self.errFileCV = open("err_opencv.csv", "a+")

    def _trainAndSaveOpenCV(self, outPath, samples, responses, split=100):
        model = cv2.ml.RTrees_create()

        term_type = cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS
        n_trees = 50
        epsilon = 0.001
        model.setTermCriteria((term_type, n_trees, epsilon))
        model.setCalculateVarImportance(True)
        model.setRegressionAccuracy(0.001)
        model.setActiveVarCount(samples.shape[1])
        model.setMinSampleCount(10)
        # Max tree depth affects the NRMSE and storage required for the model's file
        # We use 10 as an example, can be increased or decreased depending on requirements
        model.setMaxDepth(10)
        # model.setCVFolds(0)
        # model.setUse1SERule(False)

        rangeTraining = [0, int(samples.shape[0]*(split/100))]
        rangeTesting = [rangeTraining[1] + 1, -1]

        self._logger.info("Starting training of OpenCV model...")
        samples_t = np.float32(samples[rangeTraining[0]:rangeTraining[1], :])
        responses_t = np.float32(responses[rangeTraining[0]:rangeTraining[1]])
        train_data = cv2.ml.TrainData_create(samples=samples_t, layout=cv2.ml.ROW_SAMPLE, responses=responses_t)
        model.train(trainData=train_data)
        self._logger.info("Completed training of OpenCV model.")

        if rangeTesting[0] < samples.shape[0]:
            self._logger.info("Starting evaluation of OpenCV model.")
            samples_te = np.float32(samples[rangeTesting[0]:rangeTesting[1], :])
            responses_te = np.float32(responses[rangeTesting[0]:rangeTesting[1]])
            _ret, testPred = model.predict(samples_te)
            err = self._nrmse(responses_te, testPred)
            # for idx in range(len(responses_te)):
            #     self.errFileCV.write(str(responses_te[idx]) + "," + str(testPred[idx]) + "\n")
            self._logger.info('Normalized root mean squared error of OpenCV model : %s' % err)
        model.save(outPath)
