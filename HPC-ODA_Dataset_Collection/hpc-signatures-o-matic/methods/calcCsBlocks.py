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

import numpy as np
import argparse

if __name__ == '__main__':

	# Configuring the input arguments to the script, and parsing them
	parser = argparse.ArgumentParser(description="CS Method Block Calculator")
	parser.add_argument("-s", action="store", dest="sensors", type=int, default=100, help="Number of sensors.")
	parser.add_argument("-b", action="store", dest="blocks", type=int, default=10, help="Number of blocks.")
	args = parser.parse_args()

	numSens = args.sensors
	numBlocks = args.blocks
	blockLen = numSens / numBlocks
	blockB = []
	blockW = []

	print("-- Number of Sensors: %s" % numSens)
	print("-- Number of Blocks: %s" % numBlocks)
	print("-- Block Length: %s" % blockLen)
	print("-- By Block:")

	for idx in range(numBlocks):
		blockStart = int(np.floor(blockLen * idx))
		blockEnd = int(np.ceil(blockLen * (idx+1)))
		blockB.append((blockStart, blockEnd))
		blockW.append(blockEnd-blockStart+1)

		print("---- Block %s" % idx)
		print("-------- Start: %s" % blockStart)
		print("-------- End: %s" % blockEnd)
		print("-------- Len: %s" % blockW[-1])

	print("\n-- Block Lengths:\n")
	print(blockW)
	print("\n-- Block Boundaries:\n")
	print(blockB)
	exit(0)
