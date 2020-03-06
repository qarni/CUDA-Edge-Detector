################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= edgeDetector

# Cuda source files (compiled with cudacc)
CUFILES		:= canny.cu

# C/C++ source files (compiled with gcc / c++)
CCFILES		:= main.cpp

################################################################################
# Rules and targets

include ../../common/common.mk
