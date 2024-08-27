# Compiler
CXX := nvcc

# Source file
SRC := test.cu

# Output executable
EXEC := test.exe

# Compiler flags
CXXFLAGS := --std c++17 -Wno-deprecated-gpu-targets -g -O3

# Include paths
INCLUDES := -I/usr/local/cuda/include

# Libraries
LIBRARIES := -lcuda -lcudnn -lcurand -lcudart -lcublas

# Library paths
LIB_PATHS := -L/usr/local/cuda/lib64

# Default input file (change this to the actual path of your input file)
DEFAULT_INPUT := ./WikiText-2---CUDA/wikitext-2/train.csv

# Default target
all: clean build

build: $(SRC)
	$(CXX) $(SRC) $(CXXFLAGS) $(INCLUDES) $(LIB_PATHS) -o $(EXEC) $(LIBRARIES)

run:
	./$(EXEC) $(if $(INPUT),$(INPUT),$(DEFAULT_INPUT))

clean:
	rm -f $(EXEC) output*.txt *.bin

.PHONY: all build run clean