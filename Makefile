
CXX = g++
CXXFLAGS = -Wall -Wextra -pedantic -std=c++17 -O3 -mavx2

NVCC = nvcc
NVCCFLAGS = -Xptxas -dlcm=ca -O3 -DCUDA -gencode arch=compute_80,code=sm_80 

PYTHON = python3

$(shell mkdir -p build)


all: build/serial build/gpu build/bit


build/serial: gameoflife_1D.cpp gol_1D.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS)
build/bit: gameoflife_bit.cpp gol_bit.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS)

build/gpu: gameoflife_cuda.cpp gol.cu gol_1D.cpp
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

.PHONY: clean all

clean:
	rm -rf build/*
