# 使用 g++ 编译器
CXX = g++
CXXFLAGS = -Wall -Wextra -pedantic -std=c++17 -O3 -mavx2

NVCC = nvcc
NVCCFLAGS = -Xptxas -dlcm=ca -O3 -DCUDA -gencode arch=compute_80,code=sm_80 

PYTHON = python3

# 确保 build 目录存在
$(shell mkdir -p build)

# 默认目标
all: build/serial build/gpu build/bit

# 串行版本
build/serial: gameoflife_1D.cpp gol_1D.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS)
build/bit: gameoflife_bit.cpp gol_bit.cpp
	$(CXX) $^ -o $@ $(CXXFLAGS)
# CUDA 版本
build/gpu: gameoflife_cuda.cpp gol.cu gol_1D.cpp
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

# 清理目标
.PHONY: clean all

clean:
	rm -rf build/*
