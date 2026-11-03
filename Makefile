NVCC ?= nvcc
CXX ?= g++
CPPFLAGS += -I.
CXXFLAGS ?= -O2 -std=c++17 -ffp-contract=off
NVCCFLAGS ?= -O2 -std=c++17 --fmad=false
TARGET := build/hough

.PHONY: all clean
all: $(TARGET) build/hough_const build/hough_shared

$(TARGET): src/houghBase.cu common/pgm.h build/pgm.o | build
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $< build/pgm.o -o $@

build/hough_const: src/houghConst.cu common/pgm.h build/pgm.o | build
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $< build/pgm.o -o $@

build/hough_shared: src/houghShared.cu common/pgm.h build/pgm.o | build
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $< build/pgm.o -o $@

build/pgm.o: common/pgm.cpp common/pgm.h | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

build:
	mkdir -p $@

clean:
	rm -rf build

# Comprobación local sin CUDA; no valida el kernel GPU.
.PHONY: cpu-check
cpu-check: build/hough_cpu
	./build/hough_cpu data/runway.pgm

build/hough_cpu: src/houghBase.cu common/pgm.cpp common/pgm.h | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DHOUGH_CPU_ONLY -x c++ src/houghBase.cu common/pgm.cpp -o $@

.PHONY: colab-package
colab-package: | build
	python3 -c "import zipfile; from pathlib import Path; files = [Path('Makefile'), *Path('src').glob('*.cu'), *Path('common').glob('*.h'), *Path('common').glob('*.cpp'), *Path('data').glob('*.pgm')]; z = zipfile.ZipFile('build/proyecto-colab.zip', 'w', zipfile.ZIP_DEFLATED); [z.write(p, p.as_posix()) for p in files]; z.close()"
