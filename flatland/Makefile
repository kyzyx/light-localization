flatland: main.cpp cudamap.o
	g++ main.cpp cudamap.o -lcudart -lcuda -o $@ -O3

cudamap.o: cudamap.h cudamap.cu
	nvcc cudamap.cu -c -o $@

%.ppm: %.lt flatland
	./flatland 300 300 $@ -cuda < $<
