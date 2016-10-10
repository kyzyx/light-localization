flatland: main.cpp cudamap.o
	g++ $^ -lcudart -lcuda -o $@ -O3

visual: visual.cpp cudamap.o loadshader.o opengl_compat.o
	g++ $^ -lcudart -lcuda -lGL -lGLU -lglut -lGLEW -o $@ -g

cudamap.o: cudamap.h cudamap.cu
	nvcc cudamap.cu -c -o $@

%.ppm: %.lt flatland
	./flatland 300 300 $@ -cuda < $<
