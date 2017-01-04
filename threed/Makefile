CPPFLAGS=-std=c++11 -O3

threed: visual.cpp cudamap.o loadshader.o opengl_compat.o fileio.o
	g++ $(CPPFLAGS) $^ -lcudart -lcuda -lGL -lGLU -lglut -lGLEW -lpng -lIlmImf -lHalf -o $@

cudamap.o: cudamap.h cudamap.cu
	nvcc cudamap.cu -c -o $@
