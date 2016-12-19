CPPFLAGS=-std=c++11 -O3

flatland: main.cpp cudamap.o
	g++ $(CPPFLAGS) $^ -lcudart -lcuda -o $@

visual: visual.cpp cudamap.o loadshader.o opengl_compat.o fileio.o
	g++ $(CPPFLAGS) $^ -lcudart -lcuda -lGL -lGLU -lglut -lGLEW -lpng -lIlmImf -lHalf -o $@

animate: animate.cpp fileio.o
	g++ $(CPPFLAGS) $^ -lpng -lIlmImf -lHalf -o $@

cudamap.o: cudamap.h cudamap.cu
	nvcc cudamap.cu -c -o $@

%.ppm: %.lt flatland
	./flatland 300 300 $@ $< -cuda
