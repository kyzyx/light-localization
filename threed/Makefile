CPPFLAGS=-std=c++11 -O3 -fopenmp -I/usr/local/cuda/include -L/usr/local/cuda/lib64

threed: visual.cpp cudamap.o loadshader.o opengl_compat.o fileio.o mesh.o planemanager.o trackball.o extract.o
	g++ $(CPPFLAGS) $^ -lcudart -lcuda -lGL -lGLU -lglut -lGLEW -lpng -lIlmImf -lHalf -o $@

cudamap.o: cudamap.h cudamap.cu
	nvcc cudamap.cu -c -o $@

clean:
	rm -f *.o
	rm -f threed
