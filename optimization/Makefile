CPPFLAGS=-std=c++11 -O3

optimization: main.cpp solveCeres.o
	g++ $(CPPFLAGS) $^ -o $@ -lceres -lglog
