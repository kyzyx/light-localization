flatland: main.cpp
	g++ main.cpp -o $@ -O3

%.ppm: %.lt flatland
	./flatland 300 300 $@ < $<
