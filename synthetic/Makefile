CC=g++
CPPFLAGS=-std=c++11 -I../GAPS/
LDFLAGS=-L../GAPS/R3Graphics/ -L../GAPS/R3Shapes/ -L../GAPS/R2Shapes/ -L../GAPS/RNBasics/ -lR3Graphics -lR3Shapes -lR2Shapes -lRNBasics -ljpeg -lGL -lGLU
all: gensynthetic

gensynthetic: synthetic.o
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^ $(LDFLAGS) 
