CFLAGS = -I/usr/include/opencv2
CFLAGS += -g
CFLAGS += -std=c++14
CFLAGS += -O3

LFLAGS = -L/usr/lib/x86_64-linux-gnu
LFLAGS += -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui

.PHONY: clean

.DEFAULT_GOAL: sah

sah: main.o sah.o utils.o ostromoukhov.o
	g++ $^ -o $@ $(LFLAGS)

main.o: main.cpp sah.hpp utils.hpp
	g++ -c $(CFLAGS) $< -o $@

sah.o: sah.cpp sah.hpp utils.hpp ostromoukhov.hpp
	g++ -c $(CFLAGS) $< -o $@

ostromoukhov.o: ostromoukhov.cpp ostromoukhov.hpp
	g++ -c $(CFLAGS) $< -o $@

utils.o: utils.cpp utils.hpp
	g++ -c $(CFLAGS) $< -o $@

clean:
	-rm -rf sah main.o metric.o sah.o utils.o ostromoukhov.o
