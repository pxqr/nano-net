all: program

optimizations = -O3 -march=native -mtune=native

program: main.o 
	g++ $(optimizations) main.o -o program -std=c++0x

main.o:  main.cpp
	g++ $(optimizations) -c main.cpp
