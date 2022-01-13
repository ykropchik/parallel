#!/usr/bin/bash

g++-11 -fopenmp --std=c++20 main.cpp threads.cpp
./a.out