#!/usr/bin/bash

N=256
T=8

python3 RandGen.py $N
# nvcc life3d.cu -o life3d
nvcc -std=c++17 life3d.cu -o life3d
# ./life3d $N $T data/data.in data/data.out
# life3d $N $T data/data.in data/data.out
life3d 256 8 data/data.in data/data.out
