#!/usr/bin/bash

N=256
T=8

python3 RandGen.py $N
# nvcc life3d.cu -o life3d
nvcc -std=c++17 life3d.cu -o life3d
# ./life3d $N $T data/data.in data/data.out
# life3d $N $T data/data.in data/data.out
life3d 256 8 data/data.in data/data.out


CPU:
nvcc -std=c++17 life3d.cu -o life3d
life3d 256 8 data/data.in data/data.out
life3d 256 16 data/data.in data/data.out
life3d 512 8 data/data.in data/data.out

start population: 3856918
final population: 2642387
time: 15.4882s
cell per sec: 8.66581e+06

CUDA:
nvcc -std=c++17 life3d_cuda.cu -o life3d_cuda
life3d_cuda 256 8 data/data.in data/data.out
life3d_cuda 512 8 data/data.in data/data.out

start population: 3859603
final population: 2631411
time: 0.425274s
cell per sec: 3.15603e+08

life3d_cuda 256 16 data/data.in data/data.out

start population: 3859603
final population: 1386144
time: 0.38979s
cell per sec: 6.88666e+08


CUDA 进一步优化:
nvcc -std=c++17 life3d_cuda_v2.cu -o life3d_cuda_v2
life3d_cuda_v2 256 8 data/data.in data/data.out
life3d_cuda_v2 512 8 data/data.in data/data.out
life3d_cuda_v2 256 16 data/data.in data/data.out