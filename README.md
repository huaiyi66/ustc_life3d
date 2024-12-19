USTC 并行程序设计 CUDA实验

1. python3 RandGen.py $N
2. nvcc -std=c++17 life3d.cu -o life3d
3. life3d 256 8 data/data.in data/data.out
