/*-----------------------------------------------
 * 请在此处填写你的个人信息
 * 学号: SA24218133
 * 姓名: 章成胜
 * 邮箱: zhangcs66@mail.ustc.edu.cn
 ------------------------------------------------*/

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib> // 提供 atoi

#define AT(x, y, z, N) ((x) * (N) * (N) + (y) * (N) + (z))

using std::cin, std::cout, std::endl;
using std::ifstream, std::ofstream;

// CUDA核函数：模拟一个时间步
__global__ void life3d_kernel(char *universe, char *next, int N)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= N || y >= N || z >= N)
        return;

    int alive = 0;
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)
            {
                if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                int nx = (x + dx + N) % N;
                int ny = (y + dy + N) % N;
                int nz = (z + dz + N) % N;
                alive += universe[AT(nx, ny, nz, N)];
            }

    int idx = AT(x, y, z, N);
    if (universe[idx] && (alive < 5 || alive > 7))
        next[idx] = 0;
    else if (!universe[idx] && alive == 6)
        next[idx] = 1;
    else
        next[idx] = universe[idx];
}

// 核心模拟函数，将世界向前推进T个时刻（CUDA版本）
void life3d_run_cuda(int N, char *universe, int T)
{
    char *d_universe, *d_next;
    size_t size = N * N * N * sizeof(char);

    // 分配设备内存
    cudaMalloc(&d_universe, size);
    cudaMalloc(&d_next, size);
    cudaMemcpy(d_universe, universe, size, cudaMemcpyHostToDevice);

    dim3 blockDim(8, 8, 8); // 定义线程块和网格尺寸
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (N + blockDim.z - 1) / blockDim.z);

    for (int t = 0; t < T; t++)
    {
        life3d_kernel<<<gridDim, blockDim>>>(d_universe, d_next, N); // 调用核函数
        cudaMemcpy(d_universe, d_next, size, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(universe, d_universe, size, cudaMemcpyDeviceToHost); // 复制结果回主机

    // 释放设备内存
    cudaFree(d_universe);
    cudaFree(d_next);
}

void read_file(char *input_file, char *buffer)
{
    ifstream file(input_file, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        cout << "Error: Could not open file " << input_file << endl;
        exit(1);
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer, file_size))
    {
        std::cerr << "Error: Could not read file " << input_file << endl;
        exit(1);
    }
    file.close();
}

void write_file(char *output_file, char *buffer, int N)
{
    ofstream file(output_file, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        cout << "Error: Could not open file " << output_file << endl;
        exit(1);
    }
    file.write(buffer, N * N * N);
    file.close();
}

int population(int N, char *universe)
{
    int result = 0;
    for (int i = 0; i < N * N * N; i++)
        result += universe[i];
    return result;
}

int main(int argc, char **argv)
{
    if (argc < 5)
    {
        cout << "usage: ./life3d N T input output" << endl;
        return 1;
    }
    // int N = std::istringstream(argv[1]);
    // int T = std::istringstream(argv[2]);
    // int N = string_to_int(argv[1]);
    // int T = string_to_int(argv[2]);
    int N = atoi(argv[1]);
    int T = atoi(argv[2]);
    char *input_file = argv[3];
    char *output_file = argv[4];

    char *universe = (char *)malloc(N * N * N);
    read_file(input_file, universe);

    int start_pop = population(N, universe);
    auto start_time = std::chrono::high_resolution_clock::now();

    life3d_run_cuda(N, universe, T);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    int final_pop = population(N, universe);

    write_file(output_file, universe, N);

    cout << "start population: " << start_pop << endl;
    cout << "final population: " << final_pop << endl;
    double time = duration.count();
    cout << "time: " << time << "s" << endl;
    cout << "cell per sec: " << T / time * N * N * N << endl;

    free(universe);
    return 0;
}
