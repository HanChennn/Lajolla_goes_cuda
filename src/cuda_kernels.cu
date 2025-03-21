// cuda_kernels.cu
#include <cuda_runtime.h>
#include <iostream>

// CUDA 核函数
__global__ void addKernel(int *d_a, int *d_b, int *d_c, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}

// C++ 可调用的封装函数
extern "C" void launchAddKernel(int *h_a, int *h_b, int *h_c, int size) {
    int *d_a, *d_b, *d_c;
    size_t bytes = size * sizeof(int);

    // 分配 GPU 内存
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    // 复制数据到 GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 计算线程块配置
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // 启动 CUDA 核函数
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

    // 复制结果回 CPU
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
