#pragma once
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

constexpr int TILE_WIDTH = 16;

struct DeviceMemTracker {
    static inline std::vector<void*> device_mem_ptrs;

    static void log(void* ptr) {
        device_mem_ptrs.push_back(ptr);
    }

    static void print() {
        std::cout << "Allocated CUArrays:\n";
        for (auto p : device_mem_ptrs) {
            std::cout << "  " << p << std::endl;
        }
    }

    static void free() {
        for (auto p : device_mem_ptrs) {
            cudaFree(p);
        }
    }
};

template <typename T>
struct CUArray {
    T* __data = nullptr;
    size_t __size = 0;

    CUArray() = default;

    __host__ CUArray(const CUArray& other)
        : __size(other.__size)
    {
        cudaMalloc((void **)&__data, other.size() * sizeof(T));
        cudaMemcpy(__data, other.data(), other.size() * sizeof(T), cudaMemcpyDeviceToDevice);
        DeviceMemTracker::log(__data);
    }

    __host__ CUArray& operator=(const CUArray& other) {
        if (this != &other) {
            __size = other.__size;
            cudaMalloc((void **)&__data, other.size() * sizeof(T));
            cudaMemcpy(__data, other.data(), other.size() * sizeof(T), cudaMemcpyDeviceToDevice);
            DeviceMemTracker::log(__data);
        }
        return *this;
    }

    __host__ CUArray(CUArray&& other) noexcept
        : __data(other.__data), __size(other.__size)
    {
        other.__data = nullptr;
        other.__size = 0;
    }

    __host__ CUArray& operator=(CUArray&& other) noexcept {
        if (this != &other) {
            __data = other.__data;
            __size = other.__size;
            other.__data = nullptr;
            other.__size = 0;
        }
        return *this;
    }

    __host__ void init(const std::vector<T>& v){
        __size = v.size();
        cudaMalloc((void **)&__data, v.size() * sizeof(T));
        cudaMemcpy(__data, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
        DeviceMemTracker::log(__data);
    }

    __host__ __device__ inline T* data() {
        return __data;
    }

    __host__ __device__ inline const T* data() const {
        return __data;
    }

    __host__ __device__ inline size_t size() const {
        return __size;
    }

    __host__ __device__ inline T& operator[](const size_t index) {
        return __data[index];
    }

    __host__ __device__ inline const T& operator[](const size_t index) const {
        return __data[index];
    }
};