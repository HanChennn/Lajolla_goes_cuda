// cuda_kernels.h
#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

void launchAddKernel(int *h_a, int *h_b, int *h_c, int size);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_H
