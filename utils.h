#ifndef UTILS_H
#define UTILS_H

const int THREADNUM = 512;
#define GRADTYPE float


inline int GET_BLOCK_NUM(int n){
    return (n + THREADNUM - 1) / THREADNUM;
}
#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

#include <cublas_v2.h>
#include <iostream>

#include "tensor.cu"

template<typename T>
void sgemm_wrapper(cublasHandle_t& handle, T* X, T* W, T* Y, int yr, int yc, 
                    int opxc, float alpha=1, float beta=0, bool tx=false, bool tw=false){
    if (!tx && !tw){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, yc, yr, opxc, &alpha, W, yc, 
                X, opxc, &beta, Y, yc);}
    else if (!tx && tw)
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, yc, yr, opxc, &alpha, W, opxc, 
                X, opxc, &beta, Y, yc);
    else if (tx && !tw){
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, yc, yr, opxc, &alpha, W, yc, 
                X, yr, &beta, Y, yc);}
    else if (tx && tw)
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, yc, yr, opxc, &alpha, W, opxc, 
                X, yr, &beta, Y, yc);
}

template<typename T>
void print_vector(std::vector<T> x){
    for (int i=0; i<x.size(); i++)
        std::cout << x[i] << " ";
    std::cout << std::endl;
    return;
}

#endif
