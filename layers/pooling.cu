#ifndef POOLING
#define POOLING

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "../utils.h"
#include "../tensor.cu"

template<typename T>
__global__ void max_pool_forward(const T* in_data, T* out_data, int* mask, int c, int h, 
                                    int w, int nthreads) {
    CUDA_KERNEL_LOOP(idx, nthreads){
        int out_h = h / 2, out_w = w / 2;
        int nidx = idx / (c * out_h * out_w);
        int cidx = idx / (out_h * out_w) % c;
        int kh = idx / out_w % out_h, kw = idx % out_w;
        int x1_idx = nidx * c * h * w + cidx * h * w + kh * 2 * w + kw * 2, 
                x2_idx = x1_idx + 1;
        int x3_idx = nidx * c * h * w + cidx * h * w + (kh * 2 + 1) * w + kw * 2, 
                x4_idx = x3_idx + 1;
        T x1 = in_data[x1_idx], x2 = in_data[x2_idx], 
                x3 = in_data[x3_idx], x4 = in_data[x4_idx];
        T max_x = -1;
        if (x1 >= x2 && x1 >= x3 && x1 >= x4)
        {
            mask[x1_idx] = 1, mask[x2_idx] = 0, mask[x3_idx] = 0, mask[x4_idx] = 0;
            max_x = x1;
        }
        else if (x2 >= x1 && x2 >= x3 && x2 >= x4)
        {
            mask[x1_idx] = 0, mask[x2_idx] = 1, mask[x3_idx] = 0, mask[x4_idx] = 0;
            max_x = x2;
        }
        else if (x3 >= x1 && x3 >= x2 && x3 >= x4)
        {
            mask[x1_idx] = 0, mask[x2_idx] = 0, mask[x3_idx] = 1, mask[x4_idx] = 0;
            max_x = x3;
        }
        else if (x4 >= x1 && x4 >= x2 && x4 >= x3)
        {
            mask[x1_idx] = 0, mask[x2_idx] = 0, mask[x3_idx] = 0, mask[x4_idx] = 1;
            max_x = x4;
        }
        out_data[idx] = max_x;
    }
}

template<typename T>
__global__ void max_pool_backward(const T* dy, T*dx, int *mask, int c, int h, 
                                    int w, int nthreads) {
    CUDA_KERNEL_LOOP(idx, nthreads){
        int out_h = h /2, out_w = w / 2;
        int nidx = idx / (c * out_h * out_w);
        int cidx = idx / (out_h * out_w) % c;
        int kh = idx / out_w % out_h, kw = idx % out_w;
        int x1_idx = nidx * c * h * w + cidx * h * w + kh * 2 * w + kw * 2, 
                x2_idx = x1_idx + 1;
        int x3_idx = nidx * c * h * w + cidx * h * w + (kh * 2 + 1) * w + kw * 2, 
                x4_idx = x3_idx + 1;

        T dy_val = dy[idx];
        dx[x1_idx] = dy_val * mask[x1_idx];
        dx[x2_idx] = dy_val * mask[x2_idx];
        dx[x3_idx] = dy_val * mask[x3_idx];
        dx[x4_idx] = dy_val * mask[x4_idx];
    }
}

template<typename T>
class max_pool{
public:
    max_pool():mask(nullptr), dX(nullptr), Y(nullptr){}

    ~max_pool(){
        if (mask != nullptr)
        {
            cudaFree(mask);
            mask = nullptr;
        }
        if (dX != nullptr)
        {
            delete dX;
            dX = nullptr;
        }
        if (Y != nullptr)
        {
            delete Y;
            Y = nullptr;
        }
    }

    tensor<T>* forward(tensor<T>* X){
        n = X->shape[0], c = X->shape[1], h = X->shape[2], w = X->shape[3];
        out_h = h / 2, out_w = w / 2;
        nthreads = n * c* out_h * out_w;

        cudaMalloc(&mask, sizeof(int)*n*c*h*w);
        cudaMemset(mask, 0, sizeof(int)*n*c*h*w);

        Y = new tensor<T>(std::vector<int>{n, c, out_h, out_w}, "gpu");

        max_pool_forward<T><<<GET_BLOCK_NUM(nthreads), THREADNUM>>>(X->data, 
            Y->data, mask, c, h, w, nthreads);

        return Y;
    }

    tensor<T>* backward(tensor<T>* dY){
        dX = new tensor<T>(std::vector<int>{n, c, h, w}, "gpu");

        max_pool_backward<T><<<GET_BLOCK_NUM(nthreads), THREADNUM>>>(dY->data,
            dX->data, mask, c, h, w, nthreads);

        return dX;
    }

private:
    int n, c, h, w;
    int out_h, out_w;
    int nthreads;
    int *mask;
    tensor<T> *Y;
    tensor<T> *dX;
};

#endif