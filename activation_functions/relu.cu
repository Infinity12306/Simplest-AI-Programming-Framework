#ifndef RELU_CU
#define RELU_CU

#include<cstdlib>
#include<cuda_runtime.h>
#include<iostream>
#include "../tensor.cu"
#include "../utils.h"

template<typename T>
__global__ void relu_gpu(const T* in, T* out, int n){
    CUDA_KERNEL_LOOP(idx, n){
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
        // printf("%f\n", in[idx]);
        out[idx] = in[idx] > 0 ? in[idx] : 0;
    }
}

template<typename T>
__global__ void backward_relu(const T* in, const GRADTYPE* grad_y, GRADTYPE* grad_x, int n){
    CUDA_KERNEL_LOOP(idx, n){
        // int idx = blockIdx.x * blockDim.x + threadIdx.x;
        grad_x[idx] = in[idx] > 0 ? grad_y[idx] : 0;
    }
}

template<typename T>
class relu{
public:
    relu():h_in_relu(nullptr), h_out(nullptr), grad_x(nullptr){};
    ~relu(){
        delete h_out;
        delete grad_x;
    };
    tensor<T>* forward(tensor<T>* h_in){
        h_in_relu = h_in;
        h_out = new tensor<T>(h_in->shape, "gpu");
        ele_num = h_in->size / sizeof(T);
        relu_gpu<T><<<GET_BLOCK_NUM(ele_num), THREADNUM>>>(
            h_in_relu->gpu()->data, h_out->gpu()->data, ele_num);
        return h_out->cpu();
    };
    tensor<GRADTYPE>* backward(tensor<GRADTYPE>* grad_y){
        grad_x = new tensor<GRADTYPE>(grad_y->shape, "gpu");
        backward_relu<T><<<GET_BLOCK_NUM(ele_num), THREADNUM>>>(
            h_in_relu->gpu()->data, grad_y->gpu()->data, grad_x->gpu()->data, ele_num);
        return grad_x->cpu();
    };

private:
    tensor<T>* h_in_relu;
    tensor<T>* h_out;
    tensor<GRADTYPE>* grad_x;
    int ele_num;
};

#endif
