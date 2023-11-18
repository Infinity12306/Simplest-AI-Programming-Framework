#ifndef SIGMOID_CU
#define SIGMOID_CU

#include<cstdlib>
#include<cmath>
#include<cuda_runtime.h>
#include "../tensor.cu"
#include "../utils.h"

template<typename T>
__global__ void sigmoid_gpu(const T* in, T* out, int n){
    CUDA_KERNEL_LOOP(idx, n){
        out[idx] = 1 / (1 + std::exp(-in[idx]));
    }
}

template<typename T>
__global__ void backward_sigmoid(T* out, const GRADTYPE* grad_y, GRADTYPE* grad_x, int n){
    CUDA_KERNEL_LOOP(idx, n){
        grad_x[idx] = grad_y[idx] * out[idx] * (1 - out[idx]);
    }
}

template<typename T>
class sigmoid{
public:
    sigmoid():h_in_sig(nullptr), h_out(nullptr), grad_x(nullptr){};
    ~sigmoid(){
        delete h_out;
        delete grad_x;
    };
    tensor<T>* forward(tensor<T>* h_in){
        h_in_sig = h_in;
        h_out = new tensor<T>(h_in->shape, "gpu");
        ele_num = h_in->size / sizeof(T);
        sigmoid_gpu<T><<<GET_BLOCK_NUM(ele_num), THREADNUM>>>(
            h_in->gpu()->data, h_out->gpu()->data, ele_num);
        return h_out->cpu();
    };
    tensor<GRADTYPE>* backward(tensor<GRADTYPE>* grad_y){
        grad_x = new tensor<GRADTYPE>(grad_y->shape, "gpu");
        backward_sigmoid<T><<<GET_BLOCK_NUM(ele_num), THREADNUM>>>(
            h_out->gpu()->data, grad_y->gpu()->data, grad_x->gpu()->data, ele_num);
        return grad_x->cpu();
    };

private:
    tensor<T>* h_in_sig;
    tensor<T>* h_out;
    tensor<GRADTYPE>* grad_x;
    int ele_num;
};

#endif
