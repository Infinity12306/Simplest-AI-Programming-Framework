// Cross entropy with softmax
#ifndef CROSSENTROPY
#define CROSSENTROPY

#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include "../utils.h"
#include "../tensor.cu"

template<typename T>
__global__ void cross_entropy_forward(const T* x, T* p, T* loss, T* loss_val, int *labels, 
                                        int c, int nthreads)
{
    CUDA_KERNEL_LOOP(idx, nthreads){
        const int offset = idx * c;
        T max_val = x[offset];
        for (int i=1; i<c; i++)
        {
            T cur_val = x[offset + i];
            if (cur_val > max_val)
                max_val = cur_val;
        }
        T exp_sum = 0;
        for (int i=0; i<c; i++)
        {
            T exp_val = exp(x[offset + i] - max_val);
            p[offset + i] = exp_val;
            exp_sum += exp_val;
        }
        for (int i=0; i<c; i++)
            p[offset + i] /= exp_sum;
        T sample_loss = log(exp_sum) - (x[offset + labels[idx]] - max_val);
        loss[idx] = sample_loss;
        atomicAdd(loss_val, sample_loss);
    }
}

template<typename T>
__global__ void cross_entropy_backward(T* dx, T* p, int *labels, int c, int nthreads)
{
    CUDA_KERNEL_LOOP(idx, nthreads){
        const int offset = idx * c;
        int label = labels[idx];
        for (int i=0; i<c; i++)
            dx[offset + i] = (p[offset + i] - (i == label ? 1 : 0)) / nthreads;
    }
}

template<typename T>
class cross_entropy{
public:
    cross_entropy():p(nullptr), loss(nullptr), dx(nullptr), loss_val(nullptr), labels(nullptr){}

    ~cross_entropy(){
        if (p != nullptr)
            delete p, p = nullptr;
        if (loss != nullptr)
            delete loss, loss = nullptr;
        if (dx != nullptr)
            delete dx, dx = nullptr;
        delete loss_val, loss_val = nullptr;
        delete labels, labels = nullptr;
    }

    T forward(tensor<T>* X, tensor<int>* labels_){
        n = X->shape[0], c = X->shape[1];
        p = new tensor<T>(X->shape, "gpu");
        loss = new tensor<T>(std::vector<int>{n}, "gpu");
        loss_val = new tensor<T>(std::vector<int>{1}, "gpu");
        labels = labels_->deepcopy();
        cross_entropy_forward<T><<<GET_BLOCK_NUM(n), THREADNUM>>>(X->data, p->data, loss->data, 
            loss_val->data, labels->data, c, n);
        return *loss_val->cpu()->data / n;
    }

    tensor<T>* backward(){
        dx = new tensor<T>(std::vector<int>{n, c}, "gpu");
        cross_entropy_backward<T><<<GET_BLOCK_NUM(n), THREADNUM>>>(dx->data, p->data, labels->data, c, n);
        return dx;
    }

private:
    tensor<T> *p;
    tensor<T> *loss, *loss_val;
    tensor<T> *dx;
    tensor<int> *labels; 
    int n, c;
};

#endif