#ifndef POOLING
#define POOLING

#include<cuda_runtime.h>
#include<vector>

template <typename T>
__global__ void max_pool_forward(T* in_data, int num, int channels, 
        int in_h, int in_w, int out_h, int out_w, float* out_data, float* out_mask) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = index / out_w / out_h / channels; 
    int c = (index / out_w / out_h) % channels; 
    int ph = (index / out_w) % out_h;
    int pw = index % out_w;
    std::vector<T> values = {};
    values.push_back(in_data[n*channels*in_h*in_w + c*in_h*in_w + ph*2*in_w + pw*2]);
    values.push_back(in_data[n*channels*in_h*in_w + c*in_h*in_w + ph*2*in_w + pw*2 + 1]);
    values.push_back(in_data[n*channels*in_h*in_w + c*in_h*in_w + (ph*2+1)*in_w + pw*2]);
    values.push_back(in_data[n*channels*in_h*in_w + c*in_h*in_w + (ph*2+1)*in_w + pw*2 + 1]);
    out_data[n*channels*out_h*out_w + c*out_h*out_w + ph*out_w + pw] = values.max();
}

#endif