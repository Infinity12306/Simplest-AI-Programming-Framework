#ifndef TENSOR_CU
#define TENSOR_CU

#include <cstdlib>
#include <vector>
#include <cstring>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <fstream>

#include <cuda_runtime.h>

#include "utils.h"

template<typename T>
__global__ void transpose_k(T *new_data, const T *data, int *new_shape, int *dim_order, 
                                int *strides, int ndim, int n){
    CUDA_KERNEL_LOOP(idx, n){
        int index = 0;
        int remain = idx, ori_idx = 0;
        for (int i=ndim-1; i>=0; i--){
            index = remain % new_shape[i];
            ori_idx += strides[dim_order[i]] * index;
            remain = remain / new_shape[i];
        }
        new_data[idx] = data[ori_idx];
    }
}

template<typename T>
class tensor{
public:
    T* data;
    char* device;
    std::vector<int>shape;
    std::vector<int>strides = {1};
    int size = 1; // size of data in bytes

    tensor(const std::vector<int>shapeValue, const char* deviceValue): \
                shape(shapeValue) {
        device = (char*)malloc(4);
        strcpy(device, deviceValue);
        for (int i=0; i<shape.size(); i++){
            size *= shape[i];
            if (i > 0)
                strides.insert(strides.begin(), strides[0]*shape[shape.size()-i]);
        }
        size = size * sizeof(T);
        if (strcmp(device, "cpu") == 0){
            data = (T*)malloc(size);
            memset(data, 0, size);
        }
        else if (strcmp(device, "gpu") == 0){
            cudaMalloc(&data, size);
            cudaMemset(data, 0, size);
        }
    }

    tensor(T *data_val, const std::vector<int>shapeValue, const char* deviceValue): \
                shape(shapeValue){
        device = (char*)malloc(4);
        strcpy(device, deviceValue);
        for (int i=0; i<shape.size(); i++){
            size *= shape[i];
            if (i > 0)
                strides.insert(strides.begin(), strides[0]*shape[shape.size()-i]);
        }
        size = size * sizeof(T);
        if (strcmp(device, "cpu") == 0){
            data = (T*)malloc(size);
            memcpy(data, data_val, size);
        }
        else if (strcmp(device, "gpu") == 0){
            cudaMalloc(&data, size);
            cudaMemcpy(data, data_val, size, cudaMemcpyHostToDevice);
        }
    }

    tensor():data(nullptr), device(nullptr), shape(std::vector<int>({0})){}

    ~tensor(){
        if (device!=nullptr && strcmp(device, "cpu")==0){
            free(data);
        }
        else if (device!=nullptr && strcmp(device, "gpu") == 0){
            cudaFree(data);
        }
        free(device);
    }

    tensor<T>* cpu(){
        if (strcmp(device, "gpu") == 0){
            const char* deviceValue = "cpu";
            strcpy(device, deviceValue);
            T* new_data = (T*)malloc(size);
            if (data != nullptr)
                cudaMemcpy(new_data, data, size, cudaMemcpyDeviceToHost);
            cudaFree(data);
            data = new_data;
        }
        return this;
    }

    tensor<T>* gpu(){
        if (strcmp(device, "cpu") == 0){
            const char* deviceValue = "gpu";
            strcpy(device, deviceValue);
            T* new_data;
            cudaMalloc(&new_data, size);
            if (data != nullptr)
                cudaMemcpy(new_data, data, size, cudaMemcpyHostToDevice);
            free(data);
            data = new_data;
        }
        return this;
    }

    tensor<T>* deepcopy(){
        char* deviceValue = (char*)malloc(4);
        strcpy(deviceValue, this->device);
        tensor<T>* output = new tensor<T>(this->shape, deviceValue);
        memcpy((void*)output->data, (void*)this->data, this->size);
        return output;
    }

    tensor<T>* view(std::vector<int> new_shape){
        int new_size = sizeof(T);
        for (int i=0; i<new_shape.size(); i++)
            new_size *= new_shape[i];
        if (new_size != size){
            std::cout << "Invalid new shape" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        shape = new_shape;
        strides = std::vector<int>{1};
        for (int i=1; i<shape.size(); i++)
            strides.insert(strides.begin(), strides[0]*shape[shape.size()-i]);
        return this;
    }

    tensor<T>* transpose(std::vector<int> dim_order){
        std::vector<int> new_strides = {1};
        std::vector<int> new_shape;
        for (int i=dim_order.size()-1; i>=0; i--){
            if (std::find(dim_order.begin(), dim_order.end(), i) == dim_order.end()){
                std::cout << "Invalid transpose shape" << std::endl;
                std::exit(EXIT_FAILURE);
            }
            new_shape.insert(new_shape.begin(), shape[dim_order[i]]);
            if (i < dim_order.size()-1)
                new_strides.insert(new_strides.begin(), new_strides[0] * new_shape[1]);
        }
        T* new_data;
        int num = size / sizeof(T);

        if(strcmp(device, "cpu") == 0){
            new_data= (T*)malloc(size);
            for (int i=0; i<num; i++){
                std::vector<int> indices(new_shape.size(), 0);
                int remain = i, ori_idx = 0;
                for (int j=new_shape.size()-1; j>=0; j--){
                    indices[j] = remain % new_shape[j];
                    ori_idx += strides[dim_order[j]] * indices[j];
                    remain = remain / new_shape[j];
                }
                new_data[i] = data[ori_idx];
            }
            free(data);
            data = new_data;
        }
        else if (strcmp(device, "gpu") == 0){
            int *new_shape_gpu, *dim_order_gpu, *strides_gpu;
            cudaMalloc(&new_data, size);
            cudaMalloc(&new_shape_gpu, new_shape.size()*sizeof(int));
            cudaMalloc(&dim_order_gpu, dim_order.size()*sizeof(int));
            cudaMalloc(&strides_gpu, strides.size()*sizeof(int));
            cudaMemcpy(new_shape_gpu, new_shape.data(), new_shape.size()*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(dim_order_gpu, dim_order.data(), dim_order.size()*sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(strides_gpu, strides.data(), strides.size()*sizeof(int), cudaMemcpyHostToDevice);
            transpose_k<T><<<GET_BLOCK_NUM(num), THREADNUM>>>(new_data, data, new_shape_gpu,
                                dim_order_gpu, strides_gpu, new_shape.size(), num);
            cudaFree(data);
            data = new_data;
        }
        else{
            std::cout << "Invalid device type at tensor.transpose" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        shape = new_shape;
        strides = new_strides;
        return this;
    }

    void print_(){
        int num = size / sizeof(T);
        int r=0, c=shape[shape.size()-1];
        if (shape.size() == 1)
            r = 1;
        else
            r = shape[shape.size()-2];
        for (int i=0; i<num; i++){
            std::cout << data[i] << " ";
            if ((i+1) % c == 0)
                std::cout << std::endl;
            if ((i+1) % (r*c) == 0)
                std::cout << std::endl;
        }
    }

    void f_print(std::ofstream &f_x){
        this->cpu();
        int num = size / sizeof(T);
        // int r=0, c=shape[shape.size()-1];
        // if (shape.size() == 1)
        //     r = 1;
        // else
        //     r = shape[shape.size()-2];
        for (int i=0; i<shape.size(); i++)
            f_x << shape[i] << " ";
        f_x << std::endl;
        for (int i=0; i<num; i++){
            f_x << data[i] << " ";
        }
    }
};

#endif