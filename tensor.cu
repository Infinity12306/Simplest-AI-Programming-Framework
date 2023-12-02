#ifndef TENSOR_CU
#define TENSOR_CU

#include<cstdlib>
#include<cuda_runtime.h>
#include<vector>
#include<cstring>
#include<cassert>
#include<iostream>

template<typename T>
class tensor{
public:
    T* data;
    char* device;
    std::vector<int>shape;
    int size = 1; // size of data in bytes
    tensor(const std::vector<int>shapeValue, const char* deviceValue): \
                shape(shapeValue) {
        device = (char*)malloc(4);
        strcpy(device, deviceValue);
        for (int i=0; i<shape.size(); i++){
            size *= shape[i];
        }
        size = size * sizeof(T);
        if (strcmp(device, "cpu") == 0){
            data = (T*)malloc(size);
            memset(data, 0, sizeof(T)*size);
        }
        else if (strcmp(device, "gpu") == 0){
            cudaMalloc(&data, size);
            cudaMemset(data, 0, sizeof(T)*size);
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
};

#endif