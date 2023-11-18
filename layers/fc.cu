#ifndef FC
#define FC

#include <cublas_v2.h>
#include <vector>
#include "../utils.h"
#include "../tensor.cu"

template<typename T>
class fc{
public:
    fc():Y(nullptr), fc_X(nullptr), fc_W(nullptr){};
    ~fc(){
        if (Y != nullptr){
            delete Y;
            Y = nullptr;
        }
        if (dX != nullptr){
            delete dX;
            dX = nullptr;
        }
        if (dW != nullptr){
            delete dW;
            dW = nullptr;
        }
    };
    tensor<T>* forward(tensor<T>* X, tensor<T>* W){
        m = W->shape[0], k = W->shape[1], n = X->shape[1];
        fc_X = X, fc_W = W;
        Y = new tensor<T>(std::vector<int>({m, n}), "gpu");
        alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, W->data, m, 
            X->data, k, &beta, Y->data, m);
        return Y;
    };
    std::vector<tensor<T>*> backward(tensor* dY){
        tensor<T>* dX = new tensor<T>(std::vector<int>({k, n}), "gpu");
        tensor<T>* dW = new tensor<T>(std::vector<int>({m, k}), "gpu");
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, &alpha, W->data, k, 
            dY->data, m, &beta, dX->data, k);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &alpha, dY->data, m, 
            X->data, n, &beta, dW->data, m);
        return std::vector<tensor<T>*>({dX, dW});
    };
private:
    tensor<T>* Y, dX, dW;
    tensor<T>* fc_X, fc_W;
    int m, n, k;
    const float alpha, beta;
};


#endif