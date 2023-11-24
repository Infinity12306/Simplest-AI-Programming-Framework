#ifndef FC
#define FC

#include<vector>
#include<cublas_v2.h>
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
    tensor<T>* forward(tensor<T>* W, tensor<T>* X){
        int dim_x = X->shape.size();
        if (dim_x == 1)
            X->shape.push_back(1);
    
        m = W->shape[0], k = W->shape[1], n = X->shape[dim_x - 1];
        fc_X = X, fc_W = W;

        std::vector<int> yShape = X->shape;
        yShape[dim_x - 2] = m;
        Y = new tensor<T>(yShape, "gpu");

        int iter_num = 1;
        for (int i=0; i<dim_x-2; i++)
            iter_num *= X->shape[i];
        T* X_data = X->data, Y_data = Y->data; 
        for (int i=0; i<iter_num; i++){
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, W->data, m, 
                X_data, k, &beta, Y_data, m);
            X_data += k * n;
            Y_data += m * n;
        }

        return Y->cpu();
    };
    std::vector<tensor<T>*> backward(tensor* dY){
        tensor<T>* dX = new tensor<T>(fc_X->shape, "gpu");
        tensor<T>* dW = new tensor<T>(std::vector<int>({m, k}), "gpu");
        int iternum = 1;
        for (int i=0; i<fc_X->shape.size()-2; i++)
            iternum *= fc_X->shape[i];
        T* dY_data = dY->data, dX_data = dX->data;
        for (int i=0; i<iternum; i++)
        {
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, &alpha, fc_W->data, k, 
                dY_data, m, &beta, dX_data, k);
            dX_data += k*n;
            dY_data += m*n;
        }
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &alpha, dY->data, m, 
            fc_X->data, n, &beta, dW->data, m);
        return std::vector<tensor<T>*>({dX->cpu(), dW->cpu()});
    };
private:
    tensor<T>* Y, dX, dW;
    tensor<T>* fc_X, fc_W;
    int m, n, k;
    const float alpha = 1.0f, beta = 0.0f;
};

#endif