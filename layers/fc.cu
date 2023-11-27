#ifndef FC
#define FC

#include<vector>
#include<cublas_v2.h>
#include "../utils.h"
#include "../tensor.cu"

template<typename T>
class fc{
public:
    fc():Y(nullptr), dX(nullptr), dW(nullptr){
        cublasCreate(&handle);
    };
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
        cublasDestroy(handle);
    };
    tensor<T>* forward(tensor<T>* W, tensor<T>* X){
        // calculate m, k, n
        // W: (m, k), X: ((bsize,) k (, n))
        W = W->gpu(), X = X->gpu();
        int dim_x = X->shape.size();
        if (dim_x == 1){
            X->shape.push_back(1);
            dim_x++;
        }
        m = W->shape[0], k = W->shape[1], n = X->shape[dim_x - 1];
        fc_X = X, fc_W = W;

        // initialize Y
        std::vector<int> yShape = X->shape;
        yShape[dim_x - 2] = m;
        Y = new tensor<T>(yShape, "gpu");

        // calculate bsize
        int iter_num = 1;
        for (int i=0; i<dim_x-2; i++)
            iter_num *= X->shape[i];
        // calculate Y
        T *X_data = X->gpu()->data, *Y_data = Y->gpu()->data; 
        for (int i=0; i<iter_num; i++){
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, W->gpu()->data, m, 
                X_data, k, &beta, Y_data, m);
            X_data += k * n;
            Y_data += m * n;
        }
        return Y->cpu();
    };
    std::vector<tensor<T>*> backward(tensor<T>* dY){
        dY = dY->gpu();
        tensor<T>* dX = new tensor<T>(fc_X->shape, "gpu");
        tensor<T>* dW = new tensor<T>(std::vector<int>({m, k}), "gpu");
        // calculate batch size
        int iternum = 1;
        for (int i=0; i<fc_X->shape.size()-2; i++)
            iternum *= fc_X->shape[i];

        // calculate dL/dX
        // size = (bsize, k, n)
        T *dY_data = dY->gpu()->data, *dX_data = dX->gpu()->data;
        for (int i=0; i<iternum; i++)
        {
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, k, n, m, &alpha, fc_W->gpu()->data, m, 
                dY_data, m, &beta, dX_data, k);
            dX_data += k*n;
            dY_data += m*n;
        }

        // calculate dL/dW
        // size = (m, k)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &alpha, dY->data, m, 
            fc_X->gpu()->data, k, &beta, dW->data, m);
        T *X_data = fc_X->data;
        dY_data = dY->data;
        float alpha_w = 1.0f, beta_w = 1.0f;
        for (int i=1; i<iternum; i++)
        {
            X_data += k*n, dY_data += m*n;
            alpha_w = 1.0f / (i+1);
            beta_w = (float)i / (i+1);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, k, n, &alpha_w, dY_data, m, 
                X_data, k, &beta_w, dW->data, m);
        }
        return std::vector<tensor<T>*>({dW->cpu(), dX->cpu()});
    };
private:
    tensor<T> *Y, *dX, *dW;
    tensor<T> *fc_X, *fc_W;
    int m, n, k;
    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
};

#endif