#ifndef CONV
#define CONV

#include "../utils.h"
#include "../tensor.cu"
#include <vector>
#include <cstring>
#include <cublas_v2.h>

const std::vector<std::vector<int>> coords = {{-1, -1}, {0, -1}, {1, -1},
                                                {-1, 0}, {0, 0}, {1, 0},
                                                {-1, 1}, {0, 1}, {1, 1}}

template <typename T>
class conv{
public:
    conv(){};
    ~conv(){};
    // Y_hat: H*W x C_out, X_hat: H*W x C_in*K*K, W_hat: C_in*K*K x C_out
    tensor<T>* forward(tensor<T>*X, tensor<T>*W){
        int n = X->shape[0], c_in = X->shape[1], h0 = X->shape[2], w0 = X->shape[3];
        int c_out = W->shape[0], k0 = 3;
        Xh = h0*w0, Xw = c_in*k0*k0;
        Wh = c_in*k0*k0, Ww = c_out;
        tensor<T>* X_hat = new tensor<T>(std::vector<int>({Xh, Xw}), "cpu");
        tensor<T>* W_hat = new tensor<T>(std::vector<int>({Wh, Ww}), "cpu");
        tensor<T>* Y_hat = new tensor<T>(std::vector<int>({Xh, Ww}), "cpu");
        for (int i=0; i<Wh; i++)
            for (int j=0; j<Ww; j++)
                W_hat->data[i*Ww + j] = W->data[j*Wh + i]
        for (int i=0; i<n; i++){
            for (int h=0; h<h0; h++)
                for (int w=0; w<w0; w++)
                    for (int c=0; c<c_in; c++)
                        {
                            int hidx = h*w0 + w;
                            T idx2value(int h1, int w1){
                                if (h1 < 0 || w1 < 0)
                                    return (T)0;
                                return X->data[i*c_in*h0*w0 + c*h0*w0 + h1*w0 + w1]
                            }
                            for (int j=0; j<9; j++){
                                int h1 = h + coords[j][1], w1 = w + coords[j][0];
                                X_hat->data[hidx*Xw + c*k0*k0 + j] = idx2value(h1, w1);
                            }
                        }
            alpha = 1.0f, beta = 0.0f;
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Xh, Ww, Xw, &alpha, X_hat->gpu().data, Xh, 
                W_hat->gpu().data, Xw, &beta, Y_hat->gpu().data, Xh);
        }
    };

    std::vector<tensor<T>*> backward(tensor<T>* dY){
        tensor<T>* dX = new tensor<T>(std::vector<int>({Xh, Xw}), "gpu");
        tensor<T>* dW = new tensor<T>(std::vector<int>({Wh, Ww}), "gpu");
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Xh, Xw, Ww, &alpha, dY->data, Xh, 
            W_hat->data, Ww, &beta, dX->data, Xh);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Wh, Ww, Xh, &alpha, X_hat->data, Wh, 
            dY->data, Xh, &beta, dW->data, Wh);
        return std::vector<tensor<T>*>({dX, dW});
    };

private:
    const float alpha, beta;
    tensor<T>*  X_hat, W_hat, Y_hat;
    int Xh, Xw, Wh, Ww;
};

#endif