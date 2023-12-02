#ifndef CONV
#define CONV

#include "../utils.h"
#include "../tensor.cu"
#include <vector>
#include <cstring>
#include <random>
#include <cublas_v2.h>

const std::vector<std::vector<int>> coords = {{-1, -1}, {0, -1}, {1, -1},
                                                {-1, 0}, {0, 0}, {1, 0},
                                                {-1, 1}, {0, 1}, {1, 1}}

template <typename T>
class conv{
public:
    tensor<T> *W;

    conv(int c_in, int c_out):X_hat(nullptr), W(nullptr), Y_hat(nullptr), 
            dX(nullptr), dW(nullptr), c_in(c_in), c_out(c_out){
        cublasCreate(&handle);
        Wh = c_out, Ww = c_in*k*k;
        W = new tensor<T>(std::vector<int>({Wh, Ww}), "cpu");
        dW = new tensor<T>(std::vector<int>({Wh, Ww}), "gpu");

        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
        for (int j=0; j<Ww; j++)
            for (int i=0; i<Wh; i++)
                W->data[j*Wh + i] = distribution(generator);
    };
    ~conv(){
        if (X_hat != nullptr)
            delete X_hat;
        if (Y_hat != nullptr)
            delete Y_hat;
        if (W != nullptr)
            delete W;
        if (dX != nullptr)
            delete dX;
        if (dW != nullptr)
            delete dW;
        cublasDestroy(handle);
    };
    // Y_hat: C_out x H*W, X_hat: H*W x C_in*K*K, W: C_out X C_in*K*K
    // Y_hat = W * X_hat^T
    tensor<T>* forward(tensor<T>*X){
        n = X->shape[0], c_in = X->shape[1], h0 = X->shape[2], w0 = X->shape[3];
        Xh = h0*w0, Xw = c_in*k*k;
        int k_size = k*k;

        X_hat = new tensor<T>(std::vector<int>({n, Xh, Xw}), "cpu");
        Y_hat = new tensor<T>(std::vector<int>({n, Wh, Xh}), "gpu");

        forward_X_hat(X->cpu()->data, X_hat->cpu()->data)

        T *X_data = X_hat->gpu().data;
        T *Y_data = Y_hat->gpu().data;
        W->gpu();
        for (int i=0; i<n; i++){
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Wh, Xh, Ww, &alpha, W.data, Wh, 
                X_data, Xh, &beta, Y_data, Wh);
            X_data += Xh*Xw, Y_data += Wh*Xh;
        }
        return Y_hat;
    };

    std::vector<tensor<T>*> backward(tensor<T>* dY){
        dX = new tensor<T>(std::vector<int>({n, c_in, h0, w0}), "cpu");
        tensor<T> tmp_dX = tensor<T>(std::vector<int>({Xh, Xw}, "gpu"))

        dY->gpu(), X_hat->gpu();
        dX->cpu();
        T *dY_data = dY->data, *X_hat_data = X_hat->data, *dX_data = dX->data;
        float alpha_w, beta_w;
        int chw = c_in * h0 * w0;

        for (int i=0; i<n; i++)
        {
            tmp_dX.gpu();
            alpha_w = 1.0 / (float)(i+1), beta_w = (float)i / (float)(i+1);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Wh, Ww, Xh, &alpha_w, dY_data, Wh, 
                X_hat_data, Xh, &beta_w, dW->data, Wh);
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Xh, Xw, Wh, &alpha, dY_data, Xh, 
                W->data, Ww, &beta, tmp_dX.data, Xh);
            backward_dX_hat(tmp_dX.cpu()->data, dX_data);
            dY_data += Wh * Xh, X_hat_data += Xh * Xw, dX_data += chw;
        }
        return std::vector<tensor<T>*>({dX, dW});
    };

    // Xh: h0*w0, Xw: c_in*k*k
    // construct X_hat from X
    void forward_X_hat(T*X, T*X_hat){
        int chw = c_in*h0*w0, hw = h0*w0;
        for (int i=0; i<n; i++){
            for (int h=0; h<h0; h++)
                for (int w=0; w<w0; w++)
                    for (int c=0; c<c_in; c++)
                        for (int j=0; j<ksize; j++){
                            int hidx = h*w0 + w, widx = c*ksize + j; // h, w in X_hat
                            int cur_idx = i*(Xh * Xw) + widx*Xh + hidx;

                            int h1 = h + coords[j][1], w1 = w + coords[j][0]; // h, w in X
                            int ori_idx = i*chw + c*hw + w1*h0 + h1;

                            if (h1 < 0 || w1 < 0 || h1 >= h0 || w1 >= w0)
                                X_hat->data[cur_idx] = (T)0;
                            else
                                X_hat->data[cur_idx] = X->data[ori_idx];
                        }
    }

    // Xh: h0*w0, Xw: c_in*k*k
    // backpropagate gradient from X_hat to X
    void backward_dX_hat(T*tmp_dX, T* dX){
        int ksize = k * k, hw = h0 * w0;
        for (int h=0; h<h0; h++)
            for (int w=0; w<w0; w++)
                for (int c=0; c<c_in; c++)
                    for (int j=0; j<ksize; j++){
                        int h1 = h + coords[j][1], w1 = w + coords[j][0]; // corresponding h, w in dX
                        if (h1 >= 0 && h1 < h0)
                            if (w1 >= 0 && w1 < w0)
                            {
                                int h2 = h*w, w2 = c*ksize + j; // h, w in tmp_dX
                                dX[c*hw + w1*h0 + h1] += tmp_dX[w2*Xh + h2]
                            }
                    }
    };

private:
    const float alpha = 1.0f, beta = 0.0f;
    int Xh, Xw, Wh, Ww;
    int c_in, c_out;
    int k = 3;
    int n, h0, w0;
    tensor<T> *X_hat, *Y_hat;
    tensor<T> *dX, *dW;
    cublasHandle_t handle;
};
}

#endif