#ifndef CONV
#define CONV

#include <vector>
#include <cstring>
#include <random>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include "../utils.h"
#include "../tensor.cu"

int coords[18] = {-1, -1, 0, -1, 1, -1, 
                  -1, 0, 0, 0, 1, 0,
                  -1, 1, 0, 1, 1, 1};


template<typename T>
__global__ void im2col(const T* X, T* X_hat, int* coords, 
                            int h0, int w0, int c_in, int ksize, int n){
    CUDA_KERNEL_LOOP(idx, n){
        int bidx = idx / (h0 * w0 * c_in);
        int ridx = idx / (w0 * c_in) % h0;
        int cidx = idx / c_in % w0;
        int chidx = idx % c_in; // channel index
        for (int i=0; i<ksize; i++){
            int real_idx = idx * ksize + i;
            int ori_cidx = cidx + coords[i*2 + 0];
            int ori_ridx = ridx + coords[i*2 + 1];
            int ori_idx = bidx * (c_in * h0 * w0) + chidx * h0 * w0 + 
                            ori_ridx * w0 + ori_cidx;
            if (ori_cidx < 0 || ori_cidx >= w0)
                X_hat[real_idx] = 0;
            else if (ori_ridx < 0 || ori_ridx >= h0)
                X_hat[real_idx] = 0;
            else
                X_hat[real_idx] = X[ori_idx];
        }
    }
}

template<typename T>
__global__ void col2im(T* X, const T* X_hat, int* coords,
                            int h0, int w0, int c_in, int ksize, int n){
    CUDA_KERNEL_LOOP(idx, n){
        int bidx = idx / (h0 * w0 * c_in);
        int ridx = idx / (w0 * c_in) % h0;
        int cidx = idx / c_in % w0;
        int chidx = idx % c_in; // channel index
        for (int i=0; i<ksize; i++){
            int real_idx = idx * ksize + i;
            int ori_cidx = cidx + coords[i*2 + 0];
            int ori_ridx = ridx + coords[i*2 + 1];
            int ori_idx = bidx * (c_in * h0 * w0) + chidx * h0 * w0 + 
                            ori_ridx * w0 + ori_cidx;
            if (ori_cidx >= 0 && ori_cidx < w0)
                if (ori_ridx >= 0 && ori_ridx < h0){
                    atomicAdd(&X[ori_idx], X_hat[real_idx]);
                }
        }
    }
}



template <typename T>
class conv{
public:
    conv(int c_in, int c_out, cublasHandle_t handle, curandGenerator_t prng):X_hat(nullptr), W(nullptr), Y(nullptr), 
            dX(nullptr), dW(nullptr), c_in(c_in), c_out(c_out), handle(handle), prng(prng), coords_gpu(nullptr){
        Wh = c_out, Ww = c_in*k*k;

        W = new tensor<T>(std::vector<int>({Wh, Ww}), "gpu");
        curandGenerateUniform(prng, W->data, Wh*Ww);

        std::size_t coords_size = 9 * 2 * sizeof(int);
        cudaMalloc(&coords_gpu, coords_size);
        cudaMemcpy(coords_gpu, coords, coords_size, cudaMemcpyHostToDevice);
    }
    ~conv(){
        if (X_hat != nullptr)
            delete X_hat;
        if (Y != nullptr)
            delete Y;
        if (W != nullptr)
            delete W;
        if (dX != nullptr)
            delete dX;
        if (dW != nullptr)
            delete dW;
        if (dX_hat != nullptr)
            delete dX_hat;
        if (coords_gpu != nullptr)
            cudaFree(coords_gpu);
    }

    // Y: C_out x H*W, X: n c h w, X_hat: n x H*W x C_in*K*K, W: C_out X C_in*K*K
    // Y = W * X_hat^T
    tensor<T>* forward(tensor<T>*X){
        n = X->shape[0], c_in = X->shape[1], h0 = X->shape[2], w0 = X->shape[3];
        Xh = h0*w0, Xw = c_in*k*k;
        int k_size = k*k;

        X_hat = new tensor<T>(std::vector<int>({n, Xh, Xw}), "gpu");
        Y = new tensor<T>(std::vector<int>({n, Wh, Xh}), "gpu");

        int kernel_num = n * Xh * c_in;
        im2col<T><<<GET_BLOCK_NUM(kernel_num), THREADNUM>>>(X->data, 
            X_hat->data, coords_gpu, h0, w0, c_in, k_size, kernel_num);

        T *X_data = X_hat->gpu()->data;
        T *Y_data = Y->gpu()->data;
        for (int i=0; i<n; i++){
            sgemm_wrapper<T>(handle, W->data, X_data, Y_data, Wh, Xh, Xw, 
                            alpha, beta, false, true);
            X_data += Xh*Xw, Y_data += Wh*Xh;
        }
        return Y->view(std::vector<int>{n, c_out, h0, w0});
    };

    std::vector<tensor<T>*> backward(tensor<T>* dY){
        dY = dY->view(std::vector<int>{n, Wh, Xh});
        dX = new tensor<T>(std::vector<int>({n, c_in, h0, w0}), "gpu");
        dW = new tensor<T>(std::vector<int>({c_out, c_in, k, k}), "gpu");
        dX_hat = new tensor<T>(std::vector<int>({n, Xh, Xw}), "gpu");

        // dY->gpu(), X_hat->gpu();
        // dX->cpu();
        T *dY_data = dY->data, *X_hat_data = X_hat->data, *dX_hat_data = dX_hat->data;
        float alpha_w = 1.0f, beta_w = 1.0f;

        for (int i=0; i<n; i++)
        {
            // alpha_w = 1.0 / (float)(i+1), beta_w = (float)i / (float)(i+1);
            sgemm_wrapper<T>(handle, dY_data, X_hat_data, dW->data, Wh, Ww, Xh, alpha_w, beta_w);
            sgemm_wrapper<T>(handle, dY_data, W->data, dX_hat_data, Xh, Xw, Wh, 
                            alpha, beta, true, false);
            dY_data += Wh * Xh, X_hat_data += Xh * Xw, dX_hat_data += Xh * Xw;
        }
        int kernel_num = n * h0 * w0 * c_in;
        col2im<T><<<GET_BLOCK_NUM(kernel_num), THREADNUM>>>(dX->data, 
            dX_hat->data, coords_gpu, h0, w0, c_in, k*k, kernel_num);
        dY = dY->view(std::vector<int>{n, c_out, h0, w0});
        return std::vector<tensor<T>*>({dX, dW});
    };

    tensor<T>* get_w(){
        return W->view(std::vector<int>{c_out, c_in, k, k});
    }

    tensor<T>* get_dw(){
        return dW;
    }

    tensor<T>* get_Xhat(){
        return X_hat;
    }

    tensor<T>* get_dXhat(){
        return dX_hat;
    }

private:
    const float alpha = 1.0f, beta = 0.0f;
    int Xh, Xw, Wh, Ww;
    int c_in, c_out;
    int k = 3;
    int n, h0, w0;
    tensor<T> *W, *Y;
    tensor<T> *X_hat;
    tensor<T> *dX, *dW, *dX_hat;
    cublasHandle_t handle;
    curandGenerator_t prng;
    int *coords_gpu;
};

#endif