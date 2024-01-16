#ifndef CONV
#define CONV

#include "../utils.h"
#include "../tensor.cu"
#include <vector>
#include <cstring>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

// const std::vector<std::vector<int>> coords = {{-1, -1}, {0, -1}, {1, -1},
//                                                 {-1, 0}, {0, 0}, {1, 0},
//                                                 {-1, 1}, {0, 1}, {1, 1}};
int coords[18] = {-1, -1, 0, -1, 1, -1, 
                  -1, 0, 0, 0, 1, 0
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
            // int real_idx = bidx * (h0 * w0 * c_in * ksize) + (chidx * ksize + i) * 
            //                     (h0 * w0) + ridx * w0 + cidx;
            int real_idx = bidx * (h0 * w0 * c_in * ksize) + (chidx * ksize + i) * 
                            (h0 * w0) + ridx*w0 + cidx;
            int ori_cidx = cidx + coords[i*2 + 0];
            int ori_ridx = ridx + coords[i*2 + 1];
            if (ori_cidx < 0 || ori_cidx >= w0)
                X_hat[real_idx] = 0;
            else if (ori_ridx < 0 || ori_ridx >= h0)
                X_hat[real_idx] = 0;
            else
                X_hat[real_idx] = X[bidx * (c_in * h0 * w0) + chidx * (h0 * w0) + 
                                    ori_ridx * w0 + ori_cidx] ;
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
            int real_idx = bidx * (h0 * w0 * c_in * ksize) + (chidx * ksize + i) * h0 * w0 +
                                ridx * w0 + cidx;
            int ori_cidx = cidx + coords[i*2 + 0];
            int ori_ridx = ridx + coords[i*2 + 1];
            if (ori_cidx >= 0 && ori_cidx < w0)
                if (ori_ridx >= 0 && ori_ridx < h0){
                    int ori_real_idx = bidx * c_in * h0 * w0 + chidx * h0 * w0 + 
                                            ori_ridx * w0 + ori_cidx;
                    X[ori_real_idx] += X_hat[real_idx];
                }
        }
    }
}



template <typename T>
class conv{
public:
    tensor<T> *W;

    conv(int c_in, int c_out, cublasHandle_t handle, curandGenerator_t prng):X_hat(nullptr), W(nullptr), Y(nullptr), 
            dX(nullptr), dW(nullptr), c_in(c_in), c_out(c_out), handle(handle), prng(prng), coords_gpu(nullptr){
        Wh = c_out, Ww = c_in*k*k;
        W = new tensor<T>(std::vector<int>({Wh, Ww}), "cpu");
        for (int i=0; i<W->size/sizeof(T); i++){
            W->data[i] = 1;
        }
        W = W->gpu();
        // W = new tensor<T>(std::vector<int>({c_out, c_in, k, k}), "gpu");
        // curandGenerateUniform(prng, W->data, Wh*Ww);
        std::size_t coords_size = 9 * 2 * sizeof(int);
        cudaMalloc(&coords_gpu, coords_size);
        cudaMemcpy(coords_gpu, coords, coords_size, cudaMemcpyHostToDevice);
        // cudaMemcpy((void*)(&coords), (void*)coords_gpu, coords_size, cudaMemcpyHostToDevice);
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

        // forward_X_hat(X->cpu()->data, X_hat->cpu()->data)

        // W->gpu();
        int kernel_num = n * Xh * c_in;
        im2col<T><<<GET_BLOCK_NUM(kernel_num), THREADNUM>>>(X->data, 
            X_hat->data, coords_gpu, h0, w0, c_in, k_size, kernel_num);

        T *X_data = X_hat->gpu()->data;
        T *Y_data = Y->gpu()->data;
        for (int i=0; i<n; i++){
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Wh, Xh, Ww, &alpha, W->data, Wh, 
                X_data, Xh, &beta, Y_data, Wh);
            X_data += Xh*Xw, Y_data += Wh*Xh;
        }
        return Y;
    };

    std::vector<tensor<T>*> backward(tensor<T>* dY){
        dX = new tensor<T>(std::vector<int>({n, c_in, h0, w0}), "gpu");
        dW = new tensor<T>(std::vector<int>({c_out, c_in, k, k}), "gpu");
        tensor<T> dX_hat = tensor<T>(std::vector<int>({n, Xh, Xw}), "gpu");

        // dY->gpu(), X_hat->gpu();
        // dX->cpu();
        T *dY_data = dY->data, *X_hat_data = X_hat->data, *dX_hat_data = dX_hat.data;
        float alpha_w, beta_w;
        // int chw = c_in * h0 * w0;

        for (int i=0; i<n; i++)
        {
            // tmp_dX.gpu();
            alpha_w = 1.0 / (float)(i+1), beta_w = (float)i / (float)(i+1);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Wh, Ww, Xh, &alpha_w, dY_data, Wh, 
                X_hat_data, Xh, &beta_w, dW->data, Wh);
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Xh, Xw, Wh, &alpha, dY_data, Wh, 
                W->data, Wh, &beta, dX_hat_data, Xh);
            // backward_dX_hat(tmp_dX.cpu()->data, dX_data);
            dY_data += Wh * Xh, X_hat_data += Xh * Xw, dX_hat_data += Xh * Xw;
        }
        int kernel_num = n * h0 * w0 * c_in;
        col2im<T><<<GET_BLOCK_NUM(kernel_num), THREADNUM>>>(dX->data, 
            dX_hat.data, coords_gpu, h0, w0, c_in, k*k, kernel_num);
        return std::vector<tensor<T>*>({dX, dW});
    };

    // // Xh: h0*w0, Xw: c_in*k*k
    // // construct X_hat from X
    // void forward_X_hat(T*X, T*X_hat){
    //     int chw = c_in*h0*w0, hw = h0*w0;
    //     for (int i=0; i<n; i++){
    //         for (int h=0; h<h0; h++)
    //             for (int w=0; w<w0; w++)
    //                 for (int c=0; c<c_in; c++)
    //                     for (int j=0; j<ksize; j++){
    //                         int hidx = h*w0 + w, widx = c*ksize + j; // h, w in X_hat
    //                         int cur_idx = i*(Xh * Xw) + widx*Xh + hidx;

    //                         int h1 = h + coords[j][1], w1 = w + coords[j][0]; // h, w in X
    //                         int ori_idx = i*chw + c*hw + w1*h0 + h1;

    //                         if (h1 < 0 || w1 < 0 || h1 >= h0 || w1 >= w0)
    //                             X_hat->data[cur_idx] = (T)0;
    //                         else
    //                             X_hat->data[cur_idx] = X->data[ori_idx];
    //                     }
    // }

    // // Xh: h0*w0, Xw: c_in*k*k
    // // backpropagate gradient from X_hat to X
    // void backward_dX_hat(T*tmp_dX, T* dX){
    //     int ksize = k * k, hw = h0 * w0;
    //     for (int h=0; h<h0; h++)
    //         for (int w=0; w<w0; w++)
    //             for (int c=0; c<c_in; c++)
    //                 for (int j=0; j<ksize; j++){
    //                     int h1 = h + coords[j][1], w1 = w + coords[j][0]; // corresponding h, w in dX
    //                     if (h1 >= 0 && h1 < h0)
    //                         if (w1 >= 0 && w1 < w0)
    //                         {
    //                             int h2 = h*w, w2 = c*ksize + j; // h, w in tmp_dX
    //                             dX[c*hw + w1*h0 + h1] += tmp_dX[w2*Xh + h2]
    //                         }
    //                 }
    // };

private:
    const float alpha = 1.0f, beta = 0.0f;
    int Xh, Xw, Wh, Ww;
    int c_in, c_out;
    int k = 3;
    int n, h0, w0;
    tensor<T> *Y;
    tensor<T> *X_hat;
    tensor<T> *dX, *dW;
    cublasHandle_t handle;
    curandGenerator_t prng;
    // std::vector<int> *coords_gpu;
    int *coords_gpu;
};

#endif