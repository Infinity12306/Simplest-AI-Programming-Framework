#include <random>
#include <iostream>
#include <vector>
#include <fstream>
#include <cublas_v2.h>

#include "../../utils.h"
#include "../../tensor.cu"

int main(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0, 1);

    std::ofstream f_x("X.txt"), f_w("W.txt"), f_y("Y.txt");

    float *x = new float[21];
    float *w = new float[28];
    float *y = new float[12];

    for (int i=0; i<21; i++){
        x[i] = dist(gen);
        w[i] = dist(gen);
    }
    for (int j=21; j<28; j++){
        w[j] = dist(gen);
    }

    tensor<float>* X = new tensor<float>(x, std::vector<int>{3, 7}, "cpu");
    // tensor<float>* X = new tensor<float>(x, std::vector<int>{7, 3}, "cpu");
    X->gpu();
    tensor<float>* W = new tensor<float>(w, std::vector<int>{7, 4}, "cpu");
    // tensor<float>* W = new tensor<float>(w, std::vector<int>{4, 7}, "cpu");
    W->gpu();
    tensor<float>* Y = new tensor<float>(std::vector<int>{3, 4}, "gpu");

    cublasHandle_t handle;
    cublasCreate(&handle);

    sgemm_wrapper<float>(handle, X->data, W->data, Y->data, Y->shape[0], Y->shape[1], X->shape[1]);
    // sgemm_wrapper<float>(handle, X->data, W->data, Y->data, Y->shape[0], Y->shape[1], X->shape[1], 1, 0, false, true);
    // sgemm_wrapper<float>(handle, X->data, W->data, Y->data, Y->shape[0], Y->shape[1], X->shape[0], 1, 0, true, false);
    // sgemm_wrapper<float>(handle, X->data, W->data, Y->data, Y->shape[0], Y->shape[1], X->shape[0], 1, 0, true, true);

    X->cpu()->f_print(f_x);
    W->cpu()->f_print(f_w);
    Y->cpu()->f_print(f_y);

    cublasDestroy(handle);
    delete x;
    delete w;
    delete X;
    delete W;
    delete Y;
}