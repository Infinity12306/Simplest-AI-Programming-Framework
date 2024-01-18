#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <fstream>
#include <cublas_v2.h>
#include <curand.h>
#include "../../tensor.cu"
#include "../../layers/fc.cu"

int main(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1, 1);
    cublasHandle_t handle;
    curandGenerator_t prng;
    cublasCreate(&handle);
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    std::ofstream f_x("X.txt"), f_w("W.txt");
    std::ofstream f_y("Y.txt");
    std::ofstream f_dy("Dy.txt"), f_dw("Dw.txt"), f_dx("Dx.txt");

    int X_r = 7, f_in = 5, f_out = 7;
    int bsize = 3;

    tensor<float>* X  = new tensor<float>(std::vector<int>{bsize, X_r, f_in}, "gpu");
    curandGenerateUniform(prng, X->data, X->size / sizeof(float));

    fc<float> fc_  = fc<float>(f_in, f_out, handle, prng);

    tensor<float>* Y = fc_.forward(X);

    tensor<float>* Dy = new tensor<float>(std::vector<int>{bsize, X_r, f_out}, "gpu");
    curandGenerateUniform(prng, Dy->data, Dy->size / sizeof(float));

    std::vector<tensor<float>*> dxdw = fc_.backward(Dy);
    tensor<float> *Dx = dwdx[0], *Dw = dwdx[1];

    X->f_print(f_x);
    fc_.fc_W->f_print(f_w);
    Y->f_print(f_y);
    Dx->f_print(f_dx);
    Dw->f_print(f_dw);
    Dy->f_print(f_dy);

    delete X;
    delete Dy;
    cublasDestroy(handle);
    curandDestroyGenerator(prng);
}