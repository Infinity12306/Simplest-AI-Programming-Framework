#include <vector>
#include <iostream>
#include <fstream>
#include <random>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include "../../tensor.cu"
// #include "../../layers/.cu"


int main(){
    cublasHandle_t handle;
    curandGenerator_t prng;
    cublasCreate(&handle);
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    std::ofstream f_x("X.txt"), f_w("W.txt"), f_y("Y.txt");
    std::ofstream f_dx("Dx.txt"), f_dw("Dw.txt"), f_dy("Dy.txt");


    tensor<float> *X = new tensor<float>(std::vector<int>{}, "gpu");
    curandGenerateUniform(prng, X->data, X->size / sizeof(float));

    tensor<float> * dY = new tensor<float>(Y->shape, "gpu");
    curandGenerateUniform(prng, dY->data, dY->size / sizeof(float));

    X->f_print(f_x);
    W->f_print(f_w);
    Y->f_print(f_y);
    dX->f_print(f_dx);
    dW->f_print(f_dw);
    dY->f_print(f_dy);

    delete X;
    X = nullptr;
    delete dY;
    dY = nullptr;
    cublasDestroy(handle);
    curandDestroyGenerator(prng);
}
