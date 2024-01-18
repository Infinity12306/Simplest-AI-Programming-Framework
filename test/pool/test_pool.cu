#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cublas_v2.h>
#include <curand.h>

#include "../../tensor.cu"
#include "../../layers/pooling.cu"


int main(){
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    std::ofstream f_x("X.txt"), f_y("Y.txt");
    std::ofstream f_dx("Dx.txt"), f_dy("Dy.txt");
    int n = 3, c = 2, h = 19, w = 15;

    max_pool<float> pool_ = max_pool<float>();

    tensor<float> *X = new tensor<float>(std::vector<int>{n, c, h, w}, "gpu");
    curandGenerateUniform(prng, X->data, X->size / sizeof(float));

    tensor<float> *Y = pool_.forward(X);

    tensor<float> *dY = new tensor<float>(Y->shape, "gpu");
    curandGenerateUniform(prng, dY->data, dY->size / sizeof(float));

    tensor<float> *dX = pool_.backward(dY);

    X->f_print(f_x);
    Y->f_print(f_y);
    dX->f_print(f_dx);
    dY->f_print(f_dy);

    delete X;
    X = nullptr;
    delete dY;
    dY = nullptr;
    curandDestroyGenerator(prng);
}
