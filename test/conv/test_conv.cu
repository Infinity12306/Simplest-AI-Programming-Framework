#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cublas_v2.h>
#include <curand.h>
#include <string>

#include "../../tensor.cu"
#include "../../layers/conv.cu"

int main(){
    cublasHandle_t handle;
    curandGenerator_t prng;
    cublasCreate(&handle);
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    std::ofstream f_x("X.txt"), f_w("W.txt"), f_y("Y.txt");
    std::ofstream f_dx("Dx.txt"), f_dw("Dw.txt"), f_dy("Dy.txt");

    int n=3, c_in=5, h=7, w=9;
    int c_out = 3;

    tensor<float> *X = new tensor<float>(std::vector<int>{n, c_in, h, w}, "gpu");
    curandGenerateUniform(prng, X->data, n*c_in*h*w);
    
    conv<float> conv_ = conv<float>(c_in, c_out, handle, prng);
    tensor<float> *Y = conv_.forward(X);

    tensor<float> *dY = new tensor<float>(Y->shape, "gpu");
    curandGenerateUniform(prng, dY->data, n*c_out*h*w);

    std::vector<tensor<float>*> dxdw = conv_.backward(dY);

    tensor<float> *dX = dxdw[0], *dW = dxdw[1];

    tensor<float> *W = conv_.get_w();

    X->f_print(f_x);
    W->f_print(f_w);
    Y->f_print(f_y);
    dX->f_print(f_dx);
    dW->f_print(f_dw);
    dY->f_print(f_dy);

    delete X;
    delete dY;
    cublasDestroy(handle);
    curandDestroyGenerator(prng);
}
