#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cublas_v2.h>
#include <curand.h>

#include "../../tensor.cu"
// #include "../../layers/.cu"

void print_res(tensor<float>* x, std::ofstream f_x, 
                bool r_first=False, bool output_shape=False){
    if (x->shape.size() == 1){
        for (int i=0; i<x->shape[0]; i++){
            std::cout << x->data[i] << " ";
            f_x << x->data[i] << " ";
        }
        return;
    }

    assert x->shape.size() >= 2;
    int x_r = x->shape[-2], x_c = x->shape[-1];

    int num = 1;
    for (int i=0; i<x->shape.size(); i++){
        num *= x->shape[i];
        if (output_shape)
            f_x << x->shape[i] << " ";
    }
    if (output_shape)
        f_x << std::endl;
    
    for (int i=0; i<num; i++){
        int bidx = i / (x_r * x_c);
        int ridx = i / x_c % x_r;
        int cidx = i % x_c;
        int real_idx = 0;
        if (r_first)
            real_idx = bidx * (x_r * x_c) + ridx * x_c + c_idx;
        else
            real_idx = bidx * (x_r * x_c) + c_idx * x_r + r_idx;
        std::cout << x->data[real_idx] << " ";
        if ((i % x_c == 0) && i > 0)
            std::cout << std::endl;
        if ((i % (x_c * x_r) == 0) && i > 0)
            std::cout << std::endl;
        f_x << x->data[real_idx] << " ";
    }
    return;
}


int main(){
    cublasHandle_t handle;
    curandGenerator_t prng;
    cublasCreate(&handle);
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    std::ofstream f_x("X.txt"), f_w("W.txt"), f_y("Y.txt");
    std::ofstream f_dx("Dx.txt"), f_dw("Dw.txt"), f_dy("Dy.txt");

    print_res(X, f_x, r_first=true, output_shape=true);
    print_res(dX, f_dx, r_first=True);
    print_res(W, f_w, output_shape=true);
    print_res(dW, f_w);
    print_res(Y, f_y, output_shape=true);
    print_res(dY, f_dy);

    // delete X;
    // delete dY;
    cublasDestroy(handle);
    curandDestroyGenerator(prng);
}
