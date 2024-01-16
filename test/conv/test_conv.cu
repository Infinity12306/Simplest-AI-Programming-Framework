#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <cublas_v2.h>
#include <curand.h>
#include <string>

#include "../../tensor.cu"
#include "../../layers/conv.cu"

void print_res(tensor<float>* x, std::ofstream& f_x, std::string name,
                bool r_first=false, bool output_shape=false, std::vector<int> shape={-1}){
    if (x->shape.size() == 1){
        for (int i=0; i<x->shape[0]; i++){
            std::cout << x->data[i] << " ";
            f_x << x->data[i] << " ";
        }
        return;
    }

    if (!(x->shape.size() >= 2)){
        std::cout << "Error!" << std::endl;
        return;
    }

    int x_r = x->shape[x->shape.size()-2], x_c = x->shape[x->shape.size()-1];

    int num = 1;
    for (int i=0; i<x->shape.size(); i++){
        num *= x->shape[i];
        if (output_shape)
            if (shape == std::vector<int>{-1})
                f_x << x->shape[i] << " ";
    }
    if (output_shape)
        if (shape == std::vector<int>{-1})
            f_x << std::endl;
    if (output_shape)
        if (shape != std::vector<int>{-1}){
            for (int i=0; i<shape.size(); i++)
                f_x << shape[i] << " ";
            std::cout << std::endl;
        }
    
    std::cout << name+":" << std::endl;
    for (int i=0; i<num; i++){
        int bidx = i / (x_r * x_c);
        int ridx = i / x_c % x_r;
        int cidx = i % x_c;
        int real_idx = 0;
        if (r_first)
            real_idx = bidx * (x_r * x_c) + ridx * x_c + cidx;
        else
            real_idx = bidx * (x_r * x_c) + cidx * x_r + ridx;
        std::cout << x->data[real_idx] << " ";
        if ((i % x_c == x_c - 1) && i > 0)
            std::cout << std::endl;
        if ((i % (x_c * x_r) == (x_c * x_r - 1)) && i > 0)
            std::cout << std::endl;
        f_x << x->data[real_idx] << " ";
    }
    std::cout << std::endl;
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

    int n=1, c_in=1, h=3, w=3, k=3;
    int c_out = 1;
    tensor<float> *X = new tensor<float>(std::vector<int>{n, c_in, h, w}, "cpu");
    for (int i=0; i<X->size/sizeof(float); i++)
        // X->data[i] = i;
        X->data[i] = 1;
    X->gpu();
    // tensor<float> *X = new tensor<float>(std::vector<int>{n, c_in, h, w}, "gpu");
    // curandGenerateUniform(prng, X->data, n*c_in*h*w);
    
    conv<float> conv_ = conv<float>(c_in, c_out, handle, prng);
    tensor<float> *Y = conv_.forward(X);
    Y = Y->cpu();

    tensor<float> *dY = new tensor<float>(Y->shape, "gpu");
    curandGenerateUniform(prng, dY->data, n*c_out*h*w);

    std::vector<tensor<float>*> dxdw = conv_.backward(dY);
    tensor<float> *dX = dxdw[0]->cpu(), *dW = dxdw[1]->cpu();

    X = X->cpu(), dY = dY->cpu();
    tensor<float> *W = conv_.W->cpu();

    print_res(X, f_x, "X", true, true);
    print_res(dX, f_dx, "Dx", true);
    print_res(W, f_w, "W", false, true, std::vector<int>{c_out, c_in, k, k});
    print_res(dW, f_dw, "Dw");
    print_res(Y, f_y, "Y", false, true);
    print_res(dY, f_dy, "Dy");

    delete X;
    delete dY;
    cublasDestroy(handle);
    curandDestroyGenerator(prng);
}
