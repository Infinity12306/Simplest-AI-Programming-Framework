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

    std::vector<tensor<float>*> dwdx = fc_.backward(Dy);
    tensor<float> *Dw = dwdx[0], *Dx = dwdx[1];

    X->f_print(f_x);
    fc_.fc_W->f_print(f_w);
    Y->f_print(f_y);
    Dx->f_print(f_dx);
    Dw->f_print(f_dw);
    Dy->f_print(f_dy);

    // tensor<float> X = tensor<float>(std::vector<int>({bsize, bsize, X_r, f_in}), "gpu");
    // if (X.shape.size() == 1)
    //     X_r = 1;
    // int x_num = 1;
    // for (int i=0; i<X.shape.size(); i++){
    //     f_x << X.shape[i] << " ";
    //     x_num *= X.shape[i];
    // }
    // f_x << std::endl;
    // curandGenerateUniform(prng, X.data, bsize * bsize * X_r * f_in);

    // fc<float> fc_ = fc<float>(f_in, f_out, handle, prng);
    // tensor<float>*Y = fc_.forward(&X);

    // tensor<float>Dy = tensor<float>(Y->shape, "gpu");
    // int y_num = 1;
    // for (int i=0; i<Y->shape.size(); i++){
    //     y_num *= Y->shape[i];
    //     f_y << Y->shape[i] << " ";
    // }
    // f_y << std::endl;
    // curandGenerateUniform(prng, Dy.data, y_num);

    // std::vector<tensor<float>*> dWdX = fc_.backward(&Dy);

    // X.cpu();
    // for (int idx=0; idx<x_num; idx++){
    //     // float data = dist(gen);
    //     int bidx = idx / (X_r * f_in);
    //     int ridx = (idx / f_in) % X_r;
    //     int cidx = idx % f_in;
    //     int real_idx = bidx * (X_r * f_in) + cidx * X_r + ridx;
    //     // X.data[real_idx] = data;

    //     f_x << X.data[real_idx] << " ";
    // }    


    // tensor<float>* W = fc_.fc_W->cpu();
    // for (int i=0; i<W->shape.size(); i++){
    //     f_w << W->shape[i] << " ";
    // }
    // f_w << std::endl;

    // for (int h=0; h<f_in; h++)
    // {
    //     for (int w=0; w<f_out; w++)
    //     {
    //         int idx = w * f_in + h;
    //         f_w << W->data[idx] << " ";
    //     }
    // }

    // Y = Y->cpu();
    // std::cout << "Predicted Y:\n";
    // for (int idx=0; idx<y_num; idx++){
    //     int bidx = idx / (X_r*f_out);
    //     int ridx = (idx / f_out) % X_r;
    //     int cidx = idx % f_out;
    //     int real_idx = bidx * (X_r*f_out) + cidx * X_r + ridx;
    //     std::cout << Y->data[real_idx] << " ";
    //     if (cidx == f_out-1){
    //         std::cout << std::endl;
    //         if (ridx == X_r-1)
    //             std::cout << std::endl;
    //     }
    //     f_y << Y->data[real_idx] << " ";
    // }

    // Dy.cpu();
    // for (int i=0; i<y_num; i++){
    //     int batch_size = i / (X_r * f_out);
    //     int ridx = i / f_out % X_r;
    //     int cidx = i % f_out;
    //     int real_idx = batch_size * (X_r * f_out) + cidx * X_r + ridx;
    //     f_dy << Dy.data[real_idx] << " ";
    // }


    // tensor<float> *dW = dWdX[0]->cpu(), *dX = dWdX[1]->cpu();

    // std::cout << "Predicted Dw:\n";
    // for (int i=0; i<f_in; i++)
    // {
    //     for (int j=0; j<f_out; j++){
    //         std::cout << dW->data[j*f_in + i] << " ";
    //         f_dw << dW->data[j*f_in + i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // std::cout << "Predicted Dx:\n";
    // for (int idx=0; idx<x_num; idx++){
    //     int bidx = idx / (X_r*f_in);
    //     int ridx = (idx / f_in) % X_r;
    //     int cidx = idx % f_in;
    //     int real_idx = bidx * (X_r*f_in) + cidx * X_r + ridx;

    //     std::cout << dX->data[real_idx] << " ";
    //     if (cidx == f_in-1){
    //         std::cout << std::endl;
    //         if (ridx == X_r-1)
    //             std::cout << std::endl;
    //     }
    //     f_dx << dX->data[real_idx] << " ";
    // }
    delete X;
    delete Dy;
    cublasDestroy(handle);
    curandDestroyGenerator(prng);
}