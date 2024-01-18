#include <vector>
#include <iostream>
#include <fstream>
#include <random>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

#include "../../tensor.cu"
#include "../../activation_functions/relu.cu"
#include "../../layers/fc.cu"
#include "../../layers/conv.cu"
#include "../../layers/pooling.cu"
#include "../../layers/cross_entropy.cu"


int main(){
    cublasHandle_t handle;
    curandGenerator_t prng;
    cublasCreate(&handle);
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    std::ofstream f_x("x.txt"), f_label("labels.txt"), f_loss("loss.txt");
    std::ofstream f_dx("dx.txt");
    std::ofstream f_fc1_w("fc1_w.txt"), f_fc2_w("fc2_w.txt");
    std::ofstream f_fc1_dw("fc1_dw.txt"), f_fc2_dw("fc2_dw.txt");
    std::ofstream f_conv1_w("conv1_w.txt"), f_conv2_w("conv2_w.txt");
    std::ofstream f_conv1_dw("conv1_dw.txt"), f_conv2_dw("conv2_dw.txt");

    int n=16, c=1, h=28, w=28;

    tensor<float> *X = new tensor<float>(std::vector<int>{n, c, h, w}, "gpu");
    curandGenerateUniform(prng, X->data, X->size / sizeof(float));

    tensor<int> *labels = new tensor<int>(std::vector<int>{n}, "cpu");
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 9);
    for (int i=0; i<n; i++)
        labels->data[i] = dist(gen);
    labels->gpu();

    conv<float> conv1 = conv<float>(1, 32, handle, prng);
    relu<float> relu1 = relu<float>();
    max_pool<float> pool1 = max_pool<float>();
    conv<float> conv2 = conv<float>(32, 64, handle, prng);
    relu<float> relu2 = relu<float>();
    max_pool<float> pool2 = max_pool<float>();
    fc<float> fc1 = fc<float>(64*7*7, 128, handle, prng);
    relu<float> relu3 = relu<float>();
    fc<float> fc2 = fc<float>(128, 10, handle, prng);
    cross_entropy<float> ce = cross_entropy<float>();

    tensor<float> *y1 = conv1.forward(X);
    tensor<float> *y2 = relu1.forward(y1);
    tensor<float> *y3 = pool1.forward(y2);
    tensor<float> *y4 = conv2.forward(y3);
    tensor<float> *y5 = relu2.forward(y4);
    tensor<float> *y6 = pool2.forward(y5);

    y6->view(std::vector<int>{n, 64*7*7});

    tensor<float> *y7 = fc1.forward(y6);
    tensor<float> *y8 = relu3.forward(y7);
    tensor<float> *y9 = fc2.forward(y8);
    float loss = ce.forward(y9, labels);

    tensor<float> *dy9 = ce.backward();
    std::vector<tensor<float>*> dy8dw = fc2.backward(dy9);
    tensor<float> *dy8 = dy8dw[0], *fc2_dw = dy8dw[1];
    tensor<float> *dy7 = relu3.backward(dy8);
    std::vector<tensor<float>*> dy6dw = fc1.backward(dy7);
    tensor<float> *dy6 = dy6dw[0], *fc1_dw = dy6dw[1];
    tensor<float> *dy5 = pool2.backward(dy6);
    tensor<float> *dy4 = relu2.backward(dy5);
    std::vector<tensor<float>*> dy3dw = conv2.backward(dy4);
    tensor<float> *dy3 = dy3dw[0], *conv2_dw = dy3dw[1];
    tensor<float> *dy2 = pool1.backward(dy3);
    tensor<float> *dy1 = relu1.backward(dy2);
    std::vector<tensor<float>*> dxdw = conv1.backward(dy1);
    tensor<float> *dx = dxdw[0], *conv1_dw = dxdw[1];

    tensor<float>* fc1_w = fc1.get_w();
    tensor<float>* fc2_w = fc2.get_w();

    tensor<float>* conv1_w = conv1.get_w();
    tensor<float>* conv2_w = conv2.get_w();

    X->f_print(f_x);
    labels->f_print(f_label);
    f_loss << loss;

    fc1_w->f_print(f_fc1_w);
    fc1_dw->f_print(f_fc1_dw);
    fc2_w->f_print(f_fc2_w);
    fc2_dw->f_print(f_fc2_dw);

    conv1_w->f_print(f_conv1_w);
    conv1_dw->f_print(f_conv1_dw);
    conv2_w->f_print(f_conv2_w);
    conv2_dw->f_print(f_conv2_dw);

    dx->f_print(f_dx);

    delete X;
    X = nullptr;
    delete labels;
    labels = nullptr;
    cublasDestroy(handle);
    curandDestroyGenerator(prng);
}
