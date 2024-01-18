#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <random>
#include <cublas_v2.h>
#include <curand.h>

#include "../../tensor.cu"
#include "../../layers/cross_entropy.cu"

int main(){
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());

    std::ofstream f_x("X.txt"), f_label("labels.txt"), f_y("loss.txt");
    std::ofstream f_dx("Dx.txt");

    int n = 5, c = 7;

    tensor<float> *X = new tensor<float>(std::vector<int>{n, c}, "gpu");
    curandGenerateUniform(prng, X->data, X->size / sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, c-1);
    tensor<int> *labels = new tensor<int>(std::vector<int>{n}, "cpu");
    for (int i=0; i<n; i++)
        labels->data[i] = dist(gen);
    labels->gpu();

    cross_entropy<float> l_ce = cross_entropy<float>();

    float loss = l_ce.forward(X, labels);
    f_y << loss;

    tensor<float> *dX = l_ce.backward();

    X->f_print(f_x);
    
    dX->f_print(f_dx);
    labels->f_print(f_label);

    delete X;
    X = nullptr;
    delete labels;
    labels = nullptr;
    curandDestroyGenerator(prng);
}
