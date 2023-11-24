#include<cstdio>
#include<vector>
#include<cuda_runtime.h>
#include<iostream>
#include<random>
#include<fstream>
#include "tensor.cu"
#include "../layers/fc.cu"

int main(){
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(-1, 1);
    const int m = 8, k = 5, n = 1;

    std::ofstream outputW("W.txt");
    tensor<float> W = tensor<float>(std::vector<int>({m, k}), "gpu");
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<k; j++)
        {
            float data = dist(gen);
            W.data[i*k + j] = data;
            if (j == 0)
                outputW << data;
            else
                outputW << " " << data;
        }
        outputW << std::endl;
    }

    std::ofstream outputX("X.txt");
    tensor<float> X = tensor<float>(std::vector<int>({k, n}), "gpu");
    for (int i=0; i<k; i++)
    {
        for (int j=0; j<n; j++)
        {
            float data = dist(gen);
            X.data[i*n + j] = data;
            if (j == 0)
                outputX << data;
            else
                outputX << " " << data;
        }
        outputX << std::endl;
    }

    std::ofstream outputDy("Dy.txt");
    tensor<float>Dy = tensor<float>(std::vector<int>({m, n}), "gpu");
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<n; j++)
        {
            float data = dist(gen);
            Dy.data[i*n + j] = data;
            if (j == 0)
                outputDy << data;
            else
                outputDy << " " << data;
        }
        outputDy << std::endl;
    }

    fc<float> fc_;
    tensor<float>*Y = fc_.forward(&W, &X);
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<n; j++)
            std::cout << Y->data[i*n + j] << " ";
        std::cout << std::endl;
    }
}