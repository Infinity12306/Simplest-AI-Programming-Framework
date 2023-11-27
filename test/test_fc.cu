#include<cstdio>
#include<vector>
#include<cuda_runtime.h>
#include<iostream>
#include<random>
#include<fstream>
#include "../tensor.cu"
#include "../layers/fc.cu"

int main(){
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(-1, 1);
    const int m = 7, k = 5;
    int n = 3, bsize = 3;

    std::ofstream outputW("W.txt");
    tensor<float> W = tensor<float>(std::vector<int>({m, k}), "cpu");
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<k; j++)
        {
            float data = dist(gen);
            W.data[j*m + i] = data;
            if (j == 0)
                outputW << data;
            else
                outputW << " " << data;
        }
        outputW << std::endl;
    }

    std::ofstream outputX("X.txt");
    tensor<float> X = tensor<float>(std::vector<int>({bsize, k, n}), "cpu");
    if (X.shape.size() == 1)
        n = 1;
    else
        n = X.shape[X.shape.size()-1];
    int x_num = 1;
    for (int i=0; i<X.shape.size(); i++)
        x_num *= X.shape[i];
    for (int idx=0; idx<x_num; idx++){
        float data = dist(gen);
        int bidx = idx / (k*n);
        int ridx = (idx / n) % k;
        int cidx = idx % n;
        int real_idx = bidx * (k*n) + cidx * k + ridx;
        X.data[real_idx] = data;

        outputX << data << " ";
        if (cidx == n-1)
        {
            outputX << std::endl;
            if (ridx == k-1)
                outputX << std::endl;
        }        
    }

    fc<float> fc_ = fc<float>();
    tensor<float>*Y = fc_.forward(&W, &X);
    int y_num = 1;
    for (int i=0; i<Y->shape.size(); i++)
        y_num *= Y->shape[i];
    std::cout << "Predicted Y:\n";
    for (int idx=0; idx<y_num; idx++){
        int bidx = idx / (m*n);
        int ridx = (idx / n) % m;
        int cidx = idx % n;
        int real_idx = bidx * (m*n) + cidx * m + ridx;
        std::cout << Y->data[real_idx] << " ";
        if (cidx == n-1){
            std::cout << std::endl;
            if (ridx == m-1)
                std::cout << std::endl;
        }
    }

    std::ofstream outputDy("Dy.txt");
    tensor<float>Dy = tensor<float>(Y->shape, "cpu");
    int dy_num = 1;
    for (int i=0; i<Dy.shape.size(); i++)
        dy_num *= Dy.shape[i];
    for (int idx=0; idx<dy_num; idx++){
        float data = dist(gen);
        int bidx = idx / (m*n);
        int ridx = (idx / n) % m;
        int cidx = idx % n;
        int real_idx = bidx * (m*n) + cidx * m + ridx;

        Dy.data[real_idx] = data;
        outputDy << data << " ";
        if (cidx == n-1)
        {
            outputDy << std::endl;
            if (ridx == m-1)
                outputDy << std::endl;
        }
    }

    std::vector<tensor<float>*> dWdX = fc_.backward(&Dy);

    tensor<float> *dW = dWdX[0], *dX = dWdX[1];
    std::cout << "Predicted Dw:\n";
    for (int i=0; i<m; i++)
    {
        for (int j=0; j<k; j++)
            std::cout << dW->data[j*m + i] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Predicted Dx:\n";
    for (int idx=0; idx<x_num; idx++){
        int bidx = idx / (k*n);
        int ridx = (idx / n) % k;
        int cidx = idx % n;
        int real_idx = bidx * (k*n) + cidx * k + ridx;

        std::cout << dX->data[real_idx] << " ";
        if (cidx == n-1){
            std::cout << std::endl;
            if (ridx == k-1)
                std::cout << std::endl;
        }
    }
}