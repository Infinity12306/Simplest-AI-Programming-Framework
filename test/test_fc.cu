#include<cstdio>
#include<vector>
#include<cuda_runtime.h>
#include<iostream>
#include<random>
#include<fstream>
#include<cublas_v2.h>
#include "../tensor.cu"
#include "../layers/fc.cu"

int main(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1, 1);

    int X_r = 7, f_in = 5, f_out = 7;
    int bsize = 3;

    std::ofstream outputX("X.txt");
    tensor<float> X = tensor<float>(std::vector<int>({bsize, bsize, X_r, f_in}), "cpu");
    if (X.shape.size() == 1)
        X_r = 1;
    int x_num = 1;
    for (int i=0; i<X.shape.size(); i++)
        x_num *= X.shape[i];
    for (int idx=0; idx<x_num; idx++){
        float data = dist(gen);
        int bidx = idx / (X_r * f_in);
        int ridx = (idx / f_in) % X_r;
        int cidx = idx % f_in;
        int real_idx = bidx * (X_r * f_in) + cidx * X_r + ridx;
        X.data[real_idx] = data;

        outputX << data << " ";
        if (cidx == f_in-1)
        {
            outputX << std::endl;
            if (ridx == X_r-1)
                outputX << std::endl;
        }        
    }    

    fc<float> fc_ = fc<float>(f_in, f_out);

    std::ofstream outputW("W.txt");
    tensor<float>* W = fc_.W;
    for (int h=0; h<f_in; h++)
    {
        for (int w=0; w<f_out; w++)
        {
            int idx = w * f_in + h;
            outputW << W->data[idx] << " ";
        }
        outputW << std::endl;
    }
    outputW << std::endl;

    tensor<float>*Y = fc_.forward(&X);

    int y_num = 1;
    for (int i=0; i<Y->shape.size(); i++)
        y_num *= Y->shape[i];
    std::cout << "Predicted Y:\n";
    for (int idx=0; idx<y_num; idx++){
        int bidx = idx / (X_r*f_out);
        int ridx = (idx / f_out) % X_r;
        int cidx = idx % f_out;
        int real_idx = bidx * (X_r*f_out) + cidx * X_r + ridx;
        std::cout << Y->data[real_idx] << " ";
        if (cidx == f_out-1){
            std::cout << std::endl;
            if (ridx == X_r-1)
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
        int bidx = idx / (X_r*f_out);
        int ridx = (idx / f_out) % X_r;
        int cidx = idx % f_out;
        int real_idx = bidx * (X_r*f_out) + cidx * X_r + ridx;

        Dy.data[real_idx] = data;
        outputDy << data << " ";
        if (cidx == f_out-1)
        {
            outputDy << std::endl;
            if (ridx == X_r-1)
                outputDy << std::endl;
        }
    }

    std::vector<tensor<float>*> dWdX = fc_.backward(&Dy);

    tensor<float> *dW = dWdX[0], *dX = dWdX[1];
    std::cout << "Predicted Dw:\n";
    for (int i=0; i<f_in; i++)
    {
        for (int j=0; j<f_out; j++)
            std::cout << dW->data[j*f_in + i] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Predicted Dx:\n";
    for (int idx=0; idx<x_num; idx++){
        int bidx = idx / (X_r*f_in);
        int ridx = (idx / f_in) % X_r;
        int cidx = idx % f_in;
        int real_idx = bidx * (X_r*f_in) + cidx * X_r + ridx;

        std::cout << dX->data[real_idx] << " ";
        if (cidx == f_in-1){
            std::cout << std::endl;
            if (ridx == X_r-1)
                std::cout << std::endl;
        }
    }
}