#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <fstream>
#include "../../activation_functions/relu.cu"
#include "../../activation_functions/sigmoid.cu"
#include "../../tensor.cu"

int main(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    std::ofstream input_data("X.txt");
    std::ofstream out_grad("Dy.txt");
    std::ofstream relu_out("relu_Y.txt");
    std::ofstream relu_dx("relu_Dx.txt");
    std::ofstream sig_out("sig_Y.txt");
    std::ofstream sig_dx("sig_Dx.txt");

    relu<float> relu_func = relu<float>();
    sigmoid<float> sigmoid_func = sigmoid<float>();
    std::vector<int> shape;

    shape.push_back(64);
    const char* device = "cpu";
    tensor<float>* x = new tensor<float>(shape, device);
    for (int i=0; i<64; i++){
        x->data[i] = dis(gen);
        input_data << x->data[i] << " ";
    }

    // relu_forward
    tensor<float>* y = relu_func.forward(x);
    cudaDeviceSynchronize();
    std::cout << "relu_y: " << std::endl;
    y = y->cpu();
    for (int i=0; i<64; i++){
        if (i > 0)
            std::cout << " ";
        std::cout << y->data[i];
        relu_out << y->data[i] << " "; 
    }
    std::cout << std::endl << std::endl;

    // relu_backward
    tensor<float>* grad_y = new tensor<float>(shape, device);
    for (int i=0; i<64; i++){
        grad_y->data[i] = dis(gen);
        out_grad << grad_y->data[i] << " ";
    }
    std::cout << "Given relu_grad_y: " << std::endl;
    grad_y = grad_y->cpu();
    for (int i=0; i<64; i++)
    {
        if (i > 0)
            std::cout << " ";
        std::cout << grad_y->data[i];
    }
    std::cout << std::endl;

    tensor<float>* grad_x = relu_func.backward(grad_y);
    cudaDeviceSynchronize();
    std::cout << "relu_grad_x: " << std::endl;
    grad_x = grad_x->cpu();
    for (int i=0; i<64; i++)
    {
        if (i > 0)
            std::cout << " ";
        std::cout << grad_x->data[i];
        relu_dx << grad_x->data[i] << " ";
    }
    std::cout << std::endl << std::endl;

    std::cout << std::endl;

    // sigmoid_forward
    tensor<float>* sig_y = sigmoid_func.forward(x);
    cudaDeviceSynchronize();
    std::cout << "sigmoid_y: " << std::endl;
    sig_y = sig_y->cpu();
    for (int i=0; i<64; i++){
        if (i > 0)
            std::cout << " ";
        std::cout << sig_y->data[i];
        sig_out << sig_y->data[i] << " ";
    }
    std::cout << std::endl << std::endl;

    // sigmoid_backward
    std::cout << "Given sigmoid_grad_y: " << std::endl;
    grad_y = grad_y->cpu();
    for (int i=0; i<64; i++)
    {
        if (i > 0)
            std::cout << " ";
        std::cout << grad_y->data[i];
    }
    std::cout << std::endl;

    tensor<float>* sig_grad_x = sigmoid_func.backward(grad_y);
    cudaDeviceSynchronize();
    std::cout << "sigmoid_grad_x: " << std::endl;
    sig_grad_x = sig_grad_x->cpu();
    for (int i=0; i<64; i++)
    {
        if (i > 0)
            std::cout << " ";
        std::cout << sig_grad_x->data[i];
        sig_dx << sig_grad_x->data[i] << " ";
    }
    std::cout << std::endl << std::endl;

    delete x;
    delete grad_y;
}
