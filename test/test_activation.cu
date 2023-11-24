#include<cstdio>
#include<vector>
#include<cuda_runtime.h>
#include<iostream>
#include<random>
#include "activation_functions/relu.cu"
#include "activation_functions/sigmoid.cu"
#include "tensor.cu"

int main(){
    relu<float> relu_func = relu<float>();
    sigmoid<float> sigmoid_func = sigmoid<float>();
    std::vector<int> shape;
    shape.push_back(64);
    const char* device = "cpu";
    tensor<float>* x = new tensor<float>(shape, device);
    for (int i=0; i<64; i++){
        x->data[i] = i-32;
    }

    // relu_forward
    tensor<float>* y = relu_func.forward(x);
    cudaDeviceSynchronize();
    std::cout << "relu_y: " << std::endl;
    for (int i=0; i<64; i++){
        if (i > 0)
            std::cout << " ";
        std::cout << y->data[i];
    }
    std::cout << std::endl << std::endl;

    // relu_backward
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    tensor<float>* grad_y = new tensor<float>(shape, device);
    for (int i=0; i<64; i++){
        grad_y->data[i] = (float)dis(gen);
    }
    std::cout << "Given relu_grad_y: " << std::endl;
    for (int i=0; i<64; i++)
    {
        if (i > 0)
            std::cout << " ";
        std::cout << grad_y->cpu()->data[i];
    }
    std::cout << std::endl;
    tensor<float>* grad_x = relu_func.backward(grad_y);
    cudaDeviceSynchronize();
    std::cout << "relu_grad_x: " << std::endl;
    for (int i=0; i<64; i++)
    {
        if (i > 0)
            std::cout << " ";
        std::cout << grad_x->data[i];
    }
    std::cout << std::endl << std::endl;

    std::cout << std::endl;

    // sigmoid_forward
    tensor<float>* sig_y = sigmoid_func.forward(x);
    cudaDeviceSynchronize();
    std::cout << "sigmoid_y: " << std::endl;
    for (int i=0; i<64; i++){
        if (i > 0)
            std::cout << " ";
        std::cout << sig_y->data[i];
    }
    std::cout << std::endl << std::endl;

    // sigmoid_backward
    std::cout << "Given sigmoid_grad_y: " << std::endl;
    for (int i=0; i<64; i++)
    {
        if (i > 0)
            std::cout << " ";
        std::cout << grad_y->cpu()->data[i];
    }
    std::cout << std::endl;
    tensor<float>* sig_grad_x = sigmoid_func.backward(grad_y);
    cudaDeviceSynchronize();
    std::cout << "sigmoid_grad_x: " << std::endl;
    for (int i=0; i<64; i++)
    {
        if (i > 0)
            std::cout << " ";
        std::cout << sig_grad_x->data[i];
    }
    std::cout << std::endl << std::endl;

    delete x;
    delete grad_y;
}
