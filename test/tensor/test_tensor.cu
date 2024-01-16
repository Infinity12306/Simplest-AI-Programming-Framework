#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <random>

#include "../../tensor.cu"

template<typename T>
void print_vector(std::vector<T> x){
    for (int i=0; i<x.size(); i++)
        std::cout << x[i] << " ";
    std::cout << std::endl;
    return;
}

template<typename T>
void print_tensor(tensor<T>& x){
    x.print_();
    print_vector(x.shape);
    print_vector(x.strides);
}

int main(){
    
    tensor<float> a = tensor<float>(std::vector<int>{7, 3, 5, 2, 11, 2}, "cpu");
    print_vector<int>(a.strides);

    float b[6] = {1, 2, 3, 4, 5, 0.3};
    tensor<float> c = tensor<float>(b, std::vector<int>{2, 3}, "cpu");
    std::cout << c.size << " " << c.data[5] << std::endl;

    tensor<float>& d = *(c.view(std::vector<int>{1, 3, 2}));
    std::cout << d.size << " ";
    print_vector<int>(d.shape);
    print_vector<int>(d.strides);
    d.print_();

    tensor<float>& e = *((d.gpu()->transpose(std::vector<int>{2, 0, 1}))->cpu());
    std::cout << e.size << " ";
    print_vector<int>(e.shape);
    print_vector<int>(e.strides);
    e.print_();

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_real_distribution<float> dist(0, 1);

    float* data = new float[24];
    for(int i=0; i<24; i++)
        data[i] = dist(gen);
    tensor<float> t1 = tensor<float>(data, std::vector<int>{2, 1, 3, 4}, "cpu");
    t1.print_();
    print_vector(t1.shape);
    print_vector(t1.strides);

    tensor<float>& t2 = *t1.view(std::vector<int>{4, 6});
    print_tensor(t2);

    tensor<float>& t3 = *t1.view(std::vector<int>{3, 2, 4, 1})->transpose(std::vector<int>{2, 1, 0, 3});
    print_tensor(t3);
    delete data;
}