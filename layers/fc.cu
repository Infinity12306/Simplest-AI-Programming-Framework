#ifndef FC
#define FC

#include<vector>
#include<cublas_v2.h>
#include<random>
#include "../tensor.cu"

template<typename T>
class fc{
public:
    tensor<T> *W;
    // cublasHandle_t handle;

    fc(int in_feat, int out_feat):W(nullptr), Y(nullptr), dX(nullptr), dW(nullptr),
            fc_X(nullptr), fc_W(nullptr), f_in(in_feat), f_out(out_feat){
        cublasCreate(&handle);
        // define W and initialize it
        W = new tensor<T>(std::vector<int>({f_in, f_out}), "cpu");

        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<T> distribution(0.0f, 1.0f);
        for (int w=0; w<f_out; w++)
            for (int h=0; h<f_in; h++)
                W->data[w*f_in + h] = distribution(generator);
    };

    ~fc(){
        if (W != nullptr)
            delete W;
        if (Y != nullptr){
            delete Y;
            Y = nullptr;
        }
        if (dX != nullptr){
            delete dX;
            dX = nullptr;
        }
        if (dW != nullptr){
            delete dW;
            dW = nullptr;
        }
        cublasDestroy(handle);
    };

    // X: (*, f_in), W: (f_in, f_out), Y(X_r, f_out)
    // Y = X * W
    tensor<T>* forward(tensor<T>* X){
        // if X is a 1-d vector, turn it into 2d matrix
        int dim_x = X->shape.size();
        if (dim_x == 1){
            X->shape.insert(X->shape.begin(), 1);
            dim_x++;
        }
        fc_X = X, fc_W = W;
        X_r = X->shape[dim_x - 2];

        // initialize Y
        std::vector<int> yShape = X->shape;
        yShape[dim_x - 1] = f_out;
        Y = new tensor<T>(yShape, "gpu");

        // calculate bsize
        int iter_num = 1;
        for (int i=0; i<dim_x-2; i++)
            iter_num *= X->shape[i];

        // calculate Y
        T *X_data = X->gpu()->data, *Y_data = Y->gpu()->data;
        W->gpu(); 
        for (int i=0; i<iter_num; i++){
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, X_r, f_out, f_in, &alpha, X_data, X_r, 
                W->data, f_in, &beta, Y_data, X_r);
            X_data += X_r * f_in;
            Y_data += X_r * f_out;
        }
        return Y->cpu();
    };

    std::vector<tensor<T>*> backward(tensor<T>* dY){
        tensor<T>* dX = new tensor<T>(fc_X->shape, "gpu");
        tensor<T>* dW = new tensor<T>(W->shape, "gpu");

        // calculate batch size
        int iternum = 1;
        for (int i=0; i<fc_X->shape.size()-2; i++)
            iternum *= fc_X->shape[i];

        // calculate dL/dX
        // size = (*, X_r, f_in)
        T *dY_data = dY->gpu()->data, *dX_data = dX->gpu()->data;
        W->gpu();
        for (int i=0; i<iternum; i++)
        {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, X_r, f_in, f_out, &alpha, dY_data, X_r, 
                W->data, f_in, &beta, dX_data, X_r);
            dX_data += X_r * f_in;
            dY_data += X_r * f_out;
        }

        // calculate dL/dW
        // size = (f_in, f_out)
        T *X_data = fc_X->data;
        dY_data = dY->data;
        float alpha_w = 1.0f, beta_w = 1.0f;
        for (int i=0; i<iternum; i++)
        {
            alpha_w = 1.0f / (i+1);
            beta_w = (float)i / (i+1);
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, f_in, f_out, X_r, &alpha_w, X_data, X_r, 
                dY_data, X_r, &beta_w, dW->data, f_in);
            X_data += X_r*f_in, dY_data += X_r*f_out;
        }
        return std::vector<tensor<T>*>({dW->cpu(), dX->cpu()});
    };

private:
    tensor<T> *Y, *dX, *dW;
    tensor<T> *fc_X, *fc_W;
    int X_r;
    int f_in, f_out;
    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
};

#endif