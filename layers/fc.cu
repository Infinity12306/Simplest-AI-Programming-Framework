#ifndef FC
#define FC

#include <vector>
#include <cublas_v2.h>
#include <curand.h>
#include <random>

#include "../tensor.cu"
#include "../utils.h"

template<typename T>
class fc{
public:
    fc(int in_feat, int out_feat, cublasHandle_t handle, curandGenerator_t prng):Y(nullptr), dX(nullptr), dW(nullptr),
            fc_X(nullptr), fc_W(nullptr), f_in(in_feat), f_out(out_feat), handle(handle), prng(prng){
        // define W and initialize it
        fc_W = new tensor<T>(std::vector<int>({f_in, f_out}), "gpu");
        curandGenerateUniform(prng, fc_W->data, f_in*f_out);
    };

    ~fc(){
        if (fc_W != nullptr){
            delete fc_W;
            fc_W = nullptr;
        }
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
        fc_X = X;
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
        fc_W->gpu(); 
        for (int i=0; i<iter_num; i++){
            sgemm_wrapper<T>(handle, X_data, fc_W->data, Y_data, X_r, f_out, f_in);
            X_data += X_r * f_in;
            Y_data += X_r * f_out;
        }
        return Y;
    };

    std::vector<tensor<T>*> backward(tensor<T>* dY){
        tensor<T>* dX = new tensor<T>(fc_X->shape, "gpu");
        tensor<T>* dW = new tensor<T>(fc_W->shape, "gpu");

        // calculate batch size
        int iternum = 1;
        for (int i=0; i<fc_X->shape.size()-2; i++)
            iternum *= fc_X->shape[i];

        // calculate dL/dX
        // size = (*, X_r, f_in)
        T *dY_data = dY->gpu()->data, *dX_data = dX->gpu()->data;
        fc_W->gpu();
        for (int i=0; i<iternum; i++)
        {
            sgemm_wrapper<T>(handle, dY_data, fc_W->data, dX_data, X_r, f_in, f_out, 
                            alpha, beta, false, true);
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
            // alpha_w = 1.0f / (i+1);
            // beta_w = (float)i / (i+1);
            sgemm_wrapper<T>(handle, X_data, dY_data, dW->data, f_in, f_out, X_r, 
                            alpha_w, beta_w, true, false);
            X_data += X_r*f_in, dY_data += X_r*f_out;
        }
        return std::vector<tensor<T>*>({dX, dW});
    };

    tensor<T>* get_w(){
        return fc_W;
    }

    tensor<T>* get_dw(){
        return dW;
    }

private:
    tensor<T> *Y, *dX, *dW;
    tensor<T> *fc_X, *fc_W;
    int X_r;
    int f_in, f_out;
    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
    curandGenerator_t prng;
};

#endif