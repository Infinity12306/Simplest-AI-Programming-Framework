#ifndef UTILS_H
#define UTILS_H

const int THREADNUM = 512;
#define GRADTYPE float


inline int GET_BLOCK_NUM(int n){
    return (n + THREADNUM - 1) / THREADNUM;
}
#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

#endif
