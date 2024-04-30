#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "common.h"
#include <mutex>
#include <atomic>

static cublasHandle_t cublas_handle = nullptr;
static std::atomic<int> refCounter(0);
std::mutex mtx;

extern "C" void _init_cublas()
{
    if (cublas_handle == nullptr)
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (cublas_handle == nullptr)
        {
            cublasCheck(cublasCreate(&cublas_handle));
            cublasCheck(cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH));
        }
    }
    refCounter.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void _destory_cublas()
{
    if (cublas_handle != nullptr && refCounter.load(std::memory_order_relaxed)>0)
    {
        refCounter.fetch_sub(1, std::memory_order_relaxed);
        if (refCounter.load(std::memory_order_relaxed) == 0)
        {
            std::lock_guard<std::mutex> lock(mtx);

            if (cublas_handle != nullptr)
            {
                cublasCheck(cublasDestroy(cublas_handle));
                cublas_handle = nullptr;
            }
        }
    }
}

extern "C" struct Mat
{
    float * data;
    size_t size;
};

typedef struct Mat Mat;

extern "C" void to_host(float *out , Mat *m) {
    cudaCheck(cudaMemcpy(out, m->data, m->size * sizeof(float), cudaMemcpyDeviceToHost));
}

extern "C" float * to_device(float *in, size_t size) {
    float *out;
    cudaCheck(cudaMalloc(&out, size * sizeof(float)));  // Allocate GPU memory
    cudaCheck(cudaMemcpy(out, in, size * sizeof(float), cudaMemcpyHostToDevice));  // Corrected size

    return out;
}

extern "C" void cuda_free(float *m) {
    if(m) cudaCheck(cudaFree(m));
}

extern "C" void mat_free(Mat * m) {
    if(m) {
        delete m;
    }
}

extern "C" Mat * matmul_cublas_device(const float *a, const float *b,
                              int m, int n, int k)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float * out_mat;
    cudaCheck(cudaMalloc(&out_mat, m * n * sizeof(float)));
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, a, k, b, n, &beta, out_mat, m));
    
    Mat * mat = new Mat();
    mat->data = out_mat;
    mat->size = m * n * sizeof(float);
    return mat;
}

extern "C" void matmul_cublas(float *out,
                              const float *a, const float *b,
                              int m, int n, int k)
{

    const float alpha = 1.0f;
    const float beta = 0.0f;
    float *a_mat, *b_mat, *out_mat;
    cudaCheck(cudaMalloc(&a_mat, m * k * sizeof(float)));
    cudaCheck(cudaMalloc(&b_mat, n * k * sizeof(float)));
    cudaCheck(cudaMalloc(&out_mat, m * n * sizeof(float)));
    cudaCheck(cudaMemcpy(a_mat, a, m * k * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(b_mat, b, n * k * sizeof(float), cudaMemcpyHostToDevice));

    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, &alpha, a_mat, k, b_mat, n, &beta, out_mat, m));
    cudaCheck(cudaMemcpy(out, out_mat, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    if (a_mat)
        cudaCheck(cudaFree(a_mat));
    if (b_mat)
        cudaCheck(cudaFree(b_mat));
    if (out_mat)
        cudaCheck(cudaFree(out_mat));
}
